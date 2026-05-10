#!/usr/bin/env python3
"""Stage C — Synthetic FIR -> events distillation.

For every taxonomy category we ask the teacher LLM (Claude Sonnet 4.5, GPT-4.1,
or Gemini 2.5 Pro) to:

    1. WRITE a realistic Indian FIR text that contains the offence (at given
       difficulty: simple, medium, hard, multi_event).
    2. EMIT the gold events JSON in the EXACT schema our pipeline expects.

We then run the teacher's output through the same `validate_events()` validator
the inference pipeline uses, so any schema-broken example is rejected before
it enters training data.

Cost (approx): 3,000 calls × $0.03 ≈ $90 with Claude Sonnet 4.5.

Usage:
    # Set ONE of these env vars first:
    export ANTHROPIC_API_KEY=...    # OR
    export OPENAI_API_KEY=...       # OR
    export GOOGLE_API_KEY=...

    # Smoke test with 5 categories
    python -m training.build_synthetic_firs --max-categories 5 --variants-per-category 2

    # Full run (~3000 examples; ~30-60 min)
    python -m training.build_synthetic_firs --variants-per-category 20

Output:
    training/datasets/stages_raw/stage_c_synthetic_firs.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running as `python -m training.build_synthetic_firs` from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import (
    RAW_STAGE_DIR,
    SLIM_EXTRACTION_SYSTEM_PROMPT,
    TrainingRecord,
    canonical_categories,
    canonicalize_events,
    category_to_snake,
    fingerprint,
    load_crime_taxonomy,
    short,
    squish,
    write_jsonl,
)
from training.teacher_client import get_teacher_client

# Reuse the production validator so training data and inference use IDENTICAL
# schema rules.
from event_extraction.ontology import (
    CATEGORY_TO_TYPE,
    VALID_CATEGORIES,
)
from event_extraction.validation import validate_events


DIFFICULTIES = ["simple", "medium", "hard", "multi_event"]


TEACHER_INSTRUCTION = """You are an expert Indian legal-data labeller.

TASK: Produce ONE realistic FIR (First Information Report) text in Indian context, then output the GOLD JSON labelling for it.

Constraints:
- Use realistic Indian names, places (cities/districts), dates, INR amounts, vehicle reg formats, phone formats.
- Voice = first-person complainant or third-person narrative, like a real police-station FIR.
- Length: SIMPLE = 60-120 words, MEDIUM = 130-220 words, HARD = 220-350 words with legal nuance, MULTI_EVENT = 280-500 words covering 3-8 distinct offences.
- For HARD: include realistic ambiguity (e.g. theft vs snatching vs robbery boundary, civil-vs-criminal overlap, intent ambiguity).
- For MULTI_EVENT: include __PRIMARY_CATEGORY__ as the principal offence + 2-7 other realistic co-offences.

OUTPUT FORMAT — return ONLY a JSON object (no prose, no markdown fences):

{
  "fir_text": "<the FIR text as a single string>",
  "fir_text_categories": ["<Human Readable Category>"],
  "events": [
    {
      "event_id": 1,
      "crime_category": "<snake_case from the taxonomy>",
      "action_summary": "<4-8 word generic action, NO names/dates/amounts/places/brands>",
      "details": "<short factual description of THIS event>",
      "evidence": "<short verbatim or near-verbatim snippet from the FIR text supporting this event>",
      "actors": "<accused | accused_persons | husband | etc>",
      "confidence": "high"
    }
  ]
}

Rules for events:
- Every distinct criminal act = one event.
- crime_category MUST be snake_case (e.g. "criminal_intimidation").
- action_summary must be GENERIC ("snatched mobile phone from victim hand" not "snatched iPhone 15 from Priya").
- DO NOT invent crime_category values that are not in the canonical taxonomy.
- For non_crime / civil-only fact patterns, set events to [].
- Skip pure procedural events (FIR registration, investigation note, witness listing).

PRIMARY CATEGORY: __PRIMARY_CATEGORY__   (snake: __PRIMARY_CATEGORY_SNAKE__)
TAXONOMY DEFINITION: __PRIMARY_DEFINITION__
TAXONOMY GROUP: __PRIMARY_GROUP__
DIFFICULTY: __DIFFICULTY__

CRITICAL — DO NOT CONFUSE WITH THESE NEAREST-NEIGHBOUR OFFENCES:
__NEIGHBOUR_BLOCK__

When you label crime_category, write the FIR clearly enough that the chosen
category is distinguishable from each of the neighbours above. If the fact
pattern actually fits a neighbour better than the primary category, label it
with the neighbour — accuracy beats sticking to the requested primary.
"""


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first balanced JSON object out of a possibly-noisy reply."""
    if not text:
        return None
    # Strip code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text)
    # Find first { and parse incrementally
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start:i + 1]
                try:
                    return json.loads(snippet)
                except Exception:  # noqa: BLE001
                    return None
    return None


# -------------------------------------------------------------------
# Nearest-neighbour lookup so the teacher can disambiguate similar
# offences (theft / snatching / robbery / dacoity, hurt / grievous_hurt
# / attempt_to_murder, etc.).  Computed once per process from the
# taxonomy.  Same group = same neighbourhood.
# -------------------------------------------------------------------
_NEIGHBOUR_CACHE: Dict[str, List[Dict[str, str]]] = {}


def _build_neighbour_cache() -> None:
    """For every category, precompute its top neighbours (same group)."""
    if _NEIGHBOUR_CACHE:
        return
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for ent in load_crime_taxonomy():
        by_group.setdefault(ent.get("_group", ""), []).append(ent)
    for group, items in by_group.items():
        for ent in items:
            cat_snake = category_to_snake(ent.get("category", ""))
            others = [o for o in items if o is not ent]
            # Keep up to 6 neighbours per category for prompt budget
            neighbours = []
            for nb in others[:6]:
                neighbours.append({
                    "category": nb.get("category", ""),
                    "snake": category_to_snake(nb.get("category", "")),
                    "key_difference": short(nb.get("definition", ""), 220),
                })
            _NEIGHBOUR_CACHE[cat_snake] = neighbours


def _format_neighbour_block(cat_snake: str) -> str:
    _build_neighbour_cache()
    nbs = _NEIGHBOUR_CACHE.get(cat_snake, [])
    if not nbs:
        return "(no nearest neighbours in this group)"
    lines: List[str] = []
    for nb in nbs:
        lines.append(f"- '{nb['category']}' (snake: {nb['snake']}): {nb['key_difference']}")
    return "\n".join(lines)


def _build_user_prompt(entry: Dict[str, Any], difficulty: str) -> str:
    cat = entry.get("category", "")
    cat_snake = category_to_snake(cat)
    out = TEACHER_INSTRUCTION
    out = out.replace("__PRIMARY_CATEGORY__", cat)
    out = out.replace("__PRIMARY_CATEGORY_SNAKE__", cat_snake)
    out = out.replace("__PRIMARY_DEFINITION__", short(entry.get("definition", ""), 600))
    out = out.replace("__PRIMARY_GROUP__", entry.get("_group", ""))
    out = out.replace("__DIFFICULTY__", difficulty)
    out = out.replace("__NEIGHBOUR_BLOCK__", _format_neighbour_block(cat_snake))
    return out


def _coerce_fir_text_categories(events: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for ev in events:
        label = CATEGORY_TO_TYPE.get(ev.get("crime_category", ""))
        if label and label not in out:
            out.append(label)
    return out


def _validate_teacher_output(payload: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Returns (ok, reason, normalized_payload)."""
    if not isinstance(payload, dict):
        return False, "not a dict", {}
    fir_text = payload.get("fir_text") or ""
    if not isinstance(fir_text, str) or len(fir_text.strip()) < 30:
        return False, "fir_text missing or too short", {}

    raw_events = payload.get("events") or []
    if not isinstance(raw_events, list):
        return False, "events not a list", {}

    # Run through production validator — same rules as inference
    validated = validate_events(raw_events, fir_text=fir_text)

    # ---------------------------------------------------------------
    # Coerce every event into the official canonical taxonomy.
    #
    # `validate_events` permits BFF LEGACY_ALIASES (e.g. `assault`); SFT
    # cannot tolerate those because the alias is not a real taxonomy label.
    # `canonicalize_events` walks our LAW_RAW_DATA/TAXONOMY_ALIASES.json
    # map; events whose category cannot be resolved are dropped (better to
    # ship one fewer event than a label the model can never reuse).
    # ---------------------------------------------------------------
    canonicalized = canonicalize_events(validated)
    canon_set = canonical_categories()

    # Allow zero events ONLY if teacher explicitly said it's a non-crime case
    fir_categories = payload.get("fir_text_categories") or []
    if not canonicalized and "Non-Criminal / Civil Matter" not in fir_categories:
        return False, "no valid events extracted", {}

    # Final hard check: every label must now be canonical taxonomy.
    for ev in canonicalized:
        cat = ev.get("crime_category", "")
        if cat and cat not in canon_set:
            return False, f"non-canonical category survived alias map: {cat!r}", {}

    if not fir_categories and canonicalized:
        fir_categories = _coerce_fir_text_categories(canonicalized)

    return True, "", {
        "fir_text": fir_text.strip(),
        "fir_text_categories": fir_categories,
        "events": canonicalized,
    }


def _make_training_record(payload: Dict[str, Any], meta: Dict[str, Any]) -> TrainingRecord:
    """Convert validated teacher output into an SFT training record.

    The `assistant` content matches EXACTLY what we want our fine-tuned model
    to output at inference time: a single JSON object with fir_text_categories
    + events.
    """
    user_msg = f"FIR:\n{payload['fir_text']}"
    assistant_obj = {
        "fir_text_categories": payload["fir_text_categories"],
        "events": payload["events"],
    }
    assistant_msg = json.dumps(assistant_obj, ensure_ascii=False, indent=2)
    return TrainingRecord(
        stage="C",
        task="fir_event_extraction",
        messages=[
            {"role": "system", "content": SLIM_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        meta=meta,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill a teacher LLM into FIR -> events training pairs.")
    parser.add_argument("--out", type=Path, default=RAW_STAGE_DIR / "stage_c_synthetic_firs.jsonl")
    parser.add_argument("--variants-per-category", type=int, default=20,
                        help="Number of FIR variants generated per taxonomy category. "
                             "Total ≈ 150 categories × N. Default 20 -> ~3000 examples.")
    parser.add_argument("--max-categories", type=int, default=0,
                        help="If >0, only run on the first N taxonomy categories (smoke test).")
    parser.add_argument("--provider", choices=["anthropic", "openai", "gemini"], default=None,
                        help="Override teacher provider (default: auto-detect from env).")
    parser.add_argument("--model", default=None,
                        help="Override teacher model id (e.g. claude-sonnet-4-5, gpt-4.1).")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--resume", action="store_true",
                        help="Skip categories already produced in the output file.")
    parser.add_argument("--include-non-crime", action="store_true",
                        help="Also generate non_crime fact patterns (civil disputes, "
                             "procedural-only narratives) for hard-negative coverage.")
    parser.add_argument("--throttle-ms", type=int, default=0,
                        help="Sleep N ms between calls to respect rate limits.")
    args = parser.parse_args()

    if args.model:
        import os
        os.environ["TEACHER_MODEL"] = args.model

    random.seed(args.seed)

    entries = load_crime_taxonomy()
    if not args.include_non_crime:
        entries = [e for e in entries if e.get("_group") != "non_crime"]
    if args.max_categories > 0:
        entries = entries[: args.max_categories]
    print(f"[stage C] running over {len(entries)} categories × "
          f"{args.variants_per_category} variants = "
          f"~{len(entries) * args.variants_per_category} target examples")

    client = get_teacher_client(provider=args.provider)
    print(f"[stage C] teacher = {client.provider}:{client.model}")

    # Load existing output for --resume
    existing: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    if args.resume and args.out.exists():
        with args.out.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                existing.append(rec)
                meta = rec.get("meta", {})
                seen_keys.add(f"{meta.get('category_snake')}:{meta.get('difficulty')}:{meta.get('variant')}")
        print(f"[stage C] resuming with {len(existing)} existing records")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_fh = args.out.open("a" if args.resume else "w", encoding="utf-8")
    written = len(existing)
    failures = 0
    t0 = time.time()

    try:
        for entry in entries:
            cat = entry.get("category", "")
            cat_snake = category_to_snake(cat)
            for variant in range(args.variants_per_category):
                difficulty = DIFFICULTIES[variant % len(DIFFICULTIES)]
                key = f"{cat_snake}:{difficulty}:{variant}"
                if key in seen_keys:
                    continue

                user_prompt = _build_user_prompt(entry, difficulty)
                try:
                    raw = client.complete(
                        system="You output only valid JSON. No prose, no markdown.",
                        user=user_prompt,
                        max_tokens=2000,
                        temperature=0.7 if difficulty != "simple" else 0.4,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[stage C] teacher call failed for {key}: {exc}")
                    failures += 1
                    continue

                parsed = _extract_json_object(raw)
                if parsed is None:
                    print(f"[stage C] could not parse JSON for {key}")
                    failures += 1
                    continue

                ok, reason, normalized = _validate_teacher_output(parsed)
                if not ok:
                    print(f"[stage C] rejected {key}: {reason}")
                    failures += 1
                    continue

                meta = {
                    "category": cat,
                    "category_snake": cat_snake,
                    "group": entry.get("_group", ""),
                    "difficulty": difficulty,
                    "variant": variant,
                    "teacher_provider": client.provider,
                    "teacher_model": client.model,
                }
                rec = _make_training_record(normalized, meta)
                out_fh.write(json.dumps(rec.to_dict(), ensure_ascii=False))
                out_fh.write("\n")
                out_fh.flush()
                written += 1
                seen_keys.add(key)

                if written % 25 == 0:
                    elapsed = time.time() - t0
                    rate = written / max(1.0, elapsed)
                    print(f"[stage C] {written} written  |  "
                          f"{failures} failed  |  {rate:.2f}/s  |  "
                          f"latest: {key}")

                if args.throttle_ms > 0:
                    time.sleep(args.throttle_ms / 1000.0)
    finally:
        out_fh.close()

    print(f"[stage C] done. wrote {written} records, {failures} failures, "
          f"elapsed {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
