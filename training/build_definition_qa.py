#!/usr/bin/env python3
"""Stage A — Definition QA Injection.

Reads every term from `LAW_RAW_DATA/DEFINITIONS/*.json` and emits ~5-7 QA
examples per term, leveraging the rich schema-v3 fields:

    - definition          (formal legal text)
    - plain_english       (lay-person explanation)
    - real_world_examples (concrete Indian context examples)
    - what_it_covers      (positive scope bullets)
    - what_it_excludes    (negative scope bullets)
    - why_it_matters      (legal significance)

The goal is to TEACH the model Indian legal vocabulary so it stops mistaking
"laptop" for not-a-computer-under-IT-Act.

Output:
    training/datasets/stages_raw/stage_a_definitions.jsonl
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from training.common import (
    DEFINITION_SYSTEM_PROMPT,
    RAW_STAGE_DIR,
    TrainingRecord,
    fingerprint,
    load_all_definitions,
    short,
    squish,
    write_jsonl,
)


# ---------------------------------------------------------------------------
# QA template generators.  Each yields zero or more (user, assistant) pairs.
# ---------------------------------------------------------------------------


def _qa_define_term(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """'What does X mean under Act Y?'"""
    term = entry.get("term", "").strip()
    law = entry.get("_law_name", "")
    section = entry.get("section_id", "")
    plain = squish(entry.get("plain_english", ""))
    legal = squish(entry.get("definition", ""))

    if not term or not (plain or legal):
        return []

    user_variants = [
        f"Under the {law}, what does '{term}' mean?",
        f"Define the term '{term}' as used in the {law}.",
        f"What is meant by '{term}' in {law}, Section {section}?" if section else None,
    ]
    user = random.choice([u for u in user_variants if u])

    assistant_parts: List[str] = []
    if plain:
        assistant_parts.append(f"In plain terms: {plain}")
    if legal:
        assistant_parts.append(f"Legally: {legal}")
    if section:
        assistant_parts.append(f"(Source: {law}, Section {section}.)")
    elif law:
        assistant_parts.append(f"(Source: {law}.)")

    return [
        {
            "user": user,
            "assistant": short("\n\n".join(assistant_parts), max_chars=1200),
        }
    ]


def _qa_does_x_qualify(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """'Is <example> a <term>?' → Yes/No grounded in covers/excludes."""
    term = entry.get("term", "").strip()
    law = entry.get("_law_name", "")
    section = entry.get("section_id", "")
    out: List[Dict[str, str]] = []

    covers = [squish(c) for c in (entry.get("what_it_covers") or []) if c]
    excludes = [squish(c) for c in (entry.get("what_it_excludes") or []) if c]
    examples = [squish(e) for e in (entry.get("real_world_examples") or []) if e]

    section_ref = f", Section {section}" if section else ""
    citation = f"({law}{section_ref}.)" if law else ""

    # POSITIVE: pick up to 2 covers OR examples
    pos_pool = covers[:3] + examples[:3]
    random.shuffle(pos_pool)
    for snippet in pos_pool[:2]:
        if not snippet:
            continue
        # Build a YES question from the snippet
        out.append({
            "user": (
                f"Read this scenario:\n\"{short(snippet, 360)}\"\n\n"
                f"Does this fall within '{term}' under the {law}?"
            ),
            "assistant": (
                f"Yes — this clearly falls within the meaning of '{term}'. "
                f"The scope of '{term}' includes: {short(snippet, 360)} "
                f"{citation}".strip()
            ),
        })

    # NEGATIVE: pick up to 2 excludes
    for snippet in excludes[:2]:
        if not snippet:
            continue
        out.append({
            "user": (
                f"Read this scenario:\n\"{short(snippet, 360)}\"\n\n"
                f"Does this fall within '{term}' under the {law}?"
            ),
            "assistant": (
                f"No — this is expressly outside the meaning of '{term}'. "
                f"Excluded: {short(snippet, 360)} {citation}".strip()
            ),
        })

    return out


def _qa_why_it_matters(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    term = entry.get("term", "").strip()
    law = entry.get("_law_name", "")
    why = squish(entry.get("why_it_matters", ""))
    if not term or not why:
        return []
    return [
        {
            "user": f"Why is the legal definition of '{term}' important under the {law}?",
            "assistant": short(why, max_chars=1200),
        }
    ]


def _qa_alias_resolution(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """Map everyday wording -> legal term ('laptop' -> 'computer' (IT Act))."""
    term = entry.get("term", "").strip()
    law = entry.get("_law_name", "")
    examples = [squish(e) for e in (entry.get("real_world_examples") or []) if e]
    out: List[Dict[str, str]] = []
    for ex in examples[:1]:
        if not ex or len(ex) < 30:
            continue
        out.append({
            "user": (
                f"In an FIR, the complainant wrote: \"{short(ex, 300)}\"\n"
                f"Which formal legal term in the {law} captures this?"
            ),
            "assistant": (
                f"The relevant legal term is '{term}' as defined in the {law}. "
                f"This scenario fits because it matches the statutory meaning of '{term}'."
            ),
        })
    return out


def _qa_excludes_explainer(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """'What is NOT included in the meaning of <term>?'"""
    term = entry.get("term", "").strip()
    law = entry.get("_law_name", "")
    excludes = [squish(c) for c in (entry.get("what_it_excludes") or []) if c]
    if not term or not excludes:
        return []
    bullet = "\n- ".join(short(e, 220) for e in excludes[:4])
    return [
        {
            "user": f"What is expressly NOT covered by the term '{term}' under the {law}?",
            "assistant": (
                f"'{term}' under the {law} does NOT cover:\n- {bullet}"
            ),
        }
    ]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
GENERATORS = [
    _qa_define_term,
    _qa_does_x_qualify,
    _qa_why_it_matters,
    _qa_alias_resolution,
    _qa_excludes_explainer,
]


def build_records(seed: int = 7) -> List[TrainingRecord]:
    random.seed(seed)
    records: List[TrainingRecord] = []
    seen: set[str] = set()

    entries = load_all_definitions()
    print(f"[stage A] loaded {len(entries)} definitions across all acts")

    for entry in entries:
        for gen in GENERATORS:
            try:
                qas = gen(entry)
            except Exception as exc:  # noqa: BLE001
                print(f"[stage A] generator {gen.__name__} failed for "
                      f"{entry.get('_law_id')}::{entry.get('term')}: {exc}")
                continue
            for qa in qas:
                user_msg = qa["user"].strip()
                assistant_msg = qa["assistant"].strip()
                if not user_msg or not assistant_msg:
                    continue
                fp = fingerprint(user_msg, assistant_msg[:120])
                if fp in seen:
                    continue
                seen.add(fp)
                records.append(
                    TrainingRecord(
                        stage="A",
                        task="definition_qa",
                        messages=[
                            {"role": "system", "content": DEFINITION_SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": assistant_msg},
                        ],
                        meta={
                            "law_id": entry.get("_law_id", ""),
                            "term": entry.get("term", ""),
                            "section_id": entry.get("section_id", ""),
                            "generator": gen.__name__,
                        },
                    )
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage A definition-QA training data.")
    parser.add_argument("--out", type=Path, default=RAW_STAGE_DIR / "stage_a_definitions.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample", type=int, default=0,
                        help="If >0, print this many random records to stdout for inspection")
    args = parser.parse_args()

    records = build_records(seed=args.seed)
    written = write_jsonl(args.out, records)
    print(f"[stage A] wrote {written} records to {args.out}")

    if args.sample > 0 and records:
        print(f"\n--- {args.sample} random samples for inspection ---")
        sample = random.sample(records, min(args.sample, len(records)))
        import json as _json
        for s in sample:
            print(_json.dumps(s.to_dict(), ensure_ascii=False, indent=2))
            print("-" * 60)


if __name__ == "__main__":
    main()
