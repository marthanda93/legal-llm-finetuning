#!/usr/bin/env python3
"""Stage E — Definition-Grounded Extraction.

This is the BRIDGE between Stage A (definitions in isolation) and Stage C
(extraction in isolation).

Stage A teaches: "computer (IT Act) covers laptop, smartphone, server"
Stage C teaches: "FIR with hacked laptop -> events list"

Stage E EXPLICITLY teaches:
    Input  = FIR snippet + the relevant definition
    Output = events list that explicitly cites the definition

So at inference time the model knows to USE retrieved definitions when its
input includes them (which is exactly what the BFF pipeline does in Stage 2
of the production fir_section_pipeline.py).

We map taxonomy GROUPS to source ACTS:
    cyber_offences           -> IT_ACT
    data_protection_offences -> DPDP_ACT
    narcotics_offences       -> NDPS
    arms_offences            -> ARMS
    domestic_violence_offences -> DV
    sexual_harassment_offences -> POSH
    corruption_offences      -> PREVENTION_OF_CORRUPTION
    sc_st_offences           -> SC_ST
    food_safety_offences     -> FOOD_SAFETY_AND_STANDARDS
    intellectual_property_offences -> COPYRIGHT
    traffic_offences         -> MV
    labour_offences          -> LABOUR / PAYMENT_OF_WAGES
    environmental_offences   -> ENVIRONMENT_PROTECTION
    family_offences / domestic_violence_offences -> DV / DOWRY_PROHIBITION

For each taxonomy entry in those groups, we:
    1. Pick its top examples / positive_indicators as the FIR snippet.
    2. Pull the most relevant definition(s) from the matching act.
    3. Emit a training pair:
        user      = "Definition: ...\nFIR: ...\nExtract events."
        assistant = JSON events with explicit definition citation.

No teacher LLM is needed — the supervision signal lives entirely in the
existing CRIME_TAXONOMY.json + DEFINITIONS/*.json.

Output:
    training/datasets/stages_raw/stage_e_definition_grounded.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from training.common import (
    RAW_STAGE_DIR,
    SLIM_EXTRACTION_SYSTEM_PROMPT,
    TrainingRecord,
    canonical_law_name,
    category_to_snake,
    fingerprint,
    load_all_definitions,
    load_crime_taxonomy,
    short,
    squish,
    write_jsonl,
)
from event_extraction.ontology import VALID_CATEGORIES, CATEGORY_TO_TYPE


# Map taxonomy group -> primary law_id(s) we should pull definitions from.
GROUP_TO_LAW_IDS: Dict[str, List[str]] = {
    "cyber_offences": ["IT_ACT"],
    "data_protection_offences": ["DPDP_ACT", "IT_ACT"],
    "narcotics_offences": ["NDPS"],
    "arms_offences": ["ARMS"],
    "domestic_violence_offences": ["DV", "DOWRY_PROHIBITION"],
    "family_offences": ["DV", "DOWRY_PROHIBITION", "BNS"],
    "sexual_harassment_offences": ["POSH"],
    "child_protection_offences": ["POCSO", "JUVENILE_JUSTICE"],
    "juvenile_justice_offences": ["JUVENILE_JUSTICE"],
    "corruption_offences": ["PREVENTION_OF_CORRUPTION"],
    "financial_concealment": ["PREVENTION_OF_MONEY_LAUNDERING"],
    "sc_st_offences": ["SC_ST"],
    "national_security_offences": ["UAPA"],
    "food_safety_offences": ["FOOD_SAFETY_AND_STANDARDS"],
    "intellectual_property_offences": ["COPYRIGHT"],
    "traffic_offences": ["MV"],
    "labour_offences": ["LABOUR", "PAYMENT_OF_WAGES"],
    "environmental_offences": ["ENVIRONMENT_PROTECTION"],
    "registration_offences": ["REGISTRATION"],
    # The big BNS-derived groups also benefit from BNS definitions
    "violent_offences": ["BNS"],
    "sexual_offences": ["BNS", "POCSO"],
    "property_offences": ["BNS"],
    "economic_offences": ["BNS"],
    "public_order": ["BNS"],
    "state_and_public_authority_offences": ["BNS"],
    "animal_offences": ["BNS"],
}


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{2,}")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _build_definition_index(law_ids: List[str]) -> List[Dict[str, Any]]:
    """Pull definitions for the requested act(s)."""
    all_defs = load_all_definitions()
    out: List[Dict[str, Any]] = []
    wanted = {lid.upper() for lid in law_ids}
    for d in all_defs:
        if d.get("_law_id", "").upper() in wanted:
            tokens = _tokens(d.get("term", "") + " " +
                            d.get("plain_english", "") + " " +
                            " ".join(d.get("semantic_tags", []) or []))
            d["_tokens"] = tokens
            out.append(d)
    return out


def _best_definitions_for(tax_entry: Dict[str, Any],
                          def_index: List[Dict[str, Any]],
                          k: int = 2) -> List[Dict[str, Any]]:
    """Pick the top-k definitions whose token overlap with this taxonomy
    entry is highest — i.e. the definitions most likely to be relevant.
    Falls back to lower thresholds if nothing matches strongly."""
    cat_tokens = _tokens(
        tax_entry.get("category", "") + " " +
        tax_entry.get("definition", "") + " " +
        " ".join(tax_entry.get("aliases", []) or []) + " " +
        " ".join(tax_entry.get("trigger_conditions", []) or []) + " " +
        " ".join(tax_entry.get("positive_indicators", []) or [])
    )
    scored: List[tuple[int, Dict[str, Any]]] = []
    for d in def_index:
        overlap = len(cat_tokens & d["_tokens"])
        if overlap >= 1:
            scored.append((overlap, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        return [d for _, d in scored[:k]]
    # Fallback: any definition from this act (random) so we still teach
    # the model the FIR-with-definition input shape
    return def_index[:k] if def_index else []


def _format_definition_block(defn: Dict[str, Any]) -> str:
    term = defn.get("term", "")
    section = defn.get("section_id", "")
    law = canonical_law_name(defn.get("_law_id", ""), fallback=defn.get("_law_name", ""))
    plain = squish(defn.get("plain_english", ""))
    legal = squish(defn.get("definition", ""))
    covers = [squish(c) for c in (defn.get("what_it_covers") or [])][:3]
    excludes = [squish(c) for c in (defn.get("what_it_excludes") or [])][:2]

    parts = [f"DEFINITION '{term}' ({law}, Section {section}):"]
    if plain:
        parts.append(f"  Plain meaning: {short(plain, 320)}")
    if legal:
        parts.append(f"  Legal text: {short(legal, 320)}")
    if covers:
        parts.append("  Covers: " + " | ".join(short(c, 100) for c in covers))
    if excludes:
        parts.append("  Excludes: " + " | ".join(short(c, 100) for c in excludes))
    return "\n".join(parts)


def _build_record_from_example(tax_entry: Dict[str, Any],
                               example: str,
                               definitions: List[Dict[str, Any]]) -> Optional[TrainingRecord]:
    cat = tax_entry.get("category", "")
    cat_snake = category_to_snake(cat)
    if cat_snake not in VALID_CATEGORIES:
        return None
    if not example or len(example) < 20:
        return None

    # Build the FIR shell around the bare example so it reads like a
    # police-station narrative rather than a one-line tag.
    fir_text = (
        f"I, the complainant, state that the following incident occurred. {example} "
        "I request appropriate legal action to be taken under applicable law."
    )

    def_blocks = "\n\n".join(_format_definition_block(d) for d in definitions)
    user_msg = (
        f"{def_blocks}\n\n"
        f"FIR:\n{fir_text}\n\n"
        f"Using the definitions above, extract events. Cite the definition that justifies "
        f"each crime_category in the 'details' field."
    )

    # The grounded answer cites the definition explicitly.
    cited_term = definitions[0].get("term", "") if definitions else ""
    cited_law = canonical_law_name(definitions[0].get("_law_id", "")) if definitions else ""
    cited_section = definitions[0].get("section_id", "") if definitions else ""

    fir_text_categories = []
    label = CATEGORY_TO_TYPE.get(cat_snake)
    if label:
        fir_text_categories = [label]

    assistant_obj = {
        "fir_text_categories": fir_text_categories,
        "events": [
            {
                "event_id": 1,
                "crime_category": cat_snake,
                "action_summary": short(cat.lower(), 60),
                "details": (
                    f"This act constitutes '{cat}' "
                    + (f"because the definition of '{cited_term}' under {cited_law}"
                       + (f" §{cited_section}" if cited_section else "")
                       + " applies to this fact pattern."
                       if cited_term else f"under Indian criminal law.")
                ),
                "evidence": short(example, 280),
                "actors": "accused",
                "confidence": "high",
            }
        ],
    }
    assistant_msg = json.dumps(assistant_obj, ensure_ascii=False, indent=2)

    return TrainingRecord(
        stage="E",
        task="definition_grounded_extraction",
        messages=[
            {"role": "system", "content": SLIM_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        meta={
            "category": cat,
            "category_snake": cat_snake,
            "group": tax_entry.get("_group", ""),
            "cited_term": cited_term,
            "cited_law": cited_law,
            "cited_section": cited_section,
            "n_definitions": len(definitions),
        },
    )


def build_records(seed: int = 17) -> List[TrainingRecord]:
    rng = random.Random(seed)
    records: List[TrainingRecord] = []
    seen_fp: set[str] = set()

    # Pre-build per-group definition index to avoid repeated scans
    group_def_index: Dict[str, List[Dict[str, Any]]] = {}
    for group, law_ids in GROUP_TO_LAW_IDS.items():
        group_def_index[group] = _build_definition_index(law_ids)

    for tax_entry in load_crime_taxonomy():
        group = tax_entry.get("_group", "")
        if group not in GROUP_TO_LAW_IDS:
            continue
        if group == "non_crime":
            continue

        defs = group_def_index.get(group, [])
        if not defs:
            continue
        best = _best_definitions_for(tax_entry, defs, k=2)
        if not best:
            continue

        # Use up to 6 examples / positive_indicators per category
        snippets = [squish(e) for e in (tax_entry.get("examples") or []) if e]
        snippets += [squish(e) for e in (tax_entry.get("positive_indicators") or []) if e]
        snippets = [s for s in snippets if len(s) >= 20]
        rng.shuffle(snippets)

        for snippet in snippets[:6]:
            rec = _build_record_from_example(tax_entry, snippet, best)
            if rec is None:
                continue
            # Fingerprint on (category, snippet) so the same FIR snippet
            # under the same crime never appears twice but different
            # categories sharing a definition do not collide.
            fp = fingerprint(category_to_snake(tax_entry.get("category", "")), snippet)
            if fp in seen_fp:
                continue
            seen_fp.add(fp)
            records.append(rec)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage E definition-grounded extraction data.")
    parser.add_argument("--out", type=Path, default=RAW_STAGE_DIR / "stage_e_definition_grounded.jsonl")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    records = build_records(seed=args.seed)
    written = write_jsonl(args.out, records)
    print(f"[stage E] wrote {written} definition-grounded records to {args.out}")

    if args.sample > 0 and records:
        for r in random.sample(records, min(args.sample, len(records))):
            print("=" * 70)
            for m in r.messages:
                print(f"--- {m['role'].upper()} ---")
                print(m["content"][:900])
                print()


if __name__ == "__main__":
    main()
