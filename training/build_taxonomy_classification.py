#!/usr/bin/env python3
"""Stage B — Taxonomy Classification.

Reads `LAW_RAW_DATA/CRIME_TAXONOMY.json` and emits classification training
pairs from the rich per-category fields:

    - examples              -> "FIR snippet -> category"
    - positive_indicators   -> "Description -> category"
    - negative_indicators   -> "Description -> NOT this category"
    - aliases               -> "Everyday word -> formal category"

The goal is to CALIBRATE the model on the ~150-category ontology so it picks
the legally correct snake_case label.

Output:
    training/datasets/stages_raw/stage_b_taxonomy.jsonl
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

from training.common import (
    RAW_STAGE_DIR,
    TAXONOMY_SYSTEM_PROMPT,
    TrainingRecord,
    category_to_snake,
    fingerprint,
    load_crime_taxonomy,
    short,
    squish,
    write_jsonl,
)


# ---------------------------------------------------------------------------
# Per-category generators
# ---------------------------------------------------------------------------


def _qa_classify_example(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """For each `example`, produce: snippet -> category."""
    out: List[Dict[str, str]] = []
    cat_snake = category_to_snake(entry.get("category", ""))
    if not cat_snake:
        return out
    examples = [squish(e) for e in (entry.get("examples") or []) if e]
    for ex in examples[:3]:
        if len(ex) < 12:
            continue
        out.append({
            "user": (
                f"FIR fact pattern:\n\"{short(ex, 360)}\"\n\n"
                f"Which crime category is this? Reply with one snake_case label."
            ),
            "assistant": cat_snake,
        })
    return out


def _qa_classify_from_positive(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """For each `positive_indicator`, produce: description -> category."""
    out: List[Dict[str, str]] = []
    cat_snake = category_to_snake(entry.get("category", ""))
    if not cat_snake:
        return out
    pos = [squish(e) for e in (entry.get("positive_indicators") or []) if e]
    random.shuffle(pos)
    for p in pos[:2]:
        if len(p) < 12:
            continue
        out.append({
            "user": (
                f"FIR fact pattern:\n\"{short(p, 320)}\"\n\n"
                f"Which crime category is this? Reply with one snake_case label."
            ),
            "assistant": cat_snake,
        })
    return out


def _qa_negative_indicator(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """For each `negative_indicator`, produce: 'Is this <category>?' -> No.

    Critical for teaching the model what is NOT this category, e.g. that
    'death caused without intent' is NOT murder.
    """
    out: List[Dict[str, str]] = []
    cat = (entry.get("category") or "").strip()
    cat_snake = category_to_snake(cat)
    if not cat_snake:
        return out
    neg = [squish(e) for e in (entry.get("negative_indicators") or []) if e]
    random.shuffle(neg)
    for n in neg[:2]:
        if len(n) < 12:
            continue
        out.append({
            "user": (
                f"FIR fact pattern:\n\"{short(n, 320)}\"\n\n"
                f"Does this constitute the offence of '{cat}' (snake: {cat_snake})? "
                f"Answer 'yes' or 'no' and briefly explain."
            ),
            "assistant": (
                f"no — this fact pattern does NOT amount to '{cat}'. "
                f"The ingredient(s) required for '{cat}' are absent here."
            ),
        })
    return out


def _qa_alias(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """alias word -> formal category (e.g. 'snatch' -> snatching)."""
    out: List[Dict[str, str]] = []
    cat = (entry.get("category") or "").strip()
    cat_snake = category_to_snake(cat)
    if not cat_snake:
        return out
    aliases = [squish(a) for a in (entry.get("aliases") or []) if a]
    aliases = [a for a in aliases if 2 <= len(a) <= 40]
    random.shuffle(aliases)
    for a in aliases[:2]:
        readable = a.replace("_", " ")
        out.append({
            "user": (
                f"In an FIR, the complainant uses the word/phrase: \"{readable}\". "
                f"What canonical crime category does this map to? "
                f"Reply with one snake_case label."
            ),
            "assistant": cat_snake,
        })
    return out


def _qa_definition(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """'What is <category> in Indian law?' -> definition."""
    out: List[Dict[str, str]] = []
    cat = (entry.get("category") or "").strip()
    cat_snake = category_to_snake(cat)
    definition = squish(entry.get("definition", ""))
    if not cat or not definition:
        return out
    return [
        {
            "user": f"In Indian criminal law, what is the offence of '{cat}'? Provide its definition.",
            "assistant": (
                f"{definition}\n\n(Canonical taxonomy label: {cat_snake})"
            ),
        }
    ]


def _qa_trigger_conditions(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """List the legal ingredients of this offence."""
    out: List[Dict[str, str]] = []
    cat = (entry.get("category") or "").strip()
    triggers = [squish(t).replace("_", " ") for t in (entry.get("trigger_conditions") or []) if t]
    if not cat or not triggers:
        return out
    bullet = "\n- ".join(triggers[:6])
    return [
        {
            "user": f"What are the essential ingredients (trigger conditions) of '{cat}' under Indian law?",
            "assistant": f"The essential ingredients of '{cat}' are:\n- {bullet}",
        }
    ]


GENERATORS = [
    _qa_classify_example,
    _qa_classify_from_positive,
    _qa_negative_indicator,
    _qa_alias,
    _qa_definition,
    _qa_trigger_conditions,
]


# ---------------------------------------------------------------------------
# Cross-statute disambiguation pairs.
#
# Several offences in the canonical taxonomy refer to the SAME real-world
# act but live in different statutes (POCSO vs POSH vs BNS vs IT Act vs
# SC/ST Act). The eval audit on `fused-1000` showed the model conflates
# these because its training data has no explicit "which one applies in
# situation X" example.
#
# Each entry below is a deliberately-handcrafted FIR snippet plus the
# correct snake_case label. They are emitted as Stage B classification
# pairs and will be replayed many times during SFT, anchoring the
# statute-vs-statute distinction.
# ---------------------------------------------------------------------------
DISAMBIGUATION_PAIRS: List[Dict[str, str]] = [
    # POCSO vs POSH vs BNS — sexual harassment by setting / victim age.
    {"snippet": "Office colleague repeatedly made unwelcome sexual remarks to "
                "the woman complainant during work hours and demanded sexual favours "
                "in exchange for a promotion.",
     "label": "sexual_harassment_at_workplace",
     "rationale": "POSH Act applies — workplace, employment relationship, adult woman."},
    {"snippet": "Stranger on the street made obscene gestures and lewd remarks at "
                "an adult woman walking home; no workplace nexus.",
     "label": "sexual_harassment",
     "rationale": "BNS Section 75 sexual harassment — public place, adult, no workplace."},
    {"snippet": "Tutor made sexually coloured remarks to a 13-year-old girl during "
                "private tuition; no physical contact.",
     "label": "child_sexual_harassment",
     "rationale": "POCSO — victim under 18; non-contact sexually explicit conduct."},

    # POCSO vs IT Act — child pornography production/storage/online.
    {"snippet": "Accused recorded sexually explicit videos of a 14-year-old for "
                "pornographic distribution.",
     "label": "use_of_child_for_pornography",
     "rationale": "POCSO Section 14 — child USED in production of pornographic material."},
    {"snippet": "Accused had child sexual abuse images stored on hard drive at "
                "home; no evidence of further distribution.",
     "label": "storage_of_child_pornography",
     "rationale": "POCSO Section 15 — possession/storage of CSAM."},
    {"snippet": "Accused operated a Telegram channel that publicly transmitted "
                "child sexual abuse videos to subscribers.",
     "label": "child_pornography_online",
     "rationale": "IT Act Section 67B — electronic transmission of CSAM."},

    # BNS vs POCSO — assault on adult vs child.
    {"snippet": "Accused punched and slapped an adult complainant on the road "
                "during a parking dispute.",
     "label": "assault_or_criminal_force",
     "rationale": "BNS — victim is adult; ordinary criminal force."},
    {"snippet": "Accused fondled a 9-year-old child with sexual intent at a "
                "shop; no penetration alleged.",
     "label": "sexual_assault_on_child",
     "rationale": "POCSO Section 7 — non-penetrative sexual touching of a child."},

    # IPC/BNS cruelty vs DV Act abuse.
    {"snippet": "Husband and his mother repeatedly beat the wife and demanded "
                "additional dowry; complaint filed under cognizable offence.",
     "label": "cruelty_by_husband_or_relatives",
     "rationale": "BNS Section 85 — criminal cruelty for dowry by husband/relatives."},
    {"snippet": "Husband repeatedly slapped wife at home over trivial matters; "
                "complainant seeks protection order.",
     "label": "physical_domestic_abuse",
     "rationale": "Protection of Women from Domestic Violence Act — civil-style "
                  "remedy, no dowry nexus."},

    # Bribery: elections vs public servant.
    {"snippet": "Accused offered Rs. 5,000 to voters of his ward to vote for a "
                "particular candidate.",
     "label": "bribery_at_elections",
     "rationale": "BNS — gratification offered to influence voting."},
    {"snippet": "Tax inspector demanded Rs. 50,000 from a businessman to "
                "overlook a compliance violation.",
     "label": "bribery_of_public_servant",
     "rationale": "Prevention of Corruption Act — bribe to public servant for "
                  "official act."},

    # Criminal intimidation vs SC/ST atrocity.
    {"snippet": "Accused threatened the complainant with a knife to drop a civil "
                "suit; no caste reference.",
     "label": "criminal_intimidation",
     "rationale": "BNS — threat of injury without protected-class targeting."},
    {"snippet": "Accused publicly humiliated and threatened a Scheduled Caste "
                "person using caste slurs.",
     "label": "caste_based_insult_or_intimidation_of_sc_st",
     "rationale": "SC/ST (Prevention of Atrocities) Act — caste-targeted insult/threat."},

    # Trespass: house breaking vs general criminal trespass.
    {"snippet": "Accused broke open the front door of complainant's dwelling at "
                "night and entered without permission.",
     "label": "house_trespass_or_house_breaking",
     "rationale": "BNS — entry into a dwelling by breaking; aggravated form."},
    {"snippet": "Accused entered complainant's open agricultural field without "
                "permission and refused to leave.",
     "label": "criminal_trespass",
     "rationale": "BNS — entry into property without consent; not a dwelling."},

    # POCSO aggravated vs ordinary penetrative.
    {"snippet": "School warden committed penetrative sexual assault on a "
                "12-year-old in his care.",
     "label": "aggravated_penetrative_sexual_assault",
     "rationale": "POCSO Section 5 — assault by a person in position of trust = aggravated."},
    {"snippet": "An adult relative committed penetrative sexual assault on a "
                "13-year-old; no aggravating circumstance pleaded beyond age.",
     "label": "penetrative_sexual_assault_on_child",
     "rationale": "POCSO Section 3 — ordinary penetrative assault on a child."},

    # Arms Act vs BNS firearm-aided hurt.
    {"snippet": "Accused was found in possession of an unlicensed pistol and 12 "
                "rounds of ammunition during a routine vehicle check.",
     "label": "unlicensed_acquisition_or_possession_of_firearm",
     "rationale": "Arms Act — possession without licence; no use against person."},

    # NDPS vs IT/cyber.
    {"snippet": "Accused was caught transporting 5kg of contraband heroin from "
                "Delhi to Mumbai by car.",
     "label": "trafficking_narcotic_drugs",
     "rationale": "NDPS Act — narcotic drug trafficking."},

    # Cheating vs personation/online.
    {"snippet": "Accused created a fake bank website mimicking the complainant's "
                "bank and tricked the complainant into entering credentials.",
     "label": "computer_enabled_cheating_by_personation",
     "rationale": "IT Act Section 66D — cheating by personation using a computer resource."},
    {"snippet": "Accused promised to deliver a motorcycle on payment of an "
                "advance, took the money, and never delivered.",
     "label": "cheating",
     "rationale": "BNS — ordinary cheating; no computer-resource personation."},
]


def _qa_disambiguation_pairs() -> List[TrainingRecord]:
    """Hand-curated cross-statute disambiguation pairs.

    These are emitted as Stage B records (single-label classify + a
    follow-up rationale). They run once per `build_records` invocation,
    independent of any taxonomy entry — so they will always appear no
    matter how the canonical taxonomy is sliced upstream.
    """
    out: List[TrainingRecord] = []
    for item in DISAMBIGUATION_PAIRS:
        snippet = squish(item["snippet"])
        label = item["label"].strip()
        rationale = squish(item["rationale"])
        if not snippet or not label:
            continue
        # Variant 1: snippet -> snake_case label only.
        out.append(TrainingRecord(
            stage="B",
            task="taxonomy_disambiguation",
            messages=[
                {"role": "system", "content": TAXONOMY_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"FIR fact pattern:\n\"{snippet}\"\n\n"
                    "Which crime category is this? Reply with one snake_case label "
                    "from the canonical taxonomy."
                )},
                {"role": "assistant", "content": label},
            ],
            meta={
                "category": label,
                "category_snake": label,
                "generator": "_qa_disambiguation_pairs",
                "kind": "label_only",
            },
        ))
        # Variant 2: same snippet -> label + statute rationale.
        out.append(TrainingRecord(
            stage="B",
            task="taxonomy_disambiguation",
            messages=[
                {"role": "system", "content": TAXONOMY_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"FIR fact pattern:\n\"{snippet}\"\n\n"
                    "Identify the precise canonical crime category and explain in "
                    "one sentence why this statute/section applies (vs nearby "
                    "alternatives)."
                )},
                {"role": "assistant", "content": (
                    f"Category: {label}\nReason: {rationale}"
                )},
            ],
            meta={
                "category": label,
                "category_snake": label,
                "generator": "_qa_disambiguation_pairs",
                "kind": "label_with_reason",
            },
        ))
    return out


def build_records(seed: int = 11) -> List[TrainingRecord]:
    random.seed(seed)
    records: List[TrainingRecord] = []
    seen: set[str] = set()

    entries = load_crime_taxonomy()
    print(f"[stage B] loaded {len(entries)} taxonomy categories")

    # Emit hand-curated cross-statute disambiguation pairs FIRST so they
    # win the dedupe race against any near-duplicates from the per-category
    # generators below.
    for rec in _qa_disambiguation_pairs():
        user_msg = rec.messages[1]["content"].strip()
        assistant_msg = rec.messages[2]["content"].strip()
        fp = fingerprint(user_msg, assistant_msg[:120])
        if fp in seen:
            continue
        seen.add(fp)
        records.append(rec)
    print(f"[stage B] disambiguation pairs added: {len(records)}")

    for entry in entries:
        for gen in GENERATORS:
            try:
                qas = gen(entry)
            except Exception as exc:  # noqa: BLE001
                print(f"[stage B] generator {gen.__name__} failed for "
                      f"{entry.get('category')}: {exc}")
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
                        stage="B",
                        task="taxonomy_classify",
                        messages=[
                            {"role": "system", "content": TAXONOMY_SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": assistant_msg},
                        ],
                        meta={
                            "category": entry.get("category", ""),
                            "category_snake": category_to_snake(entry.get("category", "")),
                            "group": entry.get("_group", ""),
                            "unit_id": entry.get("unit_id", ""),
                            "generator": gen.__name__,
                        },
                    )
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage B taxonomy-classification training data.")
    parser.add_argument("--out", type=Path, default=RAW_STAGE_DIR / "stage_b_taxonomy.jsonl")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--sample", type=int, default=0,
                        help="If >0, print this many random records to stdout for inspection")
    args = parser.parse_args()

    records = build_records(seed=args.seed)
    written = write_jsonl(args.out, records)
    print(f"[stage B] wrote {written} records to {args.out}")

    if args.sample > 0 and records:
        import json as _json
        print(f"\n--- {args.sample} random samples ---")
        for s in random.sample(records, min(args.sample, len(records))):
            print(_json.dumps(s.to_dict(), ensure_ascii=False, indent=2))
            print("-" * 60)


if __name__ == "__main__":
    main()
