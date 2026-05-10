"""Shared event ontology for the FIR extraction pipeline.

VALID_CATEGORIES and CATEGORY_TO_TYPE are the canonical labels the model is
allowed to emit. They are derived from the SOURCE OF TRUTH taxonomy file at
LAW_RAW_DATA/CRIME_TAXONOMY.json so that:

    1. Every legal offence enumerated in the taxonomy is a valid extraction
       label (previously the hand-curated set covered only ~38 % of the
       taxonomy, silently dropping 178 valid categories at inference time).
    2. fir_text_categories rolls up to the taxonomy's group label, so a
       single source-of-truth drives both fine-grained labels and high-level
       grouping.

A small LEGACY_ALIASES table maps a handful of older names that appear in the
hand-written few-shot dataset to their canonical taxonomy equivalents, so
backward compatibility is preserved.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TAXONOMY_PATH = _REPO_ROOT / "LAW_RAW_DATA" / "CRIME_TAXONOMY.json"
_TAXONOMY_ALIASES_PATH = _REPO_ROOT / "LAW_RAW_DATA" / "TAXONOMY_ALIASES.json"


def _category_to_snake(category: str) -> str:
    cat = (category or "").strip().lower()
    cat = re.sub(r"[^a-z0-9]+", "_", cat)
    cat = re.sub(r"_+", "_", cat).strip("_")
    return cat


# Group label (from taxonomy keys) -> human-readable display category used in
# the `fir_text_categories` field of the extractor output. These display
# strings appear in the BFF API response and the frontend filters.
GROUP_TO_TYPE: Dict[str, str] = {
    "violent_offences": "Physical Violence / Assault",
    "property_offences": "Theft / Robbery / Property Crime",
    "sexual_offences": "Sexual Violence",
    "economic_offences": "Fraud / Financial Crime",
    "family_offences": "Family / Marriage Offences",
    "public_order": "Public Order / Riot / Unlawful Assembly",
    "child_protection_offences": "Child Protection / POCSO",
    "national_security_offences": "National Security / Terrorism",
    "traffic_offences": "Traffic / Motor Vehicle Offences",
    "corruption_offences": "Corruption / Bribery / Official Misconduct",
    "financial_concealment": "Money Laundering / Financial Concealment",
    "labour_offences": "Labour / Employment Offences",
    "food_safety_offences": "Food Safety / Adulteration",
    "intellectual_property_offences": "Intellectual Property / Copyright",
    "cyber_offences": "Cyber Crime / IT Act",
    "data_protection_offences": "Data Protection / Privacy",
    "narcotics_offences": "Narcotics / Drug Offences",
    "arms_offences": "Arms / Explosives",
    "domestic_violence_offences": "Domestic Violence / Dowry",
    "sexual_harassment_offences": "Sexual Harassment / Workplace",
    "sc_st_offences": "SC/ST Atrocity",
    "juvenile_justice_offences": "Juvenile Justice",
    "environmental_offences": "Environmental Offences",
    "registration_offences": "Registration / Document Fraud",
    "animal_offences": "Animal Offences",
    "state_and_public_authority_offences": "Offences Against State / Public Authority",
    "personal_liberty": "Kidnapping / Abduction / Wrongful Confinement",
    "inchoate": "Conspiracy / Attempt / Abetment",
    "obstruction_of_justice": "Obstruction of Justice / False Evidence",
}


def _iter_taxonomy_categories() -> Iterable[Tuple[str, str]]:
    """Yield (category_snake, group) for every offence in the taxonomy."""
    if not _TAXONOMY_PATH.exists():
        return
    with _TAXONOMY_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    for group, items in raw.items():
        if group == "non_crime":
            continue
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            cat = (it.get("category") or "").strip()
            if cat:
                yield _category_to_snake(cat), group


# Build VALID_CATEGORIES + CATEGORY_TO_TYPE from the taxonomy.
VALID_CATEGORIES: Set[str] = set()
CATEGORY_TO_TYPE: Dict[str, str] = {}
for _cat, _group in _iter_taxonomy_categories():
    VALID_CATEGORIES.add(_cat)
    CATEGORY_TO_TYPE[_cat] = GROUP_TO_TYPE.get(
        _group, _group.replace("_", " ").title()
    )


# Legacy aliases — names used in the hand-written `few_shot_database.py` and
# old client integrations. They are accepted as input but normalized to the
# canonical taxonomy category. The validator applies this mapping before the
# membership check.
LEGACY_ALIASES: Dict[str, str] = {
    # violence
    "voluntarily_causing_hurt": "hurt",
    "hurt_by_dangerous_means": "hurt_or_grievous_hurt_by_dangerous_weapons",
    # traffic
    "rash_driving": "dangerous_or_rash_driving",
    "drunk_driving": "impaired_driving",
    "reckless_driving": "dangerous_or_rash_driving",
    "dangerous_driving": "dangerous_or_rash_driving",
    # cyber
    "cyber_fraud": "computer_enabled_cheating_by_personation",
    "cyber_stalking": "stalking",
    "ransomware_attack": "computer_system_abuse",
    "publishing_obscene_content": "publication_of_obscene_electronic_content",
    "unauthorized_data_sharing": "data_breach_non_notification",
    "failure_to_protect_personal_data": "data_safeguards_or_data_rights_violation",
    "personal_data_breach": "data_breach_non_notification",
    "data_breach": "data_breach_non_notification",
    "unauthorized_access": "unauthorized_access_to_computer_system",
    "identity_theft": "identity_theft_via_computer",
    "phishing": "computer_enabled_cheating_by_personation",
    # child
    "child_grooming": "child_sexual_offence_abetment",
    "child_pornography": "use_of_child_for_pornography",
    "child_labour": "child_labour_violation",
    "child_labour_exploitation": "child_labour_violation",
    # arms
    "illegal_arms_trade": "unlicensed_manufacture_sale_or_repair_of_arms",
    "illegal_firing": "arms_possession_or_use_for_criminal_purposes",
    "unlicensed_arms_possession": "unlicensed_acquisition_or_possession_of_firearm",
    "arms_act_violation": "unlicensed_acquisition_or_possession_of_firearm",
    # corruption
    "abuse_of_official_position": "public_servant_misconduct",
    "bribery": "bribery_of_public_servant",
    "corruption": "public_servant_misconduct",
    # financial
    "concealment_of_proceeds_of_crime": "money_laundering",
    # ndps
    "drug_consumption": "consumption_of_narcotic_drugs",
    "commercial_quantity_drug_possession": "possession_of_narcotic_drugs_or_psychotropic_substances",
    "ndps_possession": "possession_of_narcotic_drugs_or_psychotropic_substances",
    "drug_trafficking": "trafficking_narcotic_drugs",
    # sexual harassment
    "hostile_workplace_environment": "sexual_harassment",
    # registration
    "forged_sale_deed": "false_statement_or_personation_in_document_registration",
    "fraudulent_property_registration": "incorrect_endorsement_or_registration_with_intent_to_injure",
    "fraudulent_multiple_registration": "incorrect_endorsement_or_registration_with_intent_to_injure",
    "false_registration": "false_statement_or_personation_in_document_registration",
    "fraudulent_execution_of_deed_of_transfer": "false_statement_or_personation_in_document_registration",
    # food
    "food_misbranding": "misbranded_food",
    "sale_of_expired_food": "sub_standard_food",
    "food_adulteration": "unsafe_food",
    # IP
    "software_piracy": "copyright_infringement",
    "circumvention_of_protection": "copyright_infringement",
    # environment
    "industrial_pollution": "emission_or_discharge_beyond_standards",
    "air_pollution": "emission_or_discharge_beyond_standards",
    "environmental_violation": "environmental_clearance_violation",
    "illegal_mining": "environmental_clearance_violation",
    # labour (taxonomy has forced_labour, child_labour_violation,
    # victimisation_of_employee but no specific wage entries — fold legacy
    # wage names into victimisation_of_employee which is the closest match)
    "non_payment_of_wages": "victimisation_of_employee",
    "illegal_wage_deduction": "victimisation_of_employee",
    "wage_theft": "victimisation_of_employee",
    "unsafe_labour_conditions": "forced_labour",
    "bonded_labour": "forced_labour",
    # SC/ST  (taxonomy uses sc_st, not sc_or_st)
    "caste_based_atrocity": "atrocity_against_sc_st_person",
    "caste_based_discrimination": "caste_based_insult_or_intimidation_of_sc_st",
    "caste_based_property_attack": "atrocity_against_sc_st_person",
    "caste_based_insult": "caste_based_insult_or_intimidation_of_sc_st",
    "atrocity_against_sc_st": "atrocity_against_sc_st_person",
    "social_boycott_of_sc_st": "social_or_economic_boycott_of_sc_st",
    "atrocity_against_sc_or_st_person": "atrocity_against_sc_st_person",
    "caste_based_insult_or_intimidation_of_sc_or_st": "caste_based_insult_or_intimidation_of_sc_st",
    "abetment_of_sc_or_st_atrocity": "abetment_of_sc_st_atrocity",
    "abuse_or_exploitation_of_sc_or_st_woman": "abuse_or_exploitation_of_sc_st_woman",
    # terror
    "terror_funding": "raising_funds_for_terrorism",
    "terror_propaganda": "terrorist_act",
    "terror_recruitment": "terrorist_act",
    # property aliases (taxonomy has theft / snatching / criminal_trespass)
    "house_trespass": "house_trespass_or_house_breaking",
    "burglary": "house_trespass_or_house_breaking",
    "arson": "arson_fire_or_explosion_to_property",
    # family / DV
    "domestic_violence": "cruelty_by_husband_or_relatives",
    "cruelty_by_husband": "cruelty_by_husband_or_relatives",
    # generic kidnapping/restraint variants
    "wrongful_restraint": "wrongful_confinement",
    # obstruction-of-justice (taxonomy has direct names — but leave aliases
    # for any caller still using older spellings)
    "harbouring_offender": "offender_harbouring",
    "destruction_of_evidence": "destroying_evidence",
    "giving_false_evidence": "false_evidence",
    "fabrication_of_false_evidence": "fabricating_false_evidence",
}


# ---------------------------------------------------------------------------
# Merge in the central training-side alias map.
#
# `LAW_RAW_DATA/TAXONOMY_ALIASES.json` is the source of truth for drift
# labels emitted by the SFT teacher / fine-tuned student. Folding it into
# LEGACY_ALIASES means inference (BFF), training (merge_and_split), and
# eval all share one drift -> canonical map.
#
# Where a key exists in BOTH maps, the file wins because it is the
# curated, explicit choice — the in-code LEGACY_ALIASES sometimes
# fuzzy-matched the wrong canonical (e.g. `assault` -> `sexual_assault_on_child`).
# ---------------------------------------------------------------------------
def _load_external_aliases() -> Dict[str, str]:
    if not _TAXONOMY_ALIASES_PATH.exists():
        return {}
    try:
        with _TAXONOMY_ALIASES_PATH.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:  # noqa: BLE001
        return {}
    return dict(payload.get("aliases") or {})


for _drift, _canon in _load_external_aliases().items():
    if _canon in VALID_CATEGORIES:
        LEGACY_ALIASES[_drift] = _canon


def _self_check() -> None:
    """Make sure every legacy alias points to a real taxonomy category.
    Stale aliases would silently break extraction. Run at import time."""
    bad: list[str] = []
    canonical = {c for c in VALID_CATEGORIES if c not in LEGACY_ALIASES}
    for legacy, target in LEGACY_ALIASES.items():
        if target not in canonical:
            bad.append(f"  {legacy!r} -> {target!r}  (target not in taxonomy)")
    if bad:
        # Don't raise — just warn, so a missing alias target doesn't break
        # the whole pipeline.
        import warnings
        warnings.warn(
            "Legacy alias targets missing from taxonomy:\n" + "\n".join(bad)
        )


_self_check()

# Make legacy names accepted by the membership check (they are normalized to
# their canonical category by the validator before downstream consumers see
# them).
for _legacy in LEGACY_ALIASES:
    VALID_CATEGORIES.add(_legacy)


def normalize_category(raw: str) -> str:
    """Canonicalize a crime category string.

    1. Lower-case, replace separators with underscores.
    2. Apply legacy alias mapping.
    3. Return canonical category (or empty string if it cannot be mapped).
    """
    if not raw:
        return ""
    cleaned = _category_to_snake(raw)
    if cleaned in LEGACY_ALIASES:
        return LEGACY_ALIASES[cleaned]
    if cleaned in VALID_CATEGORIES:
        return cleaned
    # Fuzzy fallback — unchanged from the original behaviour.
    for candidate in VALID_CATEGORIES:
        if candidate in cleaned or cleaned in candidate:
            # Prefer canonical (taxonomy) categories over legacy aliases.
            if candidate not in LEGACY_ALIASES:
                return candidate
    for candidate in VALID_CATEGORIES:
        if candidate in cleaned or cleaned in candidate:
            return LEGACY_ALIASES.get(candidate, candidate)
    return ""
