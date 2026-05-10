"""Validation and post-processing for extracted events."""

from __future__ import annotations

import re
from typing import Dict, List

from event_extraction.ontology import (
    CATEGORY_TO_TYPE,
    VALID_CATEGORIES,
    normalize_category,
)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_]{3,}", text.lower()))


def _split_evidence_candidates(fir_text: str) -> List[str]:
    primary = [segment.strip(" ,;:-") for segment in re.split(r"(?<=[.!?;])\s+|\n+", fir_text) if segment.strip()]
    candidates: List[str] = []
    for segment in primary:
        if len(segment) <= 320:
            candidates.append(segment)
            continue
        # Legal complaints are often a single long sentence; break by commas as fallback.
        comma_parts = [part.strip(" ,;:-") for part in segment.split(",") if part.strip()]
        candidates.extend(comma_parts if comma_parts else [segment])
    return candidates


def _extract_evidence_snippet(fir_text: str, event: Dict) -> str:
    if not fir_text.strip():
        return ""

    candidates = _split_evidence_candidates(fir_text)
    if not candidates:
        return re.sub(r"\s+", " ", fir_text).strip()

    query_tokens = _tokenize(str(event.get("action_summary", "")))
    category_tokens = _tokenize(str(event.get("crime_category", "")).replace("_", " "))
    query_tokens.update(category_tokens)
    if not query_tokens:
        return candidates[0]

    best_candidate = ""
    best_score = -1
    for candidate in candidates:
        candidate_tokens = _tokenize(candidate)
        overlap = len(query_tokens.intersection(candidate_tokens))
        category_overlap = len(category_tokens.intersection(candidate_tokens))
        score = (2 * overlap) + (3 * category_overlap)
        if score > best_score:
            best_score = score
            best_candidate = candidate

    return re.sub(r"\s+", " ", best_candidate).strip()


def validate_events(events: List[Dict], fir_text: str = "") -> List[Dict]:
    if not isinstance(events, list):
        return []

    cleaned: List[Dict] = []
    for event in events:
        if not isinstance(event, dict) or "action_summary" not in event:
            continue

        category = normalize_category(event.get("crime_category", ""))
        if not category:
            continue
        event["crime_category"] = category
        event.setdefault("actors", "accused")
        event.setdefault("confidence", "high")

        action = event["action_summary"]
        action = re.sub(r"\bRs\.?\s*[\d,]+", "", action)
        action = re.sub(r"₹\s*[\d,]+", "", action)
        action = re.sub(r"\b\d+\s*(?:crore|lakh|lac)\b", "", action)
        event["action_summary"] = re.sub(r"\s+", " ", action).strip()

        # Keep richer context fields for downstream legal workflows.
        details = event.get("details") or event.get("detail") or event["action_summary"]
        evidence_from_fir = _extract_evidence_snippet(fir_text, event)
        evidence = evidence_from_fir or event.get("evidence") or event["action_summary"]
        event["details"] = re.sub(r"\s+", " ", str(details)).strip()
        event["evidence"] = re.sub(r"\s+", " ", str(evidence)).strip()
        cleaned.append(event)

    # Normalize ids in narrative order (list order after validation).
    for index, event in enumerate(cleaned, start=1):
        event["event_id"] = index
        event.pop("sequence", None)

    return cleaned


def infer_categories(events: List[Dict]) -> List[str]:
    categories = {CATEGORY_TO_TYPE[event.get("crime_category", "")] for event in events if event.get("crime_category", "") in CATEGORY_TO_TYPE}
    return sorted(categories)

