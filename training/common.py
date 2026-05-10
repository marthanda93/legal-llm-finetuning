"""Shared utilities for training-data builders.

Every Stage A/B/C/D script emits records in this canonical schema:

    {
        "stage": "A" | "B" | "C" | "D",
        "task":  short string e.g. "definition_qa", "taxonomy_classify",
                 "fir_event_extraction", "hard_negative",
        "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "meta": { ...arbitrary... }
    }

`merge_and_split.py` consumes these and emits ChatML-style JSONL ready for
`mlx_lm.lora`.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
LAW_RAW_DATA = REPO_ROOT / "LAW_RAW_DATA"
DEFINITIONS_DIR = LAW_RAW_DATA / "DEFINITIONS"
CRIME_TAXONOMY_PATH = LAW_RAW_DATA / "CRIME_TAXONOMY.json"
TAXONOMY_ALIASES_PATH = LAW_RAW_DATA / "TAXONOMY_ALIASES.json"
TRAINING_DIR = REPO_ROOT / "training"
DATASETS_DIR = TRAINING_DIR / "datasets"
RAW_STAGE_DIR = DATASETS_DIR / "stages_raw"
FINAL_DIR = DATASETS_DIR / "final"

# Base model: Qwen3-14B (4-bit MLX). Trained on Mac Studio M1 64 GB.
# HF source / MLX 4-bit are pinned in the Makefile (BASE_HF / BASE_MLX_DIR);
# Qwen3's hybrid <think>…</think> reasoning mode is suppressed via the
# trailing `/no_think` directive in SLIM_EXTRACTION_SYSTEM_PROMPT, which is
# trained AND served, so the fine-tuned model emits JSON directly with no
# chain-of-thought leakage.

# ---------------------------------------------------------------------------
# Slim system prompt used by every Stage C and D record.
#
# This is THE prompt the model will see at inference time after fine-tuning.
# Keeping it identical between training and inference is crucial for SFT
# performance.  See DOCS/LLM_FINETUNE_PLAN.md section 5.
# ---------------------------------------------------------------------------
SLIM_EXTRACTION_SYSTEM_PROMPT = (
    "You are an FIR event extractor for Indian criminal law. "
    "Read the FIR text and output ONE JSON object only, no prose, no markdown.\n"
    'Schema: {"fir_text_categories":[...], "events":[{"event_id":N,'
    '"crime_category":"...","action_summary":"...","details":"...",'
    '"evidence":"...","actors":"...","confidence":"..."}]}\n'
    "Rules:\n"
    "- crime_category MUST be exactly one trained taxonomy label in snake_case "
    "(no synonyms, no merged labels, no invented categories).\n"
    "- fir_text_categories: short formal offence-family headings aligned with "
    "that taxonomy (catalogue-style wording, not colloquial).\n"
    "- details & evidence: use precise offence language implied by the FIR; "
    "quote Acts/sections only when they appear in the FIR or are unavoidable "
    "for the named offence type.\n"
    "- action_summary: 4-8 words, generic, no names/dates/amounts/places/brands.\n"
    "- Extract every distinct criminal act. Skip non-crimes "
    "(FIR registration, investigation notes, civil disputes).\n"
    "- If no crime, return {\"fir_text_categories\":[],\"events\":[]}.\n"
    "/no_think"
)


# ---------------------------------------------------------------------------
# Stage A / B helper system prompts (knowledge injection)
# ---------------------------------------------------------------------------
DEFINITION_SYSTEM_PROMPT = (
    "You are an expert on Indian statutory law. Answer questions about legal "
    "definitions accurately and concisely. Prefer statutory wording (elements, "
    "actus reus / mens rea where relevant). Cite Act and section when the "
    "question names them; otherwise state the legal test clearly.\n"
    "/no_think"
)

TAXONOMY_SYSTEM_PROMPT = (
    "You map FIR fact patterns to the canonical Indian crime taxonomy "
    "(fixed snake_case labels tied to IPC and special Acts). "
    "Choose the single label whose legal definition best fits the facts—"
    "statutory ingredients beat colloquial similarity. "
    "Reply with exactly one snake_case label. If the pattern is not a crime, "
    "reply with non_crime.\n"
    "/no_think"
)


# ---------------------------------------------------------------------------
# Legacy system-prompt upgrade
#
# Older Stage C / D / E records were generated before we adopted Qwen3 and
# therefore lack the trailing `/no_think` directive. The merge step rewrites
# any legacy system prompt into the current canonical version so the trained
# model sees the same prompt at training and inference time.
# ---------------------------------------------------------------------------
_LEGACY_SLIM_FRAGMENT = (
    "You are an FIR event extractor for Indian criminal law."
)
_LEGACY_DEFINITION_FRAGMENT = (
    "You are an expert on Indian statutory law."
)
_LEGACY_TAXONOMY_FRAGMENT = (
    "You classify FIR fact patterns into the canonical Indian crime taxonomy."
)


def upgrade_system_prompt(content: str) -> str:
    """Rewrite a legacy system prompt to the current canonical version.

    Idempotent: if the prompt already contains `/no_think` it is returned
    unchanged."""
    if not content or "/no_think" in content:
        return content
    if _LEGACY_SLIM_FRAGMENT in content:
        return SLIM_EXTRACTION_SYSTEM_PROMPT
    if _LEGACY_DEFINITION_FRAGMENT in content:
        return DEFINITION_SYSTEM_PROMPT
    if _LEGACY_TAXONOMY_FRAGMENT in content:
        return TAXONOMY_SYSTEM_PROMPT
    return content


@dataclass
class TrainingRecord:
    """One supervised training example."""

    stage: str
    task: str
    messages: List[Dict[str, str]]
    meta: Dict[str, Any] = field(default_factory=dict)
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.record_id,
            "stage": self.stage,
            "task": self.task,
            "messages": self.messages,
            "meta": self.meta,
        }


def write_jsonl(path: Path, records: Iterable[Dict[str, Any] | TrainingRecord]) -> int:
    """Write records to a JSONL file. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            payload = rec.to_dict() if isinstance(rec, TrainingRecord) else rec
            fh.write(json.dumps(payload, ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")


def squish(text: str) -> str:
    """Collapse whitespace; strip."""
    return _WS_RE.sub(" ", text or "").strip()


def short(text: str, max_chars: int = 600) -> str:
    """Truncate long text for prompt budgets."""
    text = squish(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def fingerprint(*parts: str) -> str:
    """Stable short fingerprint for dedup."""
    import hashlib

    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").strip().lower().encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
# Canonical, high-quality Indian law names used in training data.  Several
# entries in LAW_RAW_DATA/DEFINITIONS have garbage `law_name` values like
# "Mv", "Dv", "Ndps", "Posh" derived from the filename; we override them here
# so every QA pair refers to the act by its real name.
CANONICAL_LAW_NAMES: Dict[str, str] = {
    "ARMS": "Arms Act, 1959",
    "BNS": "Bharatiya Nyaya Sanhita, 2023",
    "COPYRIGHT": "Copyright Act, 1957",
    "DOWRY_PROHIBITION": "Dowry Prohibition Act, 1961",
    "DPDP_ACT": "Digital Personal Data Protection Act, 2023",
    "DPDP": "Digital Personal Data Protection Act, 2023",
    "DV": "Protection of Women from Domestic Violence Act, 2005",
    "ENVIRONMENT_PROTECTION": "Environment (Protection) Act, 1986",
    "FOOD_SAFETY_AND_STANDARDS": "Food Safety and Standards Act, 2006",
    "IT_ACT": "Information Technology Act, 2000",
    "JUVENILE_JUSTICE": "Juvenile Justice (Care and Protection of Children) Act, 2015",
    "LABOUR": "Code on Wages, 2019 / Labour Codes",
    "MV": "Motor Vehicles Act, 1988",
    "NDPS": "Narcotic Drugs and Psychotropic Substances Act, 1985",
    "PAYMENT_OF_WAGES": "Payment of Wages Act, 1936",
    "POCSO": "Protection of Children from Sexual Offences Act, 2012",
    "POSH": "Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013",
    "PREVENTION_OF_CORRUPTION": "Prevention of Corruption Act, 1988",
    "PREVENTION_OF_MONEY_LAUNDERING": "Prevention of Money Laundering Act, 2002",
    "REGISTRATION": "Registration Act, 1908",
    "SC_ST": "Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989",
    "UAPA": "Unlawful Activities (Prevention) Act, 1967",
    "PMLA": "Prevention of Money Laundering Act, 2002",
}


def canonical_law_name(law_id: str, fallback: str = "") -> str:
    """Return the canonical Indian law name for a law_id."""
    if not law_id:
        return fallback or "Indian Law"
    key = law_id.strip().upper()
    if key in CANONICAL_LAW_NAMES:
        return CANONICAL_LAW_NAMES[key]
    # Fallback: only trust source `law_name` if it looks like a real act name
    # (longer than 8 chars and contains a year, comma, or "Act").
    f = (fallback or "").strip()
    if len(f) >= 8 and any(tok in f for tok in (",", "Act", "Sanhita", "Code")):
        return f
    return key.replace("_", " ").title()


def load_all_definitions() -> List[Dict[str, Any]]:
    """Flatten every term from every act in LAW_RAW_DATA/DEFINITIONS/*.json."""
    out: List[Dict[str, Any]] = []
    if not DEFINITIONS_DIR.exists():
        return out
    for act_path in sorted(DEFINITIONS_DIR.glob("*.json")):
        try:
            with act_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            print(f"[common] failed to load {act_path.name}: {exc}")
            continue
        law_id = payload.get("law_id") or act_path.stem
        law_name = canonical_law_name(law_id, fallback=payload.get("law_name", ""))
        for entry in payload.get("definitions", []) or []:
            if not isinstance(entry, dict):
                continue
            entry = {**entry, "_law_id": law_id, "_law_name": law_name}
            out.append(entry)
    return out


def load_crime_taxonomy() -> List[Dict[str, Any]]:
    """Flatten CRIME_TAXONOMY.json -> list of category entries with `group`."""
    if not CRIME_TAXONOMY_PATH.exists():
        return []
    with CRIME_TAXONOMY_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out: List[Dict[str, Any]] = []
    for group, items in raw.items():
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            cat = (it.get("category") or "").strip()
            if not cat:
                continue
            out.append({**it, "_group": group})
    return out


def category_to_snake(category: str) -> str:
    """'attempt to murder' -> 'attempt_to_murder'."""
    cat = (category or "").strip().lower()
    cat = re.sub(r"[^a-z0-9]+", "_", cat)
    cat = re.sub(r"_+", "_", cat).strip("_")
    return cat


# ---------------------------------------------------------------------------
# Taxonomy canonicalization (alias map)
#
# The Stage C teacher LLM occasionally invents close-but-not-canonical labels
# (e.g. `assault` instead of `assault_or_criminal_force`, `house_trespass`
# instead of `house_trespass_or_house_breaking`). Those drift labels poison
# both training and evaluation:
#
#   - Training: the model learns to emit a label that does not exist in the
#     official 231-label taxonomy. Downstream consumers (BFF section graph,
#     law-mapping, F1 scoring) reject it.
#   - Eval: gold and predicted labels disagree even when they refer to the
#     same offence, so per-category F1 collapses.
#
# `LAW_RAW_DATA/TAXONOMY_ALIASES.json` contains a curated `drift -> canonical`
# map. Helpers below load it once and apply it to:
#
#   * the canonical category vocabulary (`canonical_categories()`).
#   * a single `crime_category` value (`canonicalize_label`).
#   * the events array of an extractor JSON payload (`canonicalize_events`).
#   * the assistant content of a Stage C / D / E ChatML record
#     (`canonicalize_assistant_content`).
#
# Apply the SAME helpers at training-data merge time AND at eval scoring
# time so train/eval share one vocabulary.
# ---------------------------------------------------------------------------
_ALIAS_CACHE: Optional[Dict[str, str]] = None
_CANONICAL_CACHE: Optional[set[str]] = None


def load_taxonomy_aliases() -> Dict[str, str]:
    """Return {drift_label: canonical_label} from TAXONOMY_ALIASES.json (cached)."""
    global _ALIAS_CACHE
    if _ALIAS_CACHE is None:
        if not TAXONOMY_ALIASES_PATH.exists():
            _ALIAS_CACHE = {}
        else:
            with TAXONOMY_ALIASES_PATH.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            _ALIAS_CACHE = dict(payload.get("aliases", {}))
    return _ALIAS_CACHE


def canonical_categories() -> set[str]:
    """Set of all canonical snake_case category labels in the taxonomy (cached)."""
    global _CANONICAL_CACHE
    if _CANONICAL_CACHE is None:
        cats = set()
        for entry in load_crime_taxonomy():
            s = category_to_snake(entry.get("category", ""))
            if s:
                cats.add(s)
        _CANONICAL_CACHE = cats
    return _CANONICAL_CACHE


def canonicalize_label(label: str) -> Optional[str]:
    """Return the canonical snake_case label, or None if unknown.

    Resolution order:
      1. Exact canonical match  -> return as-is.
      2. Exact alias match      -> return mapped canonical.
      3. snake_case normalization -> retry steps 1-2.
      4. Unknown -> return None (caller decides whether to drop).
    """
    if not label:
        return None
    canon = canonical_categories()
    aliases = load_taxonomy_aliases()
    s = label.strip()
    if s in canon:
        return s
    if s in aliases:
        mapped = aliases[s]
        return mapped if mapped in canon else None
    snake = category_to_snake(s)
    if snake in canon:
        return snake
    if snake in aliases:
        mapped = aliases[snake]
        return mapped if mapped in canon else None
    if snake == "non_crime":
        return "non_crime"
    return None


def canonicalize_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rewrite `crime_category` on every event to its canonical label.

    Events whose category cannot be canonicalized are DROPPED — emitting an
    off-vocabulary label is worse than emitting one fewer event."""
    out: List[Dict[str, Any]] = []
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        raw = (ev.get("crime_category") or "").strip()
        canon = canonicalize_label(raw)
        if canon is None:
            continue
        new_ev = dict(ev)
        new_ev["crime_category"] = canon
        out.append(new_ev)
    return out


def canonicalize_assistant_content(content: str) -> str:
    """If `content` is a JSON object with `events`, canonicalize the labels
    and reserialize. Otherwise (Stage A definitions, Stage B label-only
    answers like 'theft'), apply alias remap to the bare snake_case payload.

    Always returns a string suitable as the new assistant message content."""
    if not content:
        return content
    stripped = content.strip()
    # Stage B answers are bare snake_case labels (sometimes followed by a
    # short note). If the entire content is a single snake_case token, try
    # to canonicalize it.
    if "{" not in stripped and re.fullmatch(r"[a-z0-9_]+", stripped):
        canon = canonicalize_label(stripped)
        return canon if canon else stripped
    # Otherwise, attempt JSON parse; only rewrite if it succeeds.
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return content
    if not isinstance(payload, dict) or "events" not in payload:
        return content
    payload = dict(payload)
    payload["events"] = canonicalize_events(payload.get("events") or [])
    # Re-number event_id sequentially so dropped events don't leave gaps.
    for i, ev in enumerate(payload["events"], start=1):
        if "event_id" in ev:
            ev["event_id"] = i
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Conversion to ChatML JSONL (mlx_lm format)
# ---------------------------------------------------------------------------
def to_mlx_chat_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert our internal record to the mlx_lm.lora 'chat' JSONL format.

    mlx_lm.lora supports a 'chat' format where each line is:
        {"messages": [{"role": "...", "content": "..."}, ...]}

    Reference: https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md

    The first system message is run through `upgrade_system_prompt` so that
    legacy Stage C / D / E records (generated before the Qwen3 switch) get
    the trailing `/no_think` directive added at merge time, without needing
    to regenerate the raw stage files."""
    messages: Optional[List[Dict[str, str]]] = record.get("messages")
    if not messages:
        raise ValueError(f"Record {record.get('id')} has no messages")
    upgraded: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            upgraded.append({
                "role": "system",
                "content": upgrade_system_prompt(content),
            })
        elif role == "assistant":
            upgraded.append({
                "role": "assistant",
                "content": canonicalize_assistant_content(content),
            })
        else:
            upgraded.append(m)
    return {"messages": upgraded}
