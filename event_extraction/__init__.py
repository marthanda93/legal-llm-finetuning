"""Lightweight event-extraction taxonomy + parsing utilities.

These three submodules are the only inference-side contract the training
and eval pipelines depend on:

    from event_extraction.ontology import VALID_CATEGORIES, CATEGORY_TO_TYPE
    from event_extraction.parsing import parse_response
    from event_extraction.validation import validate_events

The full inference stack (retriever, few-shot DB, FastAPI extractor) lives
in the upstream BFF service and is intentionally NOT shipped here — this
is a training workbench.
"""

from event_extraction.ontology import (
    CATEGORY_TO_TYPE,
    GROUP_TO_TYPE,
    LEGACY_ALIASES,
    VALID_CATEGORIES,
    normalize_category,
)
from event_extraction.parsing import parse_response
from event_extraction.validation import infer_categories, validate_events

__all__ = [
    "CATEGORY_TO_TYPE",
    "GROUP_TO_TYPE",
    "LEGACY_ALIASES",
    "VALID_CATEGORIES",
    "infer_categories",
    "normalize_category",
    "parse_response",
    "validate_events",
]
