"""Response parsing utilities."""

from __future__ import annotations

import json
import re
from typing import Dict


# Qwen3 Instruct insists on emitting `<think>...</think>` even when
# /no_think is set; observed in 100% of eval cases. Strip them up front
# so the JSON-object regex never has to step over them (and so model
# tokens spent on the empty think block don't cause the JSON to be
# truncated by the server's max-tokens budget).
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def parse_response(content: str) -> Dict:
    cleaned = content.strip()
    cleaned = _THINK_RE.sub("", cleaned).strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    object_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if object_match:
        try:
            return json.loads(object_match.group(0))
        except json.JSONDecodeError:
            pass

    array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if array_match:
        try:
            return {"events": json.loads(array_match.group(0))}
        except json.JSONDecodeError:
            pass

    return {"events": [], "fir_text_categories": []}

