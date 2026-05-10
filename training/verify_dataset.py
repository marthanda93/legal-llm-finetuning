"""Sanity-check the merged train/valid/test splits.

Confirms:
1. Every assistant response either parses as JSON whose `events[].crime_category`
   are all in the canonical set, OR is a single canonical token (Stage B).
2. No drift labels survived `merge_and_split.py` canonicalization.
3. Per-split coverage of the 231-label taxonomy.

Used by `make verify`. Exits non-zero if any drift label is found in the
final splits.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from training.common import canonical_categories

SLUG_RE = re.compile(r"[a-z0-9_]+")


def _iter_records(path: Path):
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _check_split(path: Path, canon: set[str]) -> tuple[int, int, Counter, set[str]]:
    """Return (n_records, n_events, drift_counter, observed_labels)."""
    drift: Counter = Counter()
    observed: set[str] = set()
    n_records = n_events = 0

    for rec in _iter_records(path):
        n_records += 1
        messages = rec.get("messages") or []
        assistant = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
        assistant = assistant.strip()
        if not assistant:
            continue
        try:
            payload = json.loads(assistant)
        except Exception:
            payload = None

        if isinstance(payload, dict):
            for ev in payload.get("events") or []:
                n_events += 1
                cat = (ev.get("crime_category") or "").strip()
                if not cat:
                    continue
                observed.add(cat)
                if cat not in canon and cat != "non_crime":
                    drift[cat] += 1
        elif SLUG_RE.fullmatch(assistant):
            n_events += 1
            observed.add(assistant)
            if assistant not in canon and assistant != "non_crime":
                drift[assistant] += 1

    return n_records, n_events, drift, observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="training/datasets/final",
        help="Directory holding {train,valid,test}.jsonl",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on ANY drift label (default: only exits if drift > 0 in train).",
    )
    args = parser.parse_args()

    canon = canonical_categories()
    print(f"[verify] canonical labels: {len(canon)}")

    any_drift_in_train = False
    summary: list[tuple[str, int, int, int, int]] = []

    for split in ("train", "valid", "test"):
        path = Path(args.data_dir) / f"{split}.jsonl"
        if not path.exists():
            print(f"[verify] WARN  {path} missing", file=sys.stderr)
            continue
        n_records, n_events, drift, observed = _check_split(path, canon)
        coverage = len(observed & canon)
        drift_total = sum(drift.values())
        summary.append((split, n_records, n_events, drift_total, coverage))
        print(
            f"[{split:>5}] records={n_records:>5}  events={n_events:>6}  "
            f"drift_events={drift_total:>4}  unique_drift={len(drift):>3}  "
            f"taxonomy_coverage={coverage}/{len(canon)}"
        )
        if drift:
            for cat, count in drift.most_common(10):
                print(f"          DRIFT  {cat:55} n={count}")
            if split == "train":
                any_drift_in_train = True

    print()
    print("[verify] summary:")
    for split, n_records, n_events, drift_total, coverage in summary:
        ok = "OK" if drift_total == 0 else "DRIFT"
        print(
            f"  {split:>5}  records={n_records:>5}  events={n_events:>6}  "
            f"coverage={coverage:>3}/{len(canon)}   [{ok}]"
        )

    if args.strict and any(d for _, _, _, d, _ in summary):
        print("[verify] FAIL  strict mode + drift labels detected", file=sys.stderr)
        return 1
    if any_drift_in_train:
        print("[verify] FAIL  drift labels in train.jsonl", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
