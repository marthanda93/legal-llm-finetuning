#!/usr/bin/env python3
"""Merge all stage outputs, dedupe, canonicalize, balance, and split into
train / val / test JSONL.

Outputs follow the `mlx_lm.lora` chat-format spec:
    https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md

Each output line:
    {"messages": [{"role": "system", "content": "..."},
                  {"role": "user",   "content": "..."},
                  {"role": "assistant","content": "..."}]}

Files written under `training/datasets/final/`:
    train.jsonl   (~80%)
    valid.jsonl   (~10%)
    test.jsonl    (~10%)
    manifest.json (counts + provenance)

`mlx_lm.lora --data <dir>` expects exactly `train.jsonl` and `valid.jsonl` in
the directory; `test.jsonl` is for our own eval.

Pipeline applied to every record before split:
    1. Dedupe on (user, assistant[:200]).
    2. Canonicalize labels via `to_mlx_chat_record` (uses
       LAW_RAW_DATA/TAXONOMY_ALIASES.json -> only canonical 231 labels reach
       the final splits; off-vocabulary events are dropped).
    3. Per-category stratified split — every label with >= `min_for_eval`
       gold appearances gets at least one example in valid AND test.
    4. Optional class balancing:
         --downsample-cap N  : cap any single class to at most N train rows
                               (defaults to 500; protects tail labels from
                               being drowned by `criminal_intimidation`).
         --upsample-min N    : duplicate tail-class rows until each class has
                               at least N train rows (defaults to 5).

Run:
    python -m training.merge_and_split
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from training.common import (
    DATASETS_DIR,
    FINAL_DIR,
    RAW_STAGE_DIR,
    canonical_categories,
    fingerprint,
    load_taxonomy_aliases,
    read_jsonl,
    to_mlx_chat_record,
)


DEFAULT_STAGE_FILES = {
    "A": RAW_STAGE_DIR / "stage_a_definitions.jsonl",
    "B": RAW_STAGE_DIR / "stage_b_taxonomy.jsonl",
    "C": RAW_STAGE_DIR / "stage_c_synthetic_firs.jsonl",
    "D": RAW_STAGE_DIR / "stage_d_hard_negatives.jsonl",
    "E": RAW_STAGE_DIR / "stage_e_definition_grounded.jsonl",
}


# ---------------------------------------------------------------------------
# Per-record category extraction (post-canonicalization)
# ---------------------------------------------------------------------------
def _record_primary_category(rec: Dict[str, Any]) -> str:
    """Return the primary `crime_category` for stratification.

    For Stage C/E (FIR -> JSON) and Stage D (FIR -> empty events) we read
    from the assistant JSON. For Stage B (single-label classify) we read the
    bare snake_case answer. Stage A definitions have no category and live in
    a generic '__no_category__' bucket so they get random-split.
    """
    msgs = rec.get("messages") or []
    user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
    asst = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
    asst = (asst or "").strip()

    # Stage C / E (FIR -> JSON)
    if user.startswith("FIR:") or asst.startswith("{"):
        try:
            payload = json.loads(asst)
            evs = payload.get("events") or []
            if not evs:
                return "non_crime"
            cats = [e.get("crime_category") for e in evs if e.get("crime_category")]
            return cats[0] if cats else "non_crime"
        except Exception:  # noqa: BLE001
            return "__bad_json__"

    # Stage B (single label)
    if asst and " " not in asst and asst.replace("_", "").isalnum():
        return asst

    return "__no_category__"


# ---------------------------------------------------------------------------
# Stratified split: guarantee min N per class per eval split
# ---------------------------------------------------------------------------
def _stratified_split(
    records: List[Dict[str, Any]],
    *,
    train_frac: float,
    val_frac: float,
    seed: int,
    min_per_eval_split: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Per-category split.

    For every category with at least `min_per_eval_split * 2` examples, we
    reserve `min_per_eval_split` for valid and `min_per_eval_split` for test
    *first*, then split the remainder by `train_frac / val_frac / test_frac`.
    Categories with fewer examples fall through to a global random split so
    they at least land in train.
    """
    rng = random.Random(seed)
    test_frac = max(0.0, 1.0 - train_frac - val_frac)

    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_cat[_record_primary_category(rec)].append(rec)

    train: List[Dict[str, Any]] = []
    valid: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for cat, items in by_cat.items():
        rng.shuffle(items)
        n = len(items)

        if n >= min_per_eval_split * 2 + 1:
            v_keep = items[:min_per_eval_split]
            t_keep = items[min_per_eval_split : 2 * min_per_eval_split]
            rest = items[2 * min_per_eval_split :]
        else:
            v_keep, t_keep, rest = [], [], items

        n_rest = len(rest)
        n_train = int(round(n_rest * train_frac))
        n_valid = int(round(n_rest * val_frac))
        train.extend(rest[:n_train])
        valid.extend(rest[n_train : n_train + n_valid] + v_keep)
        test.extend(rest[n_train + n_valid :] + t_keep)

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return {"train": train, "valid": valid, "test": test}


# ---------------------------------------------------------------------------
# Class balancing on the train split
# ---------------------------------------------------------------------------
def _balance_train(
    train: List[Dict[str, Any]],
    *,
    downsample_cap: int,
    upsample_min: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Counter, Counter]:
    """Cap dominant categories and up-sample tails. Returns (new_train,
    pre_counts, post_counts)."""
    rng = random.Random(seed)
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in train:
        by_cat[_record_primary_category(rec)].append(rec)

    pre = Counter({c: len(v) for c, v in by_cat.items()})
    out: List[Dict[str, Any]] = []
    for cat, items in by_cat.items():
        # Don't touch generic buckets — Stage A definitions and bad-JSON.
        if cat in ("__no_category__", "__bad_json__"):
            out.extend(items)
            continue

        # Down-sample dominant classes (e.g. criminal_intimidation = 1535).
        if downsample_cap > 0 and len(items) > downsample_cap:
            rng.shuffle(items)
            items = items[:downsample_cap]

        # Up-sample tails by duplication.
        if upsample_min > 0 and 0 < len(items) < upsample_min:
            multiplier = (upsample_min + len(items) - 1) // len(items)
            items = (items * multiplier)[:upsample_min]

        out.extend(items)

    post = Counter({c: 0 for c in by_cat})
    for rec in out:
        post[_record_primary_category(rec)] += 1
    rng.shuffle(out)
    return out, pre, post


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Merge & split training data.")
    parser.add_argument("--out-dir", type=Path, default=FINAL_DIR)
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-stage-a", type=int, default=6000,
                        help="Cap Stage A records (definitions are abundant; capping "
                             "keeps the SFT mix balanced)")
    parser.add_argument(
        "--max-stage-b",
        type=int,
        default=4000,
        help="Cap Stage B (taxonomy) rows. 4000 = stronger label calibration "
             "vs Stage C; lower if you want extraction examples to dominate.",
    )
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Stage IDs to exclude (e.g. C if you haven't run it yet)")
    parser.add_argument(
        "--min-per-eval-split",
        type=int,
        default=1,
        help="Minimum examples per category in valid AND test (when supply allows). "
             "Set 0 to fall back to the legacy global random split.",
    )
    parser.add_argument(
        "--downsample-cap",
        type=int,
        default=500,
        help="Maximum train rows kept per category (0 = no cap). Protects tail "
             "categories from being drowned by `criminal_intimidation` (1535 rows).",
    )
    parser.add_argument(
        "--upsample-min",
        type=int,
        default=5,
        help="Tail-class up-sampling floor; classes with fewer rows are duplicated "
             "until they reach this count. Set 0 to disable.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    pool: List[Dict[str, Any]] = []
    seen_fp: set[str] = set()
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    for stage, path in DEFAULT_STAGE_FILES.items():
        if stage in args.exclude:
            print(f"[merge] excluding stage {stage}")
            continue
        recs = read_jsonl(path)
        if not recs:
            print(f"[merge] WARNING: no records for stage {stage} at {path}")
            continue

        # Cap Stage A and B so abundant templated data doesn't drown out
        # the precious teacher-distilled Stage C examples.
        if stage == "A" and len(recs) > args.max_stage_a:
            rng.shuffle(recs)
            recs = recs[: args.max_stage_a]
        if stage == "B" and len(recs) > args.max_stage_b:
            rng.shuffle(recs)
            recs = recs[: args.max_stage_b]

        for rec in recs:
            messages = rec.get("messages") or []
            if not messages:
                skipped[stage] += 1
                continue
            # Dedup on (user_text + first 200 chars of assistant)
            user_part = next((m["content"] for m in messages if m["role"] == "user"), "")
            asst_part = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            fp = fingerprint(user_part, asst_part[:200])
            if fp in seen_fp:
                skipped[stage] += 1
                continue
            seen_fp.add(fp)
            pool.append(rec)
            counts[stage] += 1

    if not pool:
        raise SystemExit("[merge] no records to merge — run the stage builders first")

    print(f"[merge] pool size: {len(pool)} records")
    for stage in sorted(DEFAULT_STAGE_FILES):
        print(f"        stage {stage}: kept {counts[stage]:>6}  skipped {skipped[stage]:>4}")

    # ---------------------------------------------------------------
    # Apply canonicalization NOW (so stratification keys are clean).
    # `to_mlx_chat_record` already canonicalizes labels via
    # canonicalize_assistant_content — convert once here so downstream
    # code reads the post-canonical category vocabulary.
    # ---------------------------------------------------------------
    canon_pool = [to_mlx_chat_record(rec) for rec in pool]

    # ---------------------------------------------------------------
    # Off-vocabulary audit (after canonicalization).
    # ---------------------------------------------------------------
    canon_set = canonical_categories()
    aliases = load_taxonomy_aliases()
    leftover_drift: Counter[str] = Counter()
    for rec in canon_pool:
        cat = _record_primary_category(rec)
        if cat in ("non_crime", "__no_category__", "__bad_json__"):
            continue
        if cat not in canon_set:
            leftover_drift[cat] += 1
    print(f"[merge] taxonomy aliases applied: {len(aliases)}; "
          f"canonical labels in vocabulary: {len(canon_set)}")
    if leftover_drift:
        print(f"[merge] WARNING: {sum(leftover_drift.values())} records still carry "
              f"{len(leftover_drift)} off-vocab labels post-canonicalization:")
        for c, n in leftover_drift.most_common():
            print(f"           {c:55} n={n}  (add to TAXONOMY_ALIASES.json)")

    # ---------------------------------------------------------------
    # Stratified split
    # ---------------------------------------------------------------
    if args.min_per_eval_split > 0:
        splits = _stratified_split(
            canon_pool,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
            min_per_eval_split=args.min_per_eval_split,
        )
    else:
        rng.shuffle(canon_pool)
        train_cut = int(len(canon_pool) * args.train_frac)
        val_cut = train_cut + int(len(canon_pool) * args.val_frac)
        splits = {
            "train": canon_pool[:train_cut],
            "valid": canon_pool[train_cut:val_cut],
            "test": canon_pool[val_cut:],
        }

    # ---------------------------------------------------------------
    # Class balancing on the train split only (never touch eval).
    # ---------------------------------------------------------------
    pre_counts = Counter()
    post_counts = Counter()
    if args.downsample_cap > 0 or args.upsample_min > 0:
        splits["train"], pre_counts, post_counts = _balance_train(
            splits["train"],
            downsample_cap=args.downsample_cap,
            upsample_min=args.upsample_min,
            seed=args.seed,
        )
        deltas = []
        for cat in sorted(set(pre_counts) | set(post_counts)):
            d = post_counts[cat] - pre_counts[cat]
            if d != 0:
                deltas.append((cat, pre_counts[cat], post_counts[cat], d))
        if deltas:
            print(f"[merge] train balancing: {len(deltas)} categories changed "
                  f"(downsample_cap={args.downsample_cap}, upsample_min={args.upsample_min})")
            downs = sorted([d for d in deltas if d[3] < 0], key=lambda x: x[3])[:10]
            ups = sorted([d for d in deltas if d[3] > 0], key=lambda x: -x[3])[:10]
            for cat, a, b, d in downs:
                print(f"           DOWN  {cat:50} {a:>4} -> {b:>4}  ({d:+d})")
            for cat, a, b, d in ups:
                print(f"           UP    {cat:50} {a:>4} -> {b:>4}  ({d:+d})")

    # ---------------------------------------------------------------
    # Coverage report (per split, vs canonical taxonomy).
    # ---------------------------------------------------------------
    def _category_set(recs: List[Dict[str, Any]]) -> Set[str]:
        return {
            _record_primary_category(r)
            for r in recs
            if _record_primary_category(r) in canon_set
        }

    coverage = {}
    for split_name, split_recs in splits.items():
        present = _category_set(split_recs)
        coverage[split_name] = {
            "categories_present": len(present),
            "canonical_total": len(canon_set),
            "missing_from_canonical": sorted(canon_set - present),
        }
        print(f"[merge] {split_name:5} category coverage: "
              f"{len(present)}/{len(canon_set)} canonical labels "
              f"(missing {len(canon_set) - len(present)})")

    # ---------------------------------------------------------------
    # Write splits
    # ---------------------------------------------------------------
    for split_name, split_recs in splits.items():
        path = args.out_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for rec in split_recs:
                # Records were already converted via to_mlx_chat_record; just dump.
                fh.write(json.dumps(rec, ensure_ascii=False))
                fh.write("\n")
        print(f"[merge] wrote {len(split_recs):>6} -> {path}")

    # Manifest with full provenance for reproducibility
    manifest = {
        "seed": args.seed,
        "fractions": {"train": args.train_frac, "val": args.val_frac,
                      "test": round(1 - args.train_frac - args.val_frac, 4)},
        "counts_by_stage": dict(counts),
        "skipped_by_stage": dict(skipped),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "max_stage_a": args.max_stage_a,
        "max_stage_b": args.max_stage_b,
        "min_per_eval_split": args.min_per_eval_split,
        "downsample_cap": args.downsample_cap,
        "upsample_min": args.upsample_min,
        "excluded_stages": args.exclude,
        "taxonomy": {
            "canonical_labels": len(canon_set),
            "aliases_applied": len(aliases),
            "leftover_drift": dict(leftover_drift),
        },
        "coverage": coverage,
    }
    with (args.out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[merge] manifest -> {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
