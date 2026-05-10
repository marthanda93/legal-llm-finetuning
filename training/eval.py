#!/usr/bin/env python3
"""Eval harness for FIR event extraction.

Computes the metrics that actually matter for legal extraction:

    1. JSON validity rate         (% outputs that parse + pass validate_events)
    2. Event-count MAE             (|predicted - gold| averaged)
    3. Per-category P/R/F1         (over the 231 snake_case canonical categories)
    4. Action-summary cosine       (sentence-transformers similarity, optional)
    5. Non-crime false-positive    (% of zero-event golds where model output >0)

Runs against any OpenAI-compatible chat endpoint. Default points at the local
mlx_lm.server on port 8030.

Usage:
    # eval the fine-tuned legal model (slim prompt, no few-shots)
    python -m training.eval \\
        --test-jsonl training/datasets/final/test.jsonl \\
        --model file://$(pwd)/training/models/legal-qwen3-14b-fused-v2 \\
        --report training/logs/eval-fused-v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx

from event_extraction.parsing import parse_response
from event_extraction.validation import validate_events
from training.common import (
    SLIM_EXTRACTION_SYSTEM_PROMPT,
    canonical_categories,
    canonicalize_label,
)


def _slim_user_prompt(fir_text: str) -> str:
    return f"FIR:\n{fir_text}"


def _call_llm(url: str, model: str, system: str, user: str,
              max_tokens: int, timeout: float) -> str:
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    resp = httpx.post(url, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _extract_gold_from_test_record(rec: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Recover (fir_text, gold_payload) from a Stage-C/D test JSONL line."""
    msgs = rec.get("messages") or []
    user = next((m for m in msgs if m["role"] == "user"), None)
    asst = next((m for m in msgs if m["role"] == "assistant"), None)
    if not user or not asst:
        return None
    user_text = user["content"]
    # Stage C records start with "FIR:\n<text>"
    if not user_text.startswith("FIR:"):
        return None
    fir_text = user_text[len("FIR:"):].lstrip("\n").strip()
    try:
        gold = json.loads(asst["content"])
    except Exception:  # noqa: BLE001
        return None
    return fir_text, gold


def _safe_get_categories(payload: Dict[str, Any]) -> List[str]:
    out = []
    for ev in payload.get("events", []) or []:
        cat = ev.get("crime_category", "")
        if cat:
            out.append(cat)
    return out


def _canon_set(cats: List[str], drop_off_vocab: bool) -> Tuple[set, List[str]]:
    """Apply taxonomy alias map to a list of labels.

    Returns (canonical_set, off_vocab_originals).
    Off-vocab labels (no canonical mapping) are dropped from the set when
    `drop_off_vocab=True` so they cannot artificially inflate the FP count.
    """
    canon: set = set()
    off: List[str] = []
    canonical = canonical_categories()
    for c in cats:
        mapped = canonicalize_label(c)
        if mapped is None or (drop_off_vocab and mapped not in canonical and mapped != "non_crime"):
            off.append(c)
            if mapped is not None and not drop_off_vocab:
                canon.add(mapped)
            continue
        canon.add(mapped)
    return canon, off


def _action_cosine(gold_actions: List[str], pred_actions: List[str],
                   embedder) -> float:
    if not gold_actions or not pred_actions or embedder is None:
        return 0.0
    import numpy as np
    g = embedder.encode(gold_actions)
    p = embedder.encode(pred_actions)
    g_norm = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
    p_norm = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)
    sim = g_norm @ p_norm.T  # (g, p)
    # Greedy max match per gold action
    best = sim.max(axis=1).mean() if sim.size > 0 else 0.0
    return float(best)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FIR event extraction quality.")
    parser.add_argument("--test-jsonl", type=Path, required=True,
                        help="Test split JSONL produced by training/merge_and_split.py")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8030/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-cases", type=int, default=200,
                        help="Cap how many test records to evaluate (saves time)")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--report", type=Path, default=Path("eval_report.json"))
    parser.add_argument("--no-cosine", action="store_true",
                        help="Skip action-summary cosine (saves loading sentence-transformers)")
    parser.add_argument("--trace-jsonl", type=Path, default=None,
                        help="Per-case trace (system+user prompt, raw response, gold, "
                             "pred_events, latency). One JSON object per line, flushed "
                             "after each case so you can `tail -f` while it runs. "
                             "Defaults to <report>.trace.jsonl")
    parser.add_argument("--no-trace", action="store_true",
                        help="Disable per-case trace dump.")
    parser.add_argument("--print-each", action="store_true",
                        help="Print a one-line preview of every case to stdout.")
    parser.add_argument(
        "--keep-off-vocab",
        action="store_true",
        help="Keep predictions whose category does not map to any canonical "
             "label (count them as FPs). Default: drop off-vocab predictions "
             "from the F1 calculation but record them in the report.",
    )
    args = parser.parse_args()

    # Load test data
    test_records: List[Dict[str, Any]] = []
    with args.test_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            test_records.append(json.loads(line))
    if args.max_cases > 0:
        test_records = test_records[: args.max_cases]
    print(f"[eval] loaded {len(test_records)} test records")

    # Load embedder for cosine if requested
    embedder = None
    if not args.no_cosine:
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] could not load sentence-transformers: {exc}; cosine=0")

    # Per-case trace stream (incremental, flushed after each case)
    trace_path: Optional[Path] = None
    trace_fh = None
    if not args.no_trace:
        trace_path = args.trace_jsonl or args.report.with_suffix(".trace.jsonl")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_fh = trace_path.open("w", encoding="utf-8")
        print(f"[eval] per-case trace -> {trace_path}")

    # Metrics accumulators
    n_total = 0
    n_json_ok = 0
    n_skipped_non_fir = 0
    event_count_abs_err = 0
    action_cosines: List[float] = []
    non_crime_fp = 0
    non_crime_total = 0
    latencies: List[float] = []

    # Per-category multi-label F1 needs TP/FP/FN per category
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()

    # Off-vocab predictions audit (the model emitted a non-canonical label
    # that did not survive `canonicalize_label`; impossible to score so they
    # are recorded but not counted as FP unless --keep-off-vocab is set).
    off_vocab_pred: Counter[str] = Counter()

    for rec_idx, rec in enumerate(test_records):
        unpacked = _extract_gold_from_test_record(rec)
        if unpacked is None:
            n_skipped_non_fir += 1
            continue
        fir_text, gold = unpacked
        gold_events = gold.get("events", []) or []
        gold_categories, gold_off = _canon_set(
            _safe_get_categories(gold), drop_off_vocab=not args.keep_off_vocab
        )
        gold_actions = [ev.get("action_summary", "") for ev in gold_events if ev.get("action_summary")]
        is_non_crime = len(gold_events) == 0

        system, user = SLIM_EXTRACTION_SYSTEM_PROMPT, _slim_user_prompt(fir_text)

        # Call LLM
        t0 = time.time()
        call_error: Optional[str] = None
        raw = ""
        try:
            raw = _call_llm(args.llm_url, args.model, system, user,
                            args.max_tokens, args.timeout)
        except Exception as exc:  # noqa: BLE001
            call_error = str(exc)
            print(f"[eval] case {rec_idx}: LLM call failed: {exc}")
        latency_s = time.time() - t0
        if call_error is not None:
            if trace_fh is not None:
                trace_fh.write(json.dumps({
                    "case_index": rec_idx,
                    "fir_text_chars": len(fir_text),
                    "system_prompt": system,
                    "user_prompt": user,
                    "raw_response": "",
                    "raw_response_chars": 0,
                    "latency_s": round(latency_s, 3),
                    "gold": gold,
                    "pred_events": [],
                    "json_ok": False,
                    "error": call_error,
                }, ensure_ascii=False) + "\n")
                trace_fh.flush()
            continue
        latencies.append(latency_s)
        n_total += 1

        # Parse + validate
        try:
            parsed = parse_response(raw)
        except Exception:
            parsed = {"events": []}
        pred_events = validate_events(parsed.get("events", []), fir_text=fir_text)
        if isinstance(pred_events, list):
            n_json_ok += 1

        pred_raw_cats = _safe_get_categories({"events": pred_events})
        pred_categories, pred_off = _canon_set(
            pred_raw_cats, drop_off_vocab=not args.keep_off_vocab
        )
        for c in pred_off:
            off_vocab_pred[c] += 1
        pred_actions = [ev.get("action_summary", "") for ev in pred_events]

        if trace_fh is not None:
            trace_fh.write(json.dumps({
                "case_index": rec_idx,
                "fir_text_chars": len(fir_text),
                "system_prompt": system,
                "user_prompt": user,
                "raw_response": raw,
                "raw_response_chars": len(raw),
                "latency_s": round(latency_s, 3),
                "gold": gold,
                "pred_events": pred_events if isinstance(pred_events, list) else [],
                "json_ok": isinstance(pred_events, list),
            }, ensure_ascii=False) + "\n")
            trace_fh.flush()

        if args.print_each:
            print(
                f"[eval] case {rec_idx}: "
                f"gold_n={len(gold_events)} pred_n={len(pred_events)} "
                f"json_ok={isinstance(pred_events, list)} "
                f"chars={len(raw)} latency={latency_s:.2f}s"
            )

        # Event count MAE
        event_count_abs_err += abs(len(pred_events) - len(gold_events))

        # Per-category multi-label F1
        for cat in gold_categories | pred_categories:
            in_gold = cat in gold_categories
            in_pred = cat in pred_categories
            if in_gold and in_pred:
                tp[cat] += 1
            elif in_pred and not in_gold:
                fp[cat] += 1
            elif in_gold and not in_pred:
                fn[cat] += 1

        if is_non_crime:
            non_crime_total += 1
            if len(pred_events) > 0:
                non_crime_fp += 1

        if gold_actions and pred_actions and embedder is not None:
            action_cosines.append(_action_cosine(gold_actions, pred_actions, embedder))

        if (rec_idx + 1) % 25 == 0:
            print(f"[eval] {rec_idx + 1}/{len(test_records)} | "
                  f"json_ok={n_json_ok}/{n_total} | mean_latency={sum(latencies)/len(latencies):.2f}s")

    # Aggregate per-category F1.
    # Categories with support=0 (the model emitted but gold never had them)
    # cannot be meaningfully scored — keep them in a separate `pred_only`
    # bucket so the main `per_category` map only carries gold-supported
    # labels (cleaner reports + macro F1 not diluted by 0/0 entries).
    per_cat: Dict[str, Dict[str, float]] = {}
    pred_only_categories: Dict[str, int] = {}
    for cat in sorted(set(list(tp.keys()) + list(fp.keys()) + list(fn.keys()))):
        support = tp[cat] + fn[cat]
        if support == 0:
            # Pred-only false positives — record but don't dilute macro F1.
            pred_only_categories[cat] = fp[cat]
            continue
        prec = tp[cat] / (tp[cat] + fp[cat]) if (tp[cat] + fp[cat]) else 0.0
        rec_ = tp[cat] / support
        f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0.0
        per_cat[cat] = {"precision": round(prec, 3), "recall": round(rec_, 3),
                         "f1": round(f1, 3), "support": support}

    macro_f1 = (sum(c["f1"] for c in per_cat.values()) / len(per_cat)) if per_cat else 0.0
    micro_tp = sum(tp.values())
    micro_fp = sum(fp.values())
    micro_fn = sum(fn.values())
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    cosine_mean: Optional[float]
    if embedder is None:
        # Cosine was disabled (--no-cosine) or sentence-transformers failed.
        # Reporting 0.0 here was misleading — readers assumed the model
        # scored zero similarity. Use null instead so the value is clearly
        # "not measured".
        cosine_mean = None
    else:
        cosine_mean = round(sum(action_cosines) / max(1, len(action_cosines)), 3)

    report = {
        "model": args.model,
        "prompt_mode": "slim",
        "n_total": n_total,
        "n_skipped_non_fir": n_skipped_non_fir,
        "json_validity_rate": round(n_json_ok / max(1, n_total), 3),
        "event_count_mae": round(event_count_abs_err / max(1, n_total), 3),
        "action_summary_cosine_mean": cosine_mean,
        "action_summary_cosine_enabled": embedder is not None,
        "non_crime_false_positive_rate": round(non_crime_fp / max(1, non_crime_total), 3),
        "non_crime_cases": non_crime_total,
        "category_macro_f1": round(macro_f1, 3),
        "category_micro_f1": round(micro_f1, 3),
        "category_macro_f1_basis": len(per_cat),
        "mean_latency_s": round(sum(latencies) / max(1, len(latencies)), 3),
        "off_vocab_pred_total": int(sum(off_vocab_pred.values())),
        "off_vocab_pred_top": dict(off_vocab_pred.most_common(20)),
        "off_vocab_dropped_from_score": (not args.keep_off_vocab),
        "pred_only_false_positive_categories": dict(
            sorted(pred_only_categories.items(), key=lambda kv: -kv[1])[:20]
        ),
        "per_category": per_cat,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    if trace_fh is not None:
        trace_fh.close()

    cosine_repr = (
        "skipped (--no-cosine)" if cosine_mean is None else f"{cosine_mean}"
    )
    print("\n" + "=" * 60)
    print(f"  Model:               {args.model}")
    print(f"  Prompt mode:         {report['prompt_mode']}")
    print(f"  Cases evaluated:     {report['n_total']}  "
          f"(skipped non-FIR: {report['n_skipped_non_fir']})")
    print(f"  JSON validity:       {report['json_validity_rate'] * 100:.1f}%")
    print(f"  Event-count MAE:     {report['event_count_mae']}")
    print(f"  Macro F1 (category): {report['category_macro_f1']}  "
          f"over {report['category_macro_f1_basis']} gold-supported labels")
    print(f"  Micro F1 (category): {report['category_micro_f1']}")
    print(f"  Action cosine:       {cosine_repr}")
    print(f"  Non-crime FP rate:   {report['non_crime_false_positive_rate'] * 100:.1f}%  ({report['non_crime_cases']} cases)")
    print(f"  Mean latency:        {report['mean_latency_s']}s")
    print(f"  Off-vocab preds:     {report['off_vocab_pred_total']}  "
          f"(scored: {'no (dropped)' if report['off_vocab_dropped_from_score'] else 'yes (counted as FP)'})")
    if report["off_vocab_pred_top"]:
        for c, n in list(report["off_vocab_pred_top"].items())[:5]:
            print(f"      drift-emit  {c:50} n={n}")
    if report["pred_only_false_positive_categories"]:
        n_pred_only = sum(report["pred_only_false_positive_categories"].values())
        print(f"  Pred-only labels (FP, no gold support): {n_pred_only} events across "
              f"{len(report['pred_only_false_positive_categories'])} categories")
    print(f"  Full report:         {args.report}")
    if trace_path is not None:
        print(f"  Per-case trace:      {trace_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
