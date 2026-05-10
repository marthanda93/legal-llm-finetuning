#!/usr/bin/env python3
"""Compare LoRA checkpoint weights on the same FIR prompts (local MLX).

Loads the base 4-bit model once per checkpoint (sequential runs + Metal cache
clear) to avoid keeping two copies in VRAM.

Example:
  python -m training.compare_adapter_checkpoints \\
    --checkpoints 0001400_adapters.safetensors 0001800_adapters.safetensors \\
    --max-prompts 4 --max-tokens 768
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import SLIM_EXTRACTION_SYSTEM_PROMPT


def _fir_prompts_from_jsonl(path: Path, max_prompts: int) -> List[Tuple[str, Dict[str, Any]]]:
    """Return [(fir_user_message, optional gold assistant json dict), ...]."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = rec.get("messages") or []
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            if not user.startswith("FIR:"):
                continue
            gold: Dict[str, Any] = {}
            asst = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            if asst:
                try:
                    gold = json.loads(asst)
                except json.JSONDecodeError:
                    gold = {}
            out.append((user, gold))
            if len(out) >= max_prompts:
                break
    return out


def _clear_metal() -> None:
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
    gc.collect()


def _unload(model: Any, tokenizer: Any) -> None:
    del model
    del tokenizer
    _clear_metal()


def _prepare_adapter_dir(adapter_root: Path, checkpoint_file: Path) -> Path:
    td = Path(tempfile.mkdtemp(prefix="mlx_adapter_"))
    shutil.copy(checkpoint_file, td / "adapters.safetensors")
    shutil.copy(adapter_root / "adapter_config.json", td / "adapter_config.json")
    return td


def run_checkpoint(
    *,
    base_model: Path,
    adapter_root: Path,
    checkpoint_name: str,
    fir_cases: List[Tuple[str, Dict[str, Any]]],
    max_tokens: int,
    max_kv_size: int,
    lazy_load: bool,
) -> Dict[str, Any]:
    from mlx_lm import generate, load

    ck_path = adapter_root / checkpoint_name
    if not ck_path.is_file():
        raise FileNotFoundError(ck_path)

    tmp = _prepare_adapter_dir(adapter_root, ck_path)
    model = tokenizer = None
    try:
        model, tokenizer = load(
            str(base_model),
            adapter_path=str(tmp),
            lazy=lazy_load,
        )
        rows = []
        for i, (user_msg, gold) in enumerate(fir_cases):
            messages = [
                {"role": "system", "content": SLIM_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            text = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                max_kv_size=max_kv_size,
                verbose=False,
            )
            parsed_ok = False
            pred_events = None
            try:
                start = text.find("{")
                end = text.rfind("}")
                if start >= 0 and end > start:
                    pred = json.loads(text[start : end + 1])
                    pred_events = pred.get("events")
                    parsed_ok = isinstance(pred_events, list)
            except json.JSONDecodeError:
                pass
            gold_n = len((gold or {}).get("events") or [])
            pred_n = len(pred_events) if isinstance(pred_events, list) else None
            rows.append(
                {
                    "case_index": i,
                    "gold_event_count": gold_n,
                    "pred_event_count": pred_n,
                    "json_object_ok": parsed_ok,
                    "response_chars": len(text),
                    "response_preview": text[:1200],
                }
            )
        return {"checkpoint": checkpoint_name, "cases": rows}
    finally:
        if model is not None and tokenizer is not None:
            _unload(model, tokenizer)
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LoRA checkpoints on shared FIR prompts.")
    parser.add_argument(
        "--base-model",
        type=Path,
        default=REPO_ROOT / "training/models/qwen3-14b-4bit",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=REPO_ROOT / "training/adapters/legal-qwen3-14b",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=REPO_ROOT / "training/datasets/final/valid.jsonl",
        help="JSONL with Chat messages; uses first N user rows starting with FIR:",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["0001400_adapters.safetensors", "0001800_adapters.safetensors"],
    )
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=4096,
        help="Cap KV cache size (lowers peak Metal memory for long contexts).",
    )
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        help="Pass lazy=True to mlx_lm.load (smaller initial allocation).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write full JSON report (default: stdout only)",
    )
    args = parser.parse_args()

    fir_cases = _fir_prompts_from_jsonl(args.jsonl, args.max_prompts)
    if not fir_cases:
        raise SystemExit(f"No FIR: prompts found in {args.jsonl}")

    results = []
    for ck in args.checkpoints:
        print(f"[compare] loading + generating: {ck}", flush=True)
        results.append(
            run_checkpoint(
                base_model=args.base_model,
                adapter_root=args.adapter_dir,
                checkpoint_name=ck,
                fir_cases=fir_cases,
                max_tokens=args.max_tokens,
                max_kv_size=args.max_kv_size,
                lazy_load=args.lazy_load,
            )
        )
        _clear_metal()

    # Side-by-side summary
    print("\n" + "=" * 72)
    print("Checkpoint comparison (same prompts)")
    print("=" * 72)
    for ck_result in results:
        ck = ck_result["checkpoint"]
        ok = sum(1 for r in ck_result["cases"] if r["json_object_ok"])
        mae = sum(
            abs((r["pred_event_count"] or 0) - r["gold_event_count"])
            for r in ck_result["cases"]
            if r["pred_event_count"] is not None
        )
        denom = sum(1 for r in ck_result["cases"] if r["pred_event_count"] is not None)
        print(f"\n{ck}")
        print(f"  JSON parse OK: {ok}/{len(ck_result['cases'])}")
        if denom:
            print(f"  Event-count MAE (parsed only): {mae / denom:.2f}")
        for r in ck_result["cases"]:
            print(
                f"    case {r['case_index']}: gold_events={r['gold_event_count']} "
                f"pred={r['pred_event_count']} json_ok={r['json_object_ok']} "
                f"chars={r['response_chars']}"
            )

    out = {
        "base_model": str(args.base_model),
        "adapter_dir": str(args.adapter_dir),
        "jsonl": str(args.jsonl),
        "max_tokens": args.max_tokens,
        "results": results,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nFull report: {args.report}")
    else:
        print("\n(raw JSON available via --report PATH)")


if __name__ == "__main__":
    main()
