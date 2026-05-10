#!/usr/bin/env python3
"""Pick the best LoRA checkpoint from a training log and (optionally) promote it.

mlx_lm.lora always overwrites `adapters.safetensors` with the LATEST step's
weights, even if those weights are demonstrably worse than an earlier
checkpoint. That cost us a usable shipped model on the v1 run: the best val
loss was at iter 1400 but `adapters.safetensors` ended up holding iter 1825
weights when we stopped the job.

This helper:
    1. Parses the lora log for `Iter <N>: Val loss <V>` lines.
    2. Prints the (val loss, iter) ranking.
    3. If --promote is set, copies the corresponding
       `<N>_adapters.safetensors` to `adapters.safetensors` so the next
       `mlx_lm.fuse` picks up the correct weights.

Usage:
    python -m training.select_best_checkpoint training/logs/lora-v2.log
    python -m training.select_best_checkpoint training/logs/lora-v2.log \
        --adapter-dir training/adapters/legal-qwen3-14b-v2 --promote

Exit codes:
    0  - best checkpoint identified (and promoted if requested)
    1  - log unparseable / no val-loss entries found
    2  - promote requested but the matching .safetensors file does not exist
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

VAL_RE = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([0-9]+\.[0-9]+)", re.IGNORECASE)


def _parse_log(log_path: Path) -> List[Tuple[int, float]]:
    """Return [(iter, val_loss), ...] in chronological order."""
    out: List[Tuple[int, float]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            for m in VAL_RE.finditer(line):
                out.append((int(m.group(1)), float(m.group(2))))
    return out


def _format_table(entries: List[Tuple[int, float]], best_iter: int) -> str:
    width = max(len("ITER"), max(len(str(i)) for i, _ in entries))
    rows = ["  {:>{w}}  {:>10}  {}".format("ITER", "VAL_LOSS", "", w=width)]
    rows.append("  " + "-" * (width + 16))
    for i, v in entries:
        marker = "<-- best" if i == best_iter else ""
        rows.append("  {:>{w}}  {:>10.4f}  {}".format(i, v, marker, w=width))
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Path to mlx_lm.lora log file")
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Adapter directory containing 0000XXX_adapters.safetensors files",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Copy the best checkpoint to adapters.safetensors (requires --adapter-dir)",
    )
    parser.add_argument(
        "--min-iter",
        type=int,
        default=0,
        help="Ignore checkpoints before this iteration (avoid early noisy minima)",
    )
    args = parser.parse_args()

    if not args.log.exists():
        print(f"[select] log file not found: {args.log}", file=sys.stderr)
        return 1

    entries = _parse_log(args.log)
    entries = [(i, v) for i, v in entries if i >= args.min_iter]
    if not entries:
        print(f"[select] no `Iter N: Val loss V` entries in {args.log}", file=sys.stderr)
        return 1

    best_iter, best_val = min(entries, key=lambda kv: kv[1])
    print(_format_table(entries, best_iter))
    print()
    print(f"[select] best checkpoint: iter {best_iter}  (val_loss={best_val:.4f})")

    if args.promote:
        if args.adapter_dir is None:
            print("[select] --promote requires --adapter-dir", file=sys.stderr)
            return 2
        src = args.adapter_dir / f"{best_iter:07d}_adapters.safetensors"
        dst = args.adapter_dir / "adapters.safetensors"
        if not src.exists():
            print(f"[select] checkpoint file not found: {src}", file=sys.stderr)
            print("[select] (mlx_lm.lora may have saved with a different stride; "
                  "check `save_every` in your config)", file=sys.stderr)
            return 2
        shutil.copy2(src, dst)
        print(f"[select] promoted: {src.name}  ->  adapters.safetensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
