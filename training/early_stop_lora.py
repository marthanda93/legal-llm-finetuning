#!/usr/bin/env python3
"""Run `mlx_lm.lora` with live val-loss-based early stopping.

`mlx_lm.lora` itself does not support early stopping; the v1 run kept
training past the val-loss minimum (iter 1400, val=0.280) all the way to
iter 1825 (val=0.527) before we manually killed it. This wrapper:

    1. Spawns `mlx_lm.lora --config <path>` as a subprocess.
    2. Tees stdout/stderr to <log_path> in real time.
    3. Watches each `Iter N: Val loss V` line as it streams.
    4. After `--patience` consecutive val-loss regressions vs the running
       best, sends SIGINT to the trainer so the LATEST checkpoint already
       on disk is the last good one.
    5. Leaves all checkpoints in place; pair with
       `training/select_best_checkpoint.py --promote` to copy the actual
       best one to `adapters.safetensors` before fusing.

Patience semantics:
    --patience 2 means "stop after the 2nd consecutive regression". So if
    the val curve is 0.40 -> 0.30 (best) -> 0.32 -> 0.34 -> 0.36, we stop
    AFTER seeing 0.36 (regression streak = 3 >= patience+1 in code, but we
    require strictly increasing for `patience` consecutive evals). Default
    is 2 to tolerate a single noisy bump.

Usage:
    python -m training.early_stop_lora \
        --config training/logs/lora_config.v2.yaml \
        --log-file training/logs/lora-v2.log \
        --patience 2

Exit codes:
    0  - mlx_lm.lora exited cleanly OR we triggered an early stop
    1  - mlx_lm.lora exited non-zero (training error)
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

VAL_RE = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([0-9]+\.[0-9]+)", re.IGNORECASE)


class EarlyStopState:
    def __init__(self, patience: int, min_iter: int) -> None:
        self.patience = patience
        self.min_iter = min_iter
        self.best_val: Optional[float] = None
        self.best_iter: Optional[int] = None
        self.consecutive_regressions = 0
        self.history: List[tuple] = []
        self.should_stop = False
        self.stop_reason = ""

    def observe(self, iter_n: int, val: float) -> None:
        self.history.append((iter_n, val))
        if iter_n < self.min_iter:
            return
        if self.best_val is None or val < self.best_val:
            self.best_val = val
            self.best_iter = iter_n
            self.consecutive_regressions = 0
            return
        # val >= best_val (regression or plateau)
        self.consecutive_regressions += 1
        if self.consecutive_regressions >= self.patience:
            self.should_stop = True
            self.stop_reason = (
                f"{self.consecutive_regressions} consecutive val regressions "
                f"vs best (iter {self.best_iter}, val {self.best_val:.4f})"
            )


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to mlx_lm.lora YAML config")
    parser.add_argument("--log-file", type=Path, required=True,
                        help="Where to tee stdout/stderr from the trainer")
    parser.add_argument("--patience", type=int, default=2,
                        help="Stop after N consecutive val-loss regressions")
    parser.add_argument("--min-iter", type=int, default=400,
                        help="Don't trigger early-stop before this iteration "
                             "(noisy first few evals can produce false positives)")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter to invoke mlx_lm.lora with")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[early-stop] config not found: {args.config}", file=sys.stderr)
        return 1
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    state = EarlyStopState(patience=args.patience, min_iter=args.min_iter)

    cmd = [args.python, "-m", "mlx_lm", "lora", "--config", str(args.config)]
    print(f"[early-stop] {_stamp()} launching: {' '.join(cmd)}")
    print(f"[early-stop] {_stamp()} log -> {args.log_file}")
    print(f"[early-stop] {_stamp()} patience={args.patience}, min-iter={args.min_iter}")

    log_fh = args.log_file.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        env=os.environ.copy(),
    )

    sigint_sent = False

    def _stop_trainer(reason: str) -> None:
        nonlocal sigint_sent
        if sigint_sent or proc.poll() is not None:
            return
        sigint_sent = True
        print(f"\n[early-stop] {_stamp()} STOPPING TRAINER: {reason}")
        # SIGINT lets mlx_lm flush its current adapter checkpoint.
        try:
            proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass

    def _drain():
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            log_fh.write(raw_line)
            log_fh.flush()
            sys.stdout.write(raw_line)
            sys.stdout.flush()
            for m in VAL_RE.finditer(raw_line):
                iter_n, val = int(m.group(1)), float(m.group(2))
                state.observe(iter_n, val)
                tag = ""
                if state.best_iter == iter_n:
                    tag = " (new best)"
                elif state.consecutive_regressions:
                    tag = f" (regression streak={state.consecutive_regressions})"
                print(f"[early-stop] {_stamp()} eval iter={iter_n} val={val:.4f}{tag}")
                if state.should_stop:
                    _stop_trainer(state.stop_reason)

    drain_thread = threading.Thread(target=_drain, daemon=True)
    drain_thread.start()

    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        print(f"\n[early-stop] {_stamp()} received Ctrl-C; forwarding SIGINT to trainer")
        _stop_trainer("user interrupt")
        rc = proc.wait()
    finally:
        drain_thread.join(timeout=10)
        log_fh.close()

    print()
    print(f"[early-stop] {_stamp()} trainer exited rc={rc}")
    if state.history:
        print(f"[early-stop] val-loss curve ({len(state.history)} evals):")
        for i, v in state.history[-12:]:
            mark = "  <-- best" if i == state.best_iter else ""
            print(f"            iter={i:>5}  val={v:.4f}{mark}")
        print(f"[early-stop] best: iter={state.best_iter}  val={state.best_val:.4f}")
        print(f"[early-stop] next: python -m training.select_best_checkpoint "
              f"{args.log_file} --adapter-dir <adapter_path> --promote")

    if sigint_sent and rc != 0:
        # We deliberately stopped it — treat that as success.
        return 0
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
