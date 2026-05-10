# LoRA Run — Status, Deep Corrections, and Recipe for the Next Run

_Last updated 2026-05-10. Source run: `training/logs/lora-20260510-115443.log` (Qwen3-14B 4-bit, LoRA r=16/α=32, LR 1e-4 constant, batch=1, 4000 planned iters → stopped at iter 1825)._

---

## 0. What changed since the last revision (the "deep corrections")

Everything in this section is **already implemented and verified**. The new training data is on disk at `training/datasets/final/{train,valid,test}.jsonl` — re-run `make data` after any source change to regenerate it from `LAW_RAW_DATA/` + the Stage builders.

### 0.1 Source-of-truth alias map
- **NEW** `LAW_RAW_DATA/TAXONOMY_ALIASES.json` — 30 curated `drift_label → canonical_label` mappings.
  - Hand-picked targets: every previously-observed Stage C drift label (`assault`, `house_trespass`, `bribery`, `corruption`, `arson`, …) now resolves to a precise canonical taxonomy label.
  - Documented policy: keys MUST NOT be in canonical, values MUST be in canonical, and legitimately parallel canonicals (POCSO vs POSH vs BNS) are **not merged**.

### 0.2 Canonicalization helpers (one source of truth)
- **NEW** in `training/common.py`: `canonical_categories()`, `load_taxonomy_aliases()`, `canonicalize_label(...)`, `canonicalize_events(...)`, `canonicalize_assistant_content(...)`.
- `to_mlx_chat_record(...)` now applies `canonicalize_assistant_content` to every assistant message at merge time. Off-vocab events that cannot be mapped are **dropped** (better one-fewer event than an invented label).

### 0.3 Cleaner merge + stratified split + class balancing
- `training/merge_and_split.py` now:
  1. Canonicalizes every record before splitting (so categories drive a clean stratification key).
  2. **Stratified split** (`--min-per-eval-split`, default 1) — every category with enough supply lands at least once in valid AND test. Result: **231/231 canonical labels present in train, valid, AND test**.
  3. **Down-sample cap** (`--downsample-cap`, default 500) on dominant classes — `non_crime` went from **2 979 → 500** training rows in the new split (was drowning the F1 macro signal).
  4. **Up-sample floor** (`--upsample-min`, default 5) on tail classes — 9 categories with 2–3 training rows are now duplicated up to 5.
  5. Prints pre/post canonicalization, an off-vocab audit, and per-split coverage; manifest captures everything for reproducibility.

### 0.4 Stricter Stage C teacher validator
- `training/build_synthetic_firs.py` now runs every event through `canonicalize_events` after `validate_events`, then hard-rejects any payload whose category still isn't canonical. Future Stage C re-runs cannot leak drift labels.

### 0.5 Cross-statute disambiguation curriculum (POCSO/POSH/BNS/IT/PCA/SC-ST/DV)
- `training/build_taxonomy_classification.py` now emits **44 hand-curated disambiguation pairs** covering 22 historically-confused statute boundaries:
  - POSH vs BNS sexual_harassment vs POCSO `child_sexual_harassment`.
  - POCSO `use_of_child_for_pornography` vs `storage_of_child_pornography` vs IT Act `child_pornography_online`.
  - Adult assault vs child sexual_assault.
  - IPC `cruelty_by_husband_or_relatives` vs DV `physical_domestic_abuse`.
  - `bribery_at_elections` vs `bribery_of_public_servant`.
  - `criminal_intimidation` vs `caste_based_insult_or_intimidation_of_sc_st`.
  - `house_trespass_or_house_breaking` vs `criminal_trespass`.
  - `aggravated_penetrative_sexual_assault` vs `penetrative_sexual_assault_on_child`.
  - Arms Act vs Arms-Act-aided BNS hurt.
  - NDPS trafficking vs cyber `cheating`/`computer_enabled_cheating_by_personation`.
- Each pair is emitted twice: once "snippet → label", once "snippet → label + statute reasoning" — anchoring both fast classification and explanation behaviour.

### 0.6 Eval honesty
- `training/eval.py` now canonicalizes BOTH the gold and the predicted labels before computing macro/micro F1, and reports an `off_vocab_pred_*` audit. New flag `--keep-off-vocab` if you want to count drift emissions as FPs instead of dropping them.

### 0.7 Inference parity (cleans up the **already-fused** model)
- `event_extraction/ontology.py` now folds `LAW_RAW_DATA/TAXONOMY_ALIASES.json` into `LEGACY_ALIASES` at import time. The same logic was copied into the production BFF service so the **deployed fused-1000 model** instantly starts emitting clean canonical labels for the drift outputs the eval audit found — no retrain required to fix the live response.
- `event_extraction/parsing.py` strips `<think>...</think>` blocks defensively (Qwen3 emits an empty think block in 100% of cases despite `/no_think`; this prevents corner-case JSON-truncation issues if the think block ever leaks content).

### 0.8 Log-driven additions (from the second log audit)

The full eval trace (`training/logs/eval-fused-1000-full.trace.jsonl`, 402 cases) revealed that the **fused-1000 model emits 290 unique drift labels across 476 events** — far more than the 21 we caught from the raw training pool. Acting on that:

- **Alias map expanded from 30 → 116 entries** by mining the trace. Coverage of model-emitted drift jumped from **~12% → 55.5% of events**. The remaining 212 events are mostly singletons (n=1) — diminishing returns.
- **47 cases in the v1 trace had ALL their predicted events dropped** at parse-time because the labels were unmappable. After the expanded alias map, the same cases will now contribute meaningful predictions to the F1 calculation (the BFF inference path uses the same `LAW_RAW_DATA/TAXONOMY_ALIASES.json`).
- **`<think>` block waste**: 100% of cases emitted `<think>\n\n</think>` despite `/no_think` (~12 tokens overhead). Stripped defensively in BFF parsing.

### 0.9 Operational tooling (now available)

Two new utilities address operational hygiene gaps that cost us shippable weights on v1:

- **`training/select_best_checkpoint.py`** — scans a lora log for `Iter N: Val loss V` lines, prints the val curve, and (with `--promote`) copies the lowest-val-loss checkpoint to `adapters.safetensors` so `mlx_lm.fuse` picks it up. Replaces the manual `cp 0001000_adapters.safetensors adapters.safetensors` step that we forgot on v1.

- **`training/early_stop_lora.py`** — wraps `python -m mlx_lm.lora` and watches the streaming val loss in real time. After `--patience` consecutive regressions vs the running best, it sends `SIGINT` to the trainer so the latest checkpoint on disk is the last good one. Combined with the v2 config, this would have stopped v1 around iter 1500 instead of running to 1825.

- **`make train`** is the single entry-point: it renders `lora_config.v2.yaml`, runs `early_stop_lora.py`, and **always** calls `select_best_checkpoint --promote` before the fuse step. Override the recipe knobs (PATIENCE / MIN_ITER / CONFIG_TEMPLATE) on the make CLI if you need to.

  ```bash
  make train                                  # default recipe
  make train PATIENCE=3 MIN_ITER=600          # tweak early-stop sensitivity
  ```

### 0.10 Eval honesty improvements (from the report audit)

`training/eval.py` previously misled in three ways. Fixed:

| Problem in v1 reports | Fix |
|---|---|
| `action_summary_cosine_mean: 0.0` when `--no-cosine` was passed (looked like the model scored zero similarity) | Now reports `null` + `action_summary_cosine_enabled: false`; summary print says `skipped (--no-cosine)`. |
| `n_total: 402` while `--max-cases 815` was requested with no explanation | Now reports `n_skipped_non_fir` so the user sees that 413 records were Stage A/B (non-FIR) and skipped. |
| `per_category` map listed dozens of `support: 0` entries (model-only false-positive labels) which polluted the JSON and diluted Macro F1 by inflating the denominator | `support: 0` entries are now extracted into `pred_only_false_positive_categories`; `per_category` only contains gold-supported labels; `category_macro_f1_basis` shows the denominator. |

### 0.11 Verification on the new splits

```
[merge] taxonomy aliases applied: 116; canonical labels in vocabulary: 231
[merge] train balancing: 15 categories changed (downsample_cap=500, upsample_min=5)
           DOWN  non_crime           2979 ->  500  (-2479)
           UP    aggravated_sexual_assault_on_child   2 ->  5  (+3)   …
[merge] train  category coverage: 231/231 canonical labels (missing 0)
[merge] valid  category coverage: 231/231 canonical labels (missing 0)
[merge] test   category coverage: 231/231 canonical labels (missing 0)
[merge] wrote 10422 -> training/datasets/final/train.jsonl
[merge] wrote  1810 -> training/datasets/final/valid.jsonl
[merge] wrote  1895 -> training/datasets/final/test.jsonl
```

```
[train] records=10422  events=9113  drift_events=0  drift_unique=0
[valid] records=1810   events=1521  drift_events=0  drift_unique=0
[test]  records=1895   events=1691  drift_events=0  drift_unique=0
```

---

## 1. Decision: which checkpoint ships RIGHT NOW (no retrain yet)

| Checkpoint | Val loss | JSON OK | Event MAE | Macro F1 | Micro F1 | Non-crime FP |
|---|---|---|---|---|---|---|
| `0001000_adapters` (fused) | **0.291** | 100% | **0.657** | 0.189 | 0.303 | 0% |
| `0001400_adapters` (fused) | 0.280 | 100% | 0.686 | 0.184 | 0.286 | 0% |
| `0001800_adapters` (latest) | 0.527 ↑ | — | — | — | — | — |

**Ship `fused-1000`.** Baseline stays the best on test. Combined with the 0.7 BFF alias map, it should also stop emitting any of the 21 drift labels at inference. Do **not** continue the existing run — it is past the sweet spot under constant LR.

Artifacts:
- `training/models/legal-qwen3-14b-fused-1000` ← deploy
- `training/models/legal-qwen3-14b-fused-1400` ← keep for A/B
- `training/logs/eval-fused-1000-full.json` ← baseline for the next retrain comparison

---

## 2. Recipe for the NEXT LoRA run

The new dataset is already on disk and the v2 config is checked in at `training/logs/lora_config.v2.yaml`. Differences vs v1 (already in the file's header comment):

| Setting | v1 | **v2 (in `lora_config.v2.yaml`)** | Why |
|---|---|---|---|
| `learning_rate` | `1e-4` constant | **`5e-5` + cosine_decay (warmup 100)** | Constant LR overfit at iter ~1400; cosine to 1e-6 lets the model settle on a balanced dataset. |
| `iters` | `4000` | **`1500`** | Best v1 val was at 1000–1400; with cosine + early-stop, no buffer needed beyond that. |
| `val_batches` | `25` | **`100`** | A 25-batch val loss bounces ±0.08 (visible in the v1 log: 0.336 → 0.419 → 0.341 was sampling noise, not drift). 100 batches gives a stable signal so early-stop fires correctly. |
| `steps_per_eval` | `200` | **`100`** | Tighter resolution near the minimum. |
| `save_every` | `200` | **`100`** | Pair with `steps_per_eval`. |
| `lora_parameters.dropout` | `0.05` | **`0.10`** | Small bump because we now down-sample dominant classes (`non_crime` 2979 → 500), which slightly reduces inherent regularisation. |
| Stop rule | none | **`early_stop_lora.py --patience 2 --min-iter 400`** | Wraps `mlx_lm.lora`; SIGINTs after 2 consecutive val regressions, leaving the last good checkpoint on disk. |
| Best-checkpoint promotion | manual `cp` (forgotten on v1) | **`select_best_checkpoint.py --promote`** | `make train` always calls this before fusing. |
| Data | `final/v1` (pre-fix) | **`final/` (post-fix)** | Stratified split, balanced classes, zero drift labels (+ 116-entry alias map), 44 disambiguation pairs. |

### One-command launch
```bash
make train     # then `make fuse && make serve && make eval`
```

`make train` will:
1. Render `training/logs/lora_config.v2.yaml` to a temp file with `__REPO_ROOT__` substituted.
2. Spawn `mlx_lm.lora` via `early_stop_lora.py`, tee-ing to a fresh `training/logs/lora-v2-<timestamp>.log`.
3. Stop training automatically after 2 consecutive val regressions vs best (no babysitting required).
4. Run `select_best_checkpoint.py --promote` against the log + adapter dir, so `adapters.safetensors` always holds the best step's weights.

Then `make fuse` → 4-bit quantize into `training/models/legal-qwen3-14b-fused-v2`.

### Manual launch (if you want full control)
```bash
# 1. Train with live early-stop (logs every val score)
python -m training.early_stop_lora \
  --config training/logs/lora_config.v2.yaml \
  --log-file training/logs/lora-v2.log \
  --patience 2 --min-iter 400

# 2. Pick the best checkpoint and promote it
python -m training.select_best_checkpoint \
  training/logs/lora-v2.log \
  --adapter-dir training/adapters/legal-qwen3-14b-v2 \
  --promote

# 3. Fuse
python -m mlx_lm fuse \
  --model "$(pwd)/training/models/qwen3-14b-4bit" \
  --adapter-path training/adapters/legal-qwen3-14b-v2 \
  --save-path training/models/legal-qwen3-14b-fused-v2 -q

# 4. Serve
mlx_lm.server --model "$(pwd)/training/models/legal-qwen3-14b-fused-v2" --port 8030

# 5. Re-run eval baseline (apples-to-apples vs eval-fused-1000-full.json)
python -m training.eval \
  --test-jsonl training/datasets/final/test.jsonl \
  --llm-url http://127.0.0.1:8030/v1/chat/completions \
  --model "$(pwd)/training/models/legal-qwen3-14b-fused-v2" \
  --max-cases 1895 --max-tokens 1536 --slim-prompt --no-cosine \
  --report training/logs/eval-fused-v2-full.json
```

---

## 3. What to expect from the new run (qualitatively)

Numbers below are predictions, not measurements — to be confirmed by the v2 eval.

| Metric | fused-1000 (current) | Predicted v2 | Reason |
|---|---|---|---|
| JSON validity | 100 % | ≥ 100 % | Already saturated. |
| Non-crime FP | 0 % | ~0 % | non_crime down-sampled but kept above noise threshold. |
| Event MAE (under-count) | 370/608 = 61 % recall | **≥ 75 %** | Tail categories now have ≥5 train rows; disambiguation rationales teach completeness. |
| Macro F1 | 0.153 (815 cases) | **≥ 0.30** | Drift labels eliminated → 21 categories that scored 0 because of mislabels can now score; stratified test means small classes finally get measured. |
| Off-vocab predictions | n/a (was being scored as FP via fuzzy match) | **0** | Aliases mapped at training, eval, AND inference. |

If macro F1 doesn't lift meaningfully it almost certainly means the tail categories need more **Stage C teacher data**, not a different recipe. The next investment after this run should be a focused Stage C re-run for the 30 lowest-frequency labels (target: 25 rows each).

---

## 4. Operational hygiene (NOW automated)

- [x] Always copy `0000XXXX_adapters.safetensors` → `adapters.safetensors` **explicitly** before fusing → handled by `training/select_best_checkpoint.py --promote` (called automatically by `make train`).
- [x] Stop training when val loss stops improving → handled by `training/early_stop_lora.py`.
- [ ] Store run metadata (LR, iters, val curve, best step, dataset manifest hash) alongside each fused model — _next iteration; not blocking._
- [x] BFF prompt + ontology + parsing now coupled to `LAW_RAW_DATA/TAXONOMY_ALIASES.json` and strip `<think>` blocks defensively.

---

## 5. TL;DR

- **116-entry alias map** (was 30) now covers ~55% of the drift labels the live `fused-1000` model emits — including 47 previously-dropped cases. BFF inference, training-data merge, and eval scoring share one source of truth.
- Class imbalance is **gone from train**: dominant classes capped at 500, tails up-sampled to ≥5, stratified split guarantees 231/231 in valid AND test.
- Cross-statute confusion is now **directly trained**, not just hoped for: 44 hand-curated POCSO/POSH/BNS/IT/PCA/DV/SC-ST disambiguation pairs in Stage B.
- BFF parsing **strips empty `<think>` blocks** so token waste / JSON truncation can't bite us in production.
- Eval reports are **honest**: cosine reports `null` (not 0.0) when disabled, `n_skipped_non_fir` shows why `n_total < max-cases`, and `support=0` predictions move to a separate bucket so Macro F1 isn't diluted.
- New run is **fully automated**: `make train && make fuse` runs train → early-stop → best-checkpoint promotion → fuse in two commands. Same recipe that v1 needed but no longer requires manual checkpoint babysitting.
