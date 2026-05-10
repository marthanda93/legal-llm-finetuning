# legal-llm-finetuning

Stand-alone LoRA fine-tuning workbench for an Indian-criminal-law FIR
event extractor. Forked out of the SwadesiLegalCopilot product repo so
training, eval, dataset generation and adapter A/B testing can iterate
independently of the production BFF service.

The shipping artefact is a 4-bit fused **Qwen3-14B** model that, given an
FIR text, emits a single JSON object: `fir_text_categories[]` plus an
`events[]` array where every `crime_category` is one of the **231
canonical labels** in `LAW_RAW_DATA/CRIME_TAXONOMY.json`.

```
              +-----------+    +-------------+    +-----------+
LAW_RAW_DATA  | Stage A/B | -> | merge/split | -> | LoRA fine | -> fused .safetensors
   +          | (template)|    | + balance + |    | tune  on  |    + 4-bit serve via mlx_lm.server
teacher LLM   | Stage C/D/E    | stratify    |    | Qwen3-14B |
              +-----------+    +-------------+    +-----------+
                                                       |
                                                       v
                                                 training.eval
```

---

## Quick start

```bash
# 0. one-time setup
make install                       # creates .venv, installs requirements + this pkg in -e mode
make convert                       # downloads + 4-bit-quantises Qwen3-14B (~30GB on disk)

# 1. (re)build the dataset from raw sources (10422 train / 1810 valid / 1895 test)
make data                          # runs Stage A/B/D/E builders + merge_and_split
make verify                        # asserts zero off-vocab labels + 231/231 coverage

# 2. train (cosine LR, val_batches=100, early-stop on 2 consecutive regressions)
make train                         # auto-promotes the best checkpoint to adapters.safetensors

# 3. fuse + serve + eval
make fuse                          # adapter -> fused 4-bit model
make serve                         # mlx_lm.server :8030 (separate terminal)
make eval                          # writes training/logs/eval-fused-v2-full.json

# common helpers
make help                          # list every target
make status                        # show what artefacts exist + sizes
make compare A=0001000 B=0001400   # side-by-side checkpoint A/B
```

If you need Stage C (synthetic FIRs from a teacher LLM), set one of
`OPENAI_API_KEY` / `GOOGLE_API_KEY` / `ANTHROPIC_API_KEY` /
`OPENROUTER_API_KEY` and run `make stage-c`.

---

## Repository layout

```
.
├── Makefile                       # ALL training operations are routed through this
├── README.md                      # you are here
├── requirements.txt               # mlx, mlx-lm, httpx, optional sentence-transformers
├── pyproject.toml                 # editable-install metadata (`pip install -e .`)
│
├── LAW_RAW_DATA/                  # 23 MB — pure data
│   ├── CRIME_TAXONOMY.json        # source of truth: 231 canonical crime categories
│   ├── TAXONOMY_ALIASES.json      # 116 drift -> canonical mappings (used at train + eval + inference)
│   └── DEFINITIONS/               # 21 statute JSON files, ~2 339 defined terms (Stage A/E source)
│
├── event_extraction/              # 24 KB — inference-side contract (3 modules, no extra deps)
│   ├── ontology.py                # builds VALID_CATEGORIES from CRIME_TAXONOMY.json + folds in TAXONOMY_ALIASES
│   ├── parsing.py                 # strips <think>...</think>, parses JSON
│   ├── validation.py              # validate_events: applies aliases + drops off-vocab events
│   └── __init__.py
│
└── training/                      # 103 MB — code + datasets + historical logs
    ├── common.py                  # shared helpers: SLIM_EXTRACTION_SYSTEM_PROMPT, canonicalize_*, TrainingRecord
    ├── build_definition_qa.py     # Stage A
    ├── build_taxonomy_classification.py  # Stage B (incl. POCSO/POSH/BNS disambiguation pairs)
    ├── build_synthetic_firs.py    # Stage C (teacher LLM)
    ├── build_hard_negatives.py    # Stage D
    ├── build_definition_grounded.py      # Stage E
    ├── teacher_client.py          # multi-provider LLM client (OpenAI / Gemini / Anthropic / OpenRouter)
    ├── merge_and_split.py         # canonicalize, stratify, balance, write final/{train,valid,test}.jsonl
    ├── verify_dataset.py          # `make verify` — drift-label + coverage assertions
    ├── early_stop_lora.py         # wraps mlx_lm.lora; SIGINTs on N consecutive val regressions
    ├── select_best_checkpoint.py  # parses lora log; --promote copies best ckpt to adapters.safetensors
    ├── compare_adapter_checkpoints.py    # side-by-side inference A/B
    ├── eval.py                    # eval harness: JSON validity, MAE, P/R/F1, per-case trace
    ├── NEXT_STEPS.md              # full audit + recipe rationale (read this if curious)
    │
    ├── datasets/
    │   ├── stages_raw/            # Stage A/B/C/D/E raw outputs (regenerable)
    │   └── final/                 # current train/valid/test JSONL + manifest.json
    │
    ├── logs/
    │   ├── lora_config.yaml       # legacy v1 recipe (with __REPO_ROOT__ placeholder)
    │   ├── lora_config.v2.yaml    # current recommended recipe
    │   ├── lora-*.log             # training logs (timestamped)
    │   ├── eval-fused-*.json      # eval reports
    │   └── eval-fused-*.trace.jsonl  # per-case input/output trace (large)
    │
    ├── adapters/                  # NOT in git: LoRA weight checkpoints (1+ GB)
    └── models/                    # NOT in git: base + fused MLX 4-bit models (30+ GB)
```

This is a **training workbench** — it deliberately does **not** contain
the full inference stack (FastAPI extractor, few-shot retriever, embedding
DB). Only the three lightweight `event_extraction/` modules are shipped
because training data needs to be validated against the same taxonomy
contract the production model emits at inference time.

The `__REPO_ROOT__` placeholder in `training/logs/lora_config*.yaml` is
substituted at run time by `make train` (target `render-config`) so the
files are portable across machines.

---

## Training data pipeline

| Stage | Source | Records (typical) | Needs API | Script |
|---|---|---|---|---|
| **A — Definition QA** | `LAW_RAW_DATA/DEFINITIONS/*.json` | ~6 000 (capped) | no | `build_definition_qa.py` |
| **B — Taxonomy classify** | `CRIME_TAXONOMY.json` + 44 hand-curated disambiguation pairs | ~2 100 | no | `build_taxonomy_classification.py` |
| **C — End-to-end FIR -> events** | Teacher LLM | ~4 500 | **yes** | `build_synthetic_firs.py` |
| **D — Hard negatives** | non-crime + civil templates | ~3 700 | no | `build_hard_negatives.py` |
| **E — Definition-grounded** | DEFINITIONS + canonical mappings | ~170 | no | `build_definition_grounded.py` |
| **Merge & split** | dedupe, **canonicalize via TAXONOMY_ALIASES.json**, stratified split, balance | 10 422 / 1 810 / 1 895 | no | `merge_and_split.py` |

Output: `training/datasets/final/{train,valid,test}.jsonl` (mlx_lm.lora
chat format) plus `manifest.json` capturing the seed, splits, alias map
size, balancing knobs, and per-split coverage report.

---

## Training recipe (v2)

`training/logs/lora_config.v2.yaml` (substitute `__REPO_ROOT__`):

| Setting | Value | Why |
|---|---|---|
| `learning_rate` | `5e-5` + `cosine_decay` (warmup 100, end 1e-6) | v1 constant 1e-4 overfit at iter ~1400; cosine + lower peak suits the now-balanced dataset. |
| `iters` | 1500 | Best v1 val was at iter 1000-1400; with early-stop, no need for headroom. |
| `val_batches` | 100 | 25 was too noisy (±0.08 sampling jitter); 100 stabilises early-stop. |
| `steps_per_eval` | 100 | Tighter resolution near the optimum. |
| `lora_parameters.dropout` | 0.10 | Slight bump because we down-sample dominant classes. |
| Stop rule | `early_stop_lora.py --patience 2 --min-iter 400` | SIGINT after 2 consecutive val regressions. |
| Best-ckpt promotion | `select_best_checkpoint.py --promote` (auto-called by `make train`) | mlx_lm overwrites `adapters.safetensors` with the LATEST step; this preserves the BEST. |

---

## Eval recipe

```bash
make serve   # in terminal A
make eval    # in terminal B
```

`make eval` writes `training/logs/eval-fused-v2-full.json` plus a
side-by-side `.trace.jsonl` (system prompt + user prompt + raw response +
gold + pred for each case, flushed per case so you can `tail -f`).

Reported metrics:

- `json_validity_rate` — % responses that parse + pass `validate_events`
- `event_count_mae` — mean abs err of `len(pred_events) - len(gold_events)`
- `category_macro_f1` / `_micro_f1` — over the 231-label taxonomy, gold-supported labels only
- `category_macro_f1_basis` — denominator (so the metric is comparable across runs)
- `non_crime_false_positive_rate` — % of zero-event golds where the model emitted ≥1 event
- `off_vocab_pred_total` / `_top` — labels the model emitted that didn't survive `canonicalize_label`
- `pred_only_false_positive_categories` — model-only labels (no gold support); kept separate so they don't dilute Macro F1
- `action_summary_cosine_mean` — `null` when `--no-cosine` is set (was misleadingly reported as 0.0 before)

---

## Notable design decisions

1. **Single source of truth for taxonomy aliases.** `LAW_RAW_DATA/TAXONOMY_ALIASES.json` (116 entries) is consumed at three layers:
   - `training/common.py` — applied during `merge_and_split` so off-vocab events never enter training
   - `training/eval.py` — applied to gold + predictions before F1
   - `event_extraction/ontology.py` — folded into `LEGACY_ALIASES` and copied verbatim into the production BFF service so the live model's drift outputs get cleaned at inference

2. **Stratified split, not random.** `merge_and_split.py` reserves at least N examples per class for valid + test, then splits the remainder. Result: 231/231 canonical labels appear in *all* three splits.

3. **Class balancing on train only.** `--downsample-cap 500` caps `non_crime` (was 2 979) and `criminal_intimidation`; `--upsample-min 5` duplicates tail classes. Eval splits are never touched.

4. **`<think>...</think>` stripped defensively in BFF parsing.** Qwen3 emits an empty think block in 100% of responses despite `/no_think`; stripping prevents corner-case JSON truncation.

5. **Best-checkpoint promotion is automatic.** mlx_lm.lora overwrites `adapters.safetensors` with the LATEST step's weights. `select_best_checkpoint.py --promote` (called by `make train`) copies the lowest-val-loss `<N>_adapters.safetensors` over it before fusing.

See `training/NEXT_STEPS.md` for the full audit log + the rationale behind every alias and recipe knob.

---

## Conventions

- All paths in committed YAML configs use the `__REPO_ROOT__` placeholder; the Makefile substitutes the absolute path at render time.
- All training records follow the **mlx_lm chat format**: each line is a `{"messages": [{"role": "system|user|assistant", "content": "..."}, ...]}` JSON object.
- `crime_category` strings are always **snake_case** and must exist in `LAW_RAW_DATA/CRIME_TAXONOMY.json` (or be folded by the alias map).
- LoRA adapter checkpoints live under `training/adapters/<run-name>/0000XXX_adapters.safetensors`; never edit `adapters.safetensors` by hand — let `make promote-best` do it.
- Logs in `training/logs/` may contain absolute paths from previous machines — they're historical artefacts and intentionally not rewritten.
