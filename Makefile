# =============================================================================
# legal-llm-finetuning — workbench Makefile
#
# One-stop interface to the LoRA training pipeline.
#
# Common workflows:
#
#   First time on a fresh machine:
#     make install          # pip install everything
#     make convert          # download + 4-bit quantize Qwen3-14B (~30GB on disk)
#     make data             # rebuild training/datasets/{stages_raw,final} from raw sources
#     make verify           # confirm zero off-vocab labels in the splits
#     make train            # v2 recipe: cosine LR + early-stop + auto best-checkpoint promotion
#     make fuse             # merge adapter into the base model and 4-bit quantize
#     make serve            # mlx_lm.server on :8030
#     make eval             # run the eval harness against the served model
#
#   Re-train after data fix:
#     make data verify train fuse eval
#
#   Compare two adapter checkpoints:
#     make compare A=0001000 B=0001400
#
# Override anything via `make VAR=value`, e.g. `make train ITERS=2000`.
# =============================================================================

REPO_ROOT  := $(abspath $(CURDIR))
PYTHON     ?= python3
VENV_DIR   ?= $(REPO_ROOT)/.venv
PIP        ?= $(VENV_DIR)/bin/pip
# Use the venv interpreter if it exists, else fall back to system python3.
# Override at the CLI:  make eval PY=/path/to/python
PY         ?= $(shell test -x $(VENV_DIR)/bin/python && echo $(VENV_DIR)/bin/python || command -v $(PYTHON))

# Make `from training.* import` and `from event_extraction.* import` both
# resolve without requiring `pip install -e .` first (handy for CI / first-run).
export PYTHONPATH := $(REPO_ROOT):$(PYTHONPATH)

# ---- model + base paths ----
BASE_HF        ?= Qwen/Qwen3-14B
BASE_MLX_DIR   ?= $(REPO_ROOT)/training/models/qwen3-14b-4bit
ADAPTER_DIR    ?= $(REPO_ROOT)/training/adapters/legal-qwen3-14b-v2
FUSED_DIR      ?= $(REPO_ROOT)/training/models/legal-qwen3-14b-fused-v2
DATA_DIR       ?= $(REPO_ROOT)/training/datasets/final
LOG_DIR        ?= $(REPO_ROOT)/training/logs

# ---- recipe knobs (used only when CONFIG=lora_config.yaml is overridden) ----
CONFIG_TEMPLATE ?= $(LOG_DIR)/lora_config.v2.yaml
CONFIG_RENDERED ?= $(LOG_DIR)/.lora_config.rendered.yaml
PATIENCE   ?= 2
MIN_ITER   ?= 400

# ---- eval / serve knobs ----
SERVE_PORT ?= 8030
EVAL_MODEL ?= $(FUSED_DIR)
EVAL_REPORT ?= $(LOG_DIR)/eval-fused-v2-full.json
EVAL_MAX_CASES ?= 1895
EVAL_MAX_TOKENS ?= 1536

# ---- compare-checkpoints knobs ----
A ?= 0001000
B ?= 0001400

# Use bash with strict mode for recipe lines.
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------

.PHONY: help
help:  ## Show this help.
	@echo "legal-llm-finetuning — common targets"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Repo root: $(REPO_ROOT)"
	@echo "Base model: $(BASE_HF) -> $(BASE_MLX_DIR)"

# -----------------------------------------------------------------------------
# 0. Setup
# -----------------------------------------------------------------------------

.PHONY: venv
venv: $(VENV_DIR)/bin/python  ## Create a local Python virtualenv at .venv

$(VENV_DIR)/bin/python:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip wheel

.PHONY: install
install: venv  ## Install runtime + dev dependencies into .venv (idempotent).
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo
	@echo "[install] Done. Activate with:  source $(VENV_DIR)/bin/activate"

# -----------------------------------------------------------------------------
# 1. Data pipeline
# -----------------------------------------------------------------------------

.PHONY: stage-a
stage-a:  ## Build Stage A (definition QA) -> training/datasets/stages_raw/stage_a_definitions.jsonl
	$(PY) -m training.build_definition_qa

.PHONY: stage-b
stage-b:  ## Build Stage B (taxonomy classification + disambiguation pairs)
	$(PY) -m training.build_taxonomy_classification

.PHONY: stage-c
stage-c:  ## Build Stage C (synthetic FIRs via teacher LLM — needs API key)
	@if [ -z "$${OPENAI_API_KEY:-}$${GOOGLE_API_KEY:-}$${ANTHROPIC_API_KEY:-}$${OPENROUTER_API_KEY:-}" ]; then \
	  echo "ERROR: set OPENAI_API_KEY / GOOGLE_API_KEY / ANTHROPIC_API_KEY / OPENROUTER_API_KEY for Stage C"; \
	  exit 2; \
	fi
	$(PY) -m training.build_synthetic_firs

.PHONY: stage-d
stage-d:  ## Build Stage D (hard negatives — non-crime + civil templates)
	$(PY) -m training.build_hard_negatives

.PHONY: stage-e
stage-e:  ## Build Stage E (definition-grounded extraction examples)
	$(PY) -m training.build_definition_grounded

.PHONY: stages
stages: stage-a stage-b stage-d stage-e  ## Run all stages EXCEPT C (no API key required)
	@echo "[stages] A/B/D/E built. Run 'make stage-c' separately if you have a teacher LLM key."

.PHONY: merge
merge:  ## Merge stages -> train/valid/test JSONL (canonicalize, stratify, balance)
	$(PY) -m training.merge_and_split

.PHONY: data
data: stages merge  ## Rebuild Stage A/B/D/E + merge into final splits.

.PHONY: verify
verify:  ## Verify zero drift labels and 231/231 coverage in train/valid/test.
	$(PY) -m training.verify_dataset --data-dir $(DATA_DIR)

# -----------------------------------------------------------------------------
# 2. Base model conversion
# -----------------------------------------------------------------------------

.PHONY: convert
convert:  ## Download + 4-bit quantize the base model into training/models/qwen3-14b-4bit
	@if [ -d "$(BASE_MLX_DIR)" ]; then \
	  echo "[convert] $(BASE_MLX_DIR) already exists, skipping (delete to force re-download)."; \
	else \
	  $(PY) -m mlx_lm convert --hf-path $(BASE_HF) --mlx-path $(BASE_MLX_DIR) -q; \
	fi

# -----------------------------------------------------------------------------
# 3. Training (LoRA + early-stop + best-checkpoint promotion)
# -----------------------------------------------------------------------------

.PHONY: render-config
render-config:  ## Render $(CONFIG_TEMPLATE) -> $(CONFIG_RENDERED) with __REPO_ROOT__ substituted.
	@sed "s|__REPO_ROOT__|$(REPO_ROOT)|g" $(CONFIG_TEMPLATE) > $(CONFIG_RENDERED)
	@echo "[render-config] $(CONFIG_TEMPLATE) -> $(CONFIG_RENDERED)"
	@grep -E '^(model|data|adapter_path):' $(CONFIG_RENDERED) | sed 's/^/    /'

.PHONY: train
train: render-config  ## v2 LoRA training with live early-stop + auto best-ckpt promotion.
	@TS=$$(date +%Y%m%d-%H%M%S); LOG_FILE="$(LOG_DIR)/lora-v2-$$TS.log"; \
	echo "[train] log -> $$LOG_FILE"; \
	$(PY) -m training.early_stop_lora \
	    --config $(CONFIG_RENDERED) \
	    --log-file $$LOG_FILE \
	    --patience $(PATIENCE) --min-iter $(MIN_ITER); \
	echo "[train] promoting best checkpoint ..."; \
	$(PY) -m training.select_best_checkpoint $$LOG_FILE \
	    --adapter-dir $(ADAPTER_DIR) --promote || \
	    echo "[train] WARNING: best-checkpoint promotion failed (using whatever mlx_lm left)"

.PHONY: train-no-stop
train-no-stop: render-config  ## v2 training WITHOUT early-stop (for debugging).
	@TS=$$(date +%Y%m%d-%H%M%S); LOG_FILE="$(LOG_DIR)/lora-v2-noES-$$TS.log"; \
	echo "[train-no-stop] log -> $$LOG_FILE"; \
	$(PY) -m mlx_lm lora --config $(CONFIG_RENDERED) 2>&1 | tee $$LOG_FILE; \
	$(PY) -m training.select_best_checkpoint $$LOG_FILE \
	    --adapter-dir $(ADAPTER_DIR) --promote || true

.PHONY: best-ckpt
best-ckpt:  ## Print the best val-loss checkpoint from LATEST log; pass LOG=path to override.
	@LOG="$${LOG:-$$(ls -t $(LOG_DIR)/lora-*.log 2>/dev/null | head -1)}"; \
	if [ -z "$$LOG" ]; then echo "no logs found in $(LOG_DIR)" >&2; exit 1; fi; \
	$(PY) -m training.select_best_checkpoint $$LOG

.PHONY: promote-best
promote-best:  ## Copy best checkpoint to adapters.safetensors (LOG=path optional).
	@LOG="$${LOG:-$$(ls -t $(LOG_DIR)/lora-*.log 2>/dev/null | head -1)}"; \
	if [ -z "$$LOG" ]; then echo "no logs found in $(LOG_DIR)" >&2; exit 1; fi; \
	$(PY) -m training.select_best_checkpoint $$LOG --adapter-dir $(ADAPTER_DIR) --promote

# -----------------------------------------------------------------------------
# 4. Fuse + serve
# -----------------------------------------------------------------------------

.PHONY: fuse
fuse:  ## Merge adapter back into the base model and 4-bit quantize -> $(FUSED_DIR)
	@if [ ! -f "$(ADAPTER_DIR)/adapters.safetensors" ]; then \
	  echo "ERROR: $(ADAPTER_DIR)/adapters.safetensors not found. Run 'make train' or 'make promote-best' first."; \
	  exit 1; \
	fi
	$(PY) -m mlx_lm fuse \
	    --model $(BASE_MLX_DIR) \
	    --adapter-path $(ADAPTER_DIR) \
	    --save-path $(FUSED_DIR) -q
	@echo "[fuse] -> $(FUSED_DIR)"

.PHONY: serve
serve:  ## Start mlx_lm.server on :$(SERVE_PORT) using $(FUSED_DIR)
	@echo "[serve] $(FUSED_DIR) -> http://127.0.0.1:$(SERVE_PORT)/v1/chat/completions"
	$(PY) -m mlx_lm server --model $(FUSED_DIR) --port $(SERVE_PORT)

# -----------------------------------------------------------------------------
# 5. Eval + checkpoint comparison
# -----------------------------------------------------------------------------

.PHONY: eval
eval:  ## Run training.eval against the served model -> $(EVAL_REPORT)
	$(PY) -m training.eval \
	    --test-jsonl $(DATA_DIR)/test.jsonl \
	    --llm-url http://127.0.0.1:$(SERVE_PORT)/v1/chat/completions \
	    --model $(EVAL_MODEL) \
	    --max-cases $(EVAL_MAX_CASES) --max-tokens $(EVAL_MAX_TOKENS) \
	    --slim-prompt --no-cosine \
	    --report $(EVAL_REPORT)

.PHONY: eval-quick
eval-quick:  ## Smoke eval on 60 cases (fast sanity check)
	$(MAKE) eval EVAL_MAX_CASES=60 EVAL_REPORT=$(LOG_DIR)/eval-quick.json

.PHONY: compare
compare:  ## A/B compare two checkpoints. Override with `make compare A=0001000 B=0001400`.
	$(PY) -m training.compare_adapter_checkpoints \
	    --base-model $(BASE_MLX_DIR) \
	    --adapter-dir $(ADAPTER_DIR) \
	    --jsonl $(DATA_DIR)/valid.jsonl \
	    --checkpoints $(A)_adapters.safetensors $(B)_adapters.safetensors \
	    --report $(LOG_DIR)/compare_$(A)_vs_$(B).json

# -----------------------------------------------------------------------------
# 6. Cleanup
# -----------------------------------------------------------------------------

.PHONY: clean-render
clean-render:  ## Remove rendered config + temp files
	rm -f $(CONFIG_RENDERED)

.PHONY: clean-adapters
clean-adapters:  ## DELETE the LoRA adapter directory (irreversible).
	@read -p "Delete $(ADAPTER_DIR) ? [y/N] " yn && [ "$$yn" = "y" ] && rm -rf $(ADAPTER_DIR) || echo aborted

.PHONY: clean-fused
clean-fused:  ## DELETE the fused model directory (irreversible).
	@read -p "Delete $(FUSED_DIR) ? [y/N] " yn && [ "$$yn" = "y" ] && rm -rf $(FUSED_DIR) || echo aborted

.PHONY: clean-pyc
clean-pyc:  ## Remove __pycache__ directories.
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: clean
clean: clean-render clean-pyc  ## Remove temporary build/render artefacts (safe).

# -----------------------------------------------------------------------------
# 7. Status
# -----------------------------------------------------------------------------

.PHONY: status
status:  ## Show data + model artefact summary.
	@echo "Repo root: $(REPO_ROOT)"
	@echo
	@echo "Datasets:"
	@for f in $(DATA_DIR)/train.jsonl $(DATA_DIR)/valid.jsonl $(DATA_DIR)/test.jsonl; do \
	  if [ -f $$f ]; then printf "  %-50s  %s lines\n" $$f $$(wc -l < $$f); else printf "  %-50s  MISSING\n" $$f; fi; \
	done
	@echo
	@echo "Adapters:"
	@if [ -d $(ADAPTER_DIR) ]; then ls -lh $(ADAPTER_DIR)/*.safetensors 2>/dev/null | sed 's/^/  /'; else echo "  $(ADAPTER_DIR) MISSING"; fi
	@echo
	@echo "Models:"
	@for d in $(BASE_MLX_DIR) $(FUSED_DIR); do \
	  if [ -d $$d ]; then printf "  %-60s  %s\n" $$d "$$(du -sh $$d | cut -f1)"; else printf "  %-60s  MISSING\n" $$d; fi; \
	done
