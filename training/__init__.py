"""Training pipeline for the Indian-legal FIR event extractor.

See ../README.md for the workflow and ./NEXT_STEPS.md for design rationale.

Stages:
- Stage A: Definition QA              (build_definition_qa.py)            — templated from LAW_RAW_DATA/DEFINITIONS/*.json
- Stage B: Taxonomy classification    (build_taxonomy_classification.py)  — templated from LAW_RAW_DATA/CRIME_TAXONOMY.json
- Stage C: End-to-end FIR -> events   (build_synthetic_firs.py)           — distilled from a teacher LLM
- Stage D: Hard negatives             (build_hard_negatives.py)           — templated from non_crime + corner cases
- Stage E: Definition-grounded examples (build_definition_grounded.py)    — anchors model on statute text
- Merge & split                       (merge_and_split.py)                — canonicalize, dedupe, stratify, balance, split
- Verify                              (verify_dataset.py)                 — drift-label + 231/231 coverage assertions
- Train                               (early_stop_lora.py)                — wraps mlx_lm.lora with live early-stop
- Best-checkpoint promotion           (select_best_checkpoint.py)         — copies lowest-val-loss step over adapters.safetensors
- A/B compare                         (compare_adapter_checkpoints.py)    — side-by-side inference
- Eval                                (eval.py)                           — F1, JSON validity, MAE, per-case trace
"""
