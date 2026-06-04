# Changes vs. upstream (yaochenzhu/Rank-GRPO)

Fork: `pinyu0405/Rank-GRPO`. This file summarizes what we changed and our workflow.

## Core changes (research + bug fixes)

| File | What | Why |
|---|---|---|
| `libs/trl/rank_grpo_trainer.py` | **Sequence-level advantage mode** (collapse the per-position reward vector into one scalar `Σ_t rank_reward[t]`, normalize within the generation group, broadcast the same advantage to every rank position). Plus import guards: local `is_rich_available`, `try/except` around `vllm_client` and `callbacks`, create `model.warnings_issued` if missing. | Implements the rank-vs-sequence ablation of toy task §5.2.1; guards let the trainer import on prebuilt/"dirty" GPU images. |
| `train_rank_grpo.py` | New CLI args `--advantage_level {rank,sequence}`, `--save_total_limit` (0 = keep all), `--save_only_model`; `output_dir` gets an `_adv{level}` suffix. | A/B switch + checkpoint/disk management. |
| `libs/metrics.py` | Added `evaluate_direct_match_truncate`. | Imported by `eval_grpo_val.py` but never defined upstream. |
| `evaluate/eval_grpo_val.py`, `evaluate/eval_grpo_test.py` | Use `parse_grpo_log_history` / `plot_rewards` with the 3-tuple `(steps, rewards, reward_stds)` signature. | Upstream imported SFT-only names that don't exist in `analyze_grpo`. |
| `libs/logs.py` | Removed the `from libs.logs import ...` self-import. | Circular import. |

**The reward functions (`libs/reward_funcs.py`, toy task §5.2.2) are unchanged** — only the advantage assignment differs between the two arms, so the A/B is a clean ablation.

## New config / launcher files

- `configs/qwen25_0.5b_grpo_3080.yaml` — single-GPU config for Qwen2.5-0.5B (10GB-class).
- `configs/llama_a6000.yaml` — single-GPU config for Llama-3.2-3B on A6000 (CPU offload OFF, CUDA-mismatch safe).
- `run_llama.sh` — one-line launcher: `bash run_llama.sh <rank|sequence> <sft_step>`, WANDB offline by default.

## Workflow

1. **Code** lives in the `pinyu0405` fork; edit → push → `git pull` on the machine.
2. **Rent GPU** on vast.ai (prefer a clean CUDA/PyTorch image; avoid Ascend/NPU templates).
3. **Env**: clone fork → pin `transformers==4.55.4`, `trl==0.21.0`, `vllm==0.10.0`, `huggingface_hub<1.0` → `pip install --force-reinstall --no-deps trl==0.21.0` → `bash INSTALL.SH` (installs our trainer into the `trl` package).
4. **Assets**: place the SFT checkpoint at `results/<model>/checkpoint-<step>` and the processed dataset under `processed_datasets/`. `gt_catalog.pkl` is in the repo.
5. **Train A/B**: `run_ab.sh` / `run_llama.sh`, switching only `--advantage_level rank|sequence`. Use `WANDB_MODE=offline` (rented boxes often can't reach W&B; `wandb sync` later).
6. **Checkpoints**: `--save_only_model` (small weight-only) + `--save_total_limit` (`0` = keep all for metric-vs-step curves, `1` = keep latest for resume).
7. **Eval**: run from `evaluate/` with `PYTHONPATH=<repo>/libs` → `eval_grpo_val.py` computes NDCG@{5,10,15,20} and Recall@{5,10,15,20} per checkpoint.
8. **Preserve**: upload checkpoints to the HF Hub via `huggingface-cli upload` (not git — GitHub rejects multi-GB files).

## Environment gotchas (already folded into the workflow)

- Dirty vast.ai images ship a forked `trl` (Ascend `vllm_ascend`, unguarded `mergekit`/`judges` imports) → handled by the trainer import guards.
- System CUDA toolkit vs. torch CUDA mismatch breaks DeepSpeed CPU-offload op build → keep offload OFF, or pick a matching-CUDA image.
- Disk fills from large checkpoints (save writes new before deleting old → 2× peak) → `--save_only_model`.
- Terminals mangle pasted multi-line commands → use the in-repo single-line launcher scripts.
