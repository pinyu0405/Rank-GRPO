# Changes vs. upstream (yaochenzhu/Rank-GRPO)

Fork: `pinyu0405/Rank-GRPO`. This file summarizes what we changed and our workflow.

## Core changes (research + bug fixes)

| File | What | Why |
|---|---|---|
| `libs/trl/rank_grpo_trainer.py` | **Sequence-level advantage mode** (collapse the per-position reward vector into one scalar `Σ_t rank_reward[t]`, normalize within the generation group, broadcast the same advantage to every rank position). Plus import guards: local `is_rich_available`, `try/except` around `vllm_client` and `callbacks`, create `model.warnings_issued` if missing. | Implements the rank-vs-sequence ablation of toy task §5.2.1; guards let the trainer import on prebuilt/"dirty" GPU images. |
| `train_rank_grpo.py` | New CLI args `--advantage_level {rank,sequence}` and `--log_completions`; `output_dir` gets an `_adv{level}` suffix. | A/B switch + inspect generations. |
| `libs/metrics.py` | Added `evaluate_direct_match_truncate`. | Imported by `eval_grpo_val.py` but never defined upstream. |
| `evaluate/eval_grpo_val.py`, `evaluate/eval_grpo_test.py` | Use `parse_grpo_log_history` / `plot_rewards` with the 3-tuple `(steps, rewards, reward_stds)` signature. | Upstream imported SFT-only names that don't exist in `analyze_grpo`. |
| `libs/logs.py` | Removed the `from libs.logs import ...` self-import. | Circular import. |

**The reward functions (`libs/reward_funcs.py`, toy task §5.2.2) are unchanged** — only the advantage assignment differs between the two arms, so the A/B is a clean ablation.

## New config / launcher files

- `configs/qwen25_0.5b_grpo_3080.yaml` — single-GPU GRPO config for Qwen2.5-0.5B (10GB-class).
- `configs/llama_a6000.yaml` — single-GPU GRPO config for Llama-3.2-3B (toggle CPU offload by VRAM).
- `configs/grpo_multigpu_4.yaml` — 4-GPU ZeRO-3 config (paper-style; optimizer sharded, no offload).
- `run_qwen.sh`, `run_llama.sh` — launchers; the two A/B arms differ only by the `rank|sequence` argument.

## Results & project page

- `eval_results/{qwen,llama}/{rank,seq}/analysis_state.json` — per-checkpoint NDCG@K / Recall@K and `eval_rec_num` (catalog hit rate), plus reward/entropy figures.
- `docs/` — GitHub Pages site (Academic-Project-Page template, `.nojekyll`): comparison tables, Qwen NDCG/Recall-vs-step curves (Chart.js), entropy, and the §5.2.1 reward/advantage formulas (MathJax). Enable via Settings → Pages → `main` `/docs`.
- **Headline:** rank-level beats sequence-level on every K, both models, gap widening with K. Llama-3B @2400: Recall@20 +17.7%, NDCG@20 +14.0%. Qwen-0.5B @3000 (aligned): Recall@20 +9.4%, NDCG@20 +9.5%.

## Repo hygiene

- `requirements.txt` — pinned working stack (see Environment setup below).
- `.gitignore` — ignores `.DS_Store`, `__pycache__/`, `*.pdf`, `wandb/`, `results/`, `.claude/`; large checkpoints go to the HF Hub, not git.

## Workflow

1. **Code** lives in the `pinyu0405` fork; edit → push → `git pull` on the machine.
2. **Rent GPU** on vast.ai (prefer a clean CUDA/PyTorch image; avoid Ascend/NPU templates).
3. **Env**: clone fork → pin `transformers==4.55.4`, `trl==0.21.0`, `vllm==0.10.0`, `huggingface_hub<1.0` → `pip install --force-reinstall --no-deps trl==0.21.0` → `bash INSTALL.SH` (installs our trainer into the `trl` package).
4. **Assets**: place the SFT checkpoint at `results/<model>/checkpoint-<step>` and the processed dataset under `processed_datasets/`. `gt_catalog.pkl` is in the repo.
5. **Train A/B**: `run_qwen.sh` / `run_llama.sh <rank|sequence> <sft_step>`, switching only `--advantage_level rank|sequence`. Use `WANDB_MODE=offline` if the box can't reach W&B (`wandb sync` later — sync each offline run under a fresh `--id`, since runs share a stable id).
6. **Checkpoints**: standard HF checkpointing (`--save_strategy steps --save_steps N`); to save disk use a larger `--save_steps` or a bigger instance disk.
7. **Eval**: run from `evaluate/` with `PYTHONPATH=<repo>/libs` → `eval_grpo_val.py` computes NDCG@{5,10,15,20} and Recall@{5,10,15,20} per checkpoint.
8. **Preserve**: upload checkpoints to the HF Hub via `huggingface-cli upload` (not git — GitHub rejects multi-GB files).

## Environment gotchas (already folded into the workflow)

- Dirty vast.ai images ship a forked `trl` (Ascend `vllm_ascend`, unguarded `mergekit`/`judges` imports) → handled by the trainer import guards.
- System CUDA toolkit vs. torch CUDA mismatch breaks DeepSpeed CPU-offload op build → keep offload OFF, or pick a matching-CUDA image.
- Disk fills from large checkpoints → use a bigger instance disk and/or a larger `--save_steps`.
- Terminals mangle pasted multi-line commands → use the in-repo single-line launcher scripts.

## Environment setup (tested working combo)

Python **3.10**. Install in this order (order matters — `trl`/`transformers` must end up at the versions below):

```bash
conda create -n rank-grpo python=3.10 -y && conda activate rank-grpo
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install vllm==0.10.0
pip install "transformers==4.55.4" "huggingface_hub<1.0" "tokenizers==0.21.4"
pip install "trl[vllm]==0.21.0" --no-deps      # then re-pin transformers if it moved
pip install accelerate deepspeed bitsandbytes datasets editdistance wandb "numpy<2" matplotlib seaborn
bash INSTALL.SH                                  # copy our patched trainer into the trl package
```

Pinned versions (the combination we verified on A100-80GB / cu126; also in `requirements.txt`):

| package | version | note |
|---|---|---|
| python | 3.10 | |
| torch / torchvision / torchaudio | 2.7.1 / 0.22.1 / 2.7.1 | **cu126** index; runs on a CUDA-12.x driver via minor-version compat |
| vllm | 0.10.0 | needs torch 2.7.x |
| transformers | 4.55.4 | trl 0.21 needs ≥4.55; vllm 0.10 is fine with it |
| huggingface_hub | <1.0 (e.g. 0.36.2) | **must stay <1.0** — hub 1.x breaks transformers/tokenizers; never `pip install -U huggingface_hub` |
| tokenizers | 0.21.4 | pairs with transformers 4.55 |
| trl | 0.21.0 | our `RankGRPOTrainer` targets this exact version |
| accelerate | ≥1.0 | |
| deepspeed | latest | ZeRO-3; needs system CUDA matching torch for CPU-offload op build |
| bitsandbytes | latest | `paged_adamw_8bit` |
| datasets, editdistance, wandb, numpy(<2), matplotlib, seaborn | latest | data / metrics / logging / plots |

Notes: pick a **clean CUDA/PyTorch** vast.ai image (not Ascend/NPU). On a single 48GB card 3B needs CPU offload (system CUDA must match torch's, else the DeepSpeed op build fails); on A100-80GB or multi-GPU ZeRO-3, offload off.
