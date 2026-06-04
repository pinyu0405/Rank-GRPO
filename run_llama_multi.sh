#!/bin/bash
# Multi-GPU (4-GPU, matches the paper) launcher for Llama-3.2-3B.
# Usage: bash run_llama_multi.sh <rank|sequence> <sft_step>
# Requires a 4-GPU instance. ZeRO-3 shards the optimizer across GPUs, so no CPU
# offload is needed (avoids the DeepSpeed CUDA-compile mismatch).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-online}
ADV=${1:-rank}
STEP=${2:-1500}
mkdir -p logs
OUT="./results/grpo/meta-llama/Llama-3.2-3B-Instruct_lr1e-06_kl0.001_mu1_adv${ADV}"
RESUME=""
if ls "$OUT"/checkpoint-* >/dev/null 2>&1; then RESUME="--resume"; echo "Resuming from existing checkpoint in $OUT"; else echo "Fresh start ($OUT)"; fi
accelerate launch --config_file configs/grpo_multigpu_4.yaml train_rank_grpo.py --train_path processed_datasets/grpo/grpo_dataset --model_name meta-llama/Llama-3.2-3B-Instruct --sft_checkpoint $STEP --reward_func exp_inf --advantage_level "$ADV" --mu 1 --lr 1e-6 --kl_beta 1e-3 --adam_beta1 0.9 --adam_beta2 0.99 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --num_generations 8 --gradient_accumulation_steps 6 --num_train_epochs 2 --gradient_checkpointing --save_strategy steps --save_steps 2000 --save_total_limit 1 --save_only_model --logging_steps 10 --use_vllm --vllm_mode colocate --vllm_gpu_memory_utilization 0.3 --vllm_tensor_parallel_size 4 --max_prompt_length 2048 --max_completion_length 1024 --seed 3407 --wandb_project rank_grpo --catalog_path gt_catalog.pkl --bf16 $RESUME 2>&1 | tee "logs/llama_adv${ADV}.txt"
