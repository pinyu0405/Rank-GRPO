export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ADV=${1:-rank}
STEP=${2:-1500}
mkdir -p logs
OUT="./results/grpo/meta-llama/Llama-3.2-3B-Instruct_lr1e-06_kl0.001_mu1_adv${ADV}"
RESUME=""
if ls "$OUT"/checkpoint-* >/dev/null 2>&1; then RESUME="--resume"; echo "Found existing checkpoint in $OUT -> resuming"; else echo "No checkpoint in $OUT -> fresh start"; fi
accelerate launch --config_file configs/llama_a6000.yaml train_rank_grpo.py --train_path processed_datasets/grpo/grpo_dataset --model_name meta-llama/Llama-3.2-3B-Instruct --sft_checkpoint $STEP --reward_func exp_inf --advantage_level "$ADV" --mu 1 --lr 1e-6 --kl_beta 1e-3 --adam_beta1 0.9 --adam_beta2 0.99 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --num_generations 4 --gradient_accumulation_steps 8 --num_train_epochs 2 --gradient_checkpointing --save_strategy steps --save_steps 200 --logging_steps 10 --use_vllm --vllm_mode colocate --vllm_gpu_memory_utilization 0. --vllm_tensor_parallel_size 1 --max_prompt_length 2048 --max_completion_length 512 --seed 3407 --wandb_project rank_grpo --catalog_path gt_catalog.pkl --bf16 $RESUME 2>&1 | tee "logs/llama_adv${ADV}.txt"
