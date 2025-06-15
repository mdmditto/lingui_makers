#!/bin/bash
set -x

# Use XFORMERS for vLLM attention backend (prevents CUDA errors)
export VLLM_ATTENTION_BACKEND=XFORMERS

# Parse command line arguments for --model
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="agentica-org/DeepScaleR-1.5B-Preview"
fi

# Ensure only one GPU is used (A6000 is usually index 0)
export CUDA_VISIBLE_DEVICES=1

# Adjust batch sizes for a single GPU (you may need to lower further if OOM)
TRAIN_BATCH_SIZE=16
VAL_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=8

# Launch training (adjusted for 1 GPU)
python3 ../../main_ppo.py \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/deepscaler/data/train.parquet \
    data.val_files=$HOME/deepscaler/data/aime.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    +actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    actor_rollout_ref.rollout.n=4 \
    +actor_rollout_ref.rollout.n_val=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='deepscaler' \
    trainer.experiment_name='l1_exact_single_a6000' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=3 "${@:1}" \
    reward_config.sigmoid_reward=False \
    reward_config.linear_reward=True \
    reward_config.multiplier_reward=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.multi_turn.enable=False \
    reward_config.alpha=0.0003
