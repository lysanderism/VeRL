set -x
export WANDB_MODE=disabled
export WHISPER_SERVERS="http://localhost:8000"
export no_proxy="localhost,127.0.0.1"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/llasa-tts-rl-wer/train.parquet \
    data.val_files=./data/llasa-tts-rl-wer/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.truncation='error' \
    actor_rollout_ref.model.path=./HKUSTAudio_model/Llasa-1B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.do_sample=true \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    custom_reward_function.path=verl/utils/reward_score/tts_cer_ddp.py \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=batch \
    trainer.project_name='llasa_tts_grpo' \
    trainer.experiment_name='whisper_cer_reward_1b_ddp' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=4 \
    trainer.test_freq=8 \
    trainer.resume_mode='auto' \
    trainer.total_epochs=2 "$@" \
    trainer.logger=['console','tensorboard'] \

# ray stop -f
# bash examples/grpo_trainer/run_llasa_tts_grpo_dp.sh
# ray start --head --dashboard-port 8880
# nohup bash examples/grpo_trainer/run_llasa_tts_grpo_dp.sh >train1b.log 2>&1 &
# validation generation end

# tensorboard --logdir=tensorboard_log --port 6019