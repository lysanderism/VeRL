CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nproc_per_node=8  \
  tts/whisper_ddp.py --port 8000


# bash lanuch.sh
# ray start --head --dashboard-port 8888
# ray stop -f

# python scripts/model_merger.py merge \
#     --backend fsdp \
#     --local_dir checkpoints/llasa_tts_grpo/whisper_cer_reward_1b_ddp/global_step_1/actor \
#     --target_dir checkpoints/llasa_tts_grpo/whisper_cer_reward_1b_ddp/global_step_1/huggingface
# find path -maxdepth 1 -type f -exec cp {} ./ \;