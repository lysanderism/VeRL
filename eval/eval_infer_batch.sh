export http_proxy=http://star-proxy.oa.com:3128 
export https_proxy=http://star-proxy.oa.com:3128

epoch=46
name=checkpoints/llasa_tts_grpo/whisper_cer_reward_1b_ddp/
##merge
# python scripts/model_merger.py merge \
#     --backend fsdp \
#     --local_dir ${name}/global_step_${epoch}/actor \
#     --target_dir ${name}/global_step_${epoch}/huggingface

test_file=eval/data/test_en1_prompt.json

## raw base
# accelerate launch --multi_gpu eval/metric/infer_prompt_batch.py --model_path "HKUSTAudio_model/Llasa-1B" --data_file $test_file --out_audio eval/base/llasa_1b_prompt --batch_size 8
# python eval/eval_basetts.py --eval_task wer --lang en --gen_wav_dir eval/base/llasa_1b_prompt --gpu_nums 8

## small base 
# accelerate launch --multi_gpu eval/metric/infer_batch.py --model_path "${name}/global_step_${epoch}/huggingface" --out_audio eval/base/llasa_1b_${epoch}  --data_file $test_file  --batch_size 8
## multi base 
accelerate launch --multi_gpu eval/metric/infer_prompt_batch.py --model_path ${name}/global_step_${epoch}/huggingface --out_audio eval/base/llasa_1b_prompt_${epoch}  --data_file $test_file  --batch_size 8
# python eval/eval_basetts.py --eval_task wer --lang en --gen_wav_dir eval/base/llasa_1b_prompt_${epoch} --gpu_nums 8


test_file=eval/data/en_seedtts_hard.json
# raw base
# accelerate launch --multi_gpu --main_process_port 29501 eval/metric/infer_prompt_batch.py --model_path "HKUSTAudio_model/Llasa-1B" --data_file $test_file --out_audio "eval/seed/en_1B" --batch_size 8

# accelerate launch --multi_gpu eval/metric/infer_prompt_batch.py --model_path ${name}/global_step_${epoch}/huggingface --out_audio eval/seed/llasa_1b_rl_prompt --data_file $test_file  --batch_size 8
# python eval/eval_seed_hard.py --eval_task wer --lang en --gen_wav_dir "eval/seed/llasa_1b_rl_prompt" --gpu_nums 8

# bash eval/eval_infer_batch.sh
