#basetts
# python eval/eval_basetts.py --eval_task wer --lang en --gen_wav_dir "eval/eval_audio/llasa_1b_2" --gpu_nums 8

python eval/eval_basetts.py --eval_task wer --lang en --gen_wav_dir "eval/eval_audio/llasa_1b_10" --gpu_nums 8

# python eval/eval_basetts.py --eval_task wer --lang en --gen_wav_dir "eval/eval_audio/llasa_v2_48" --gpu_nums 8

# python eval/eval_basetts.py --eval_task sim --lang en --gen_wav_dir "" --gpu_nums 8
# python eval/eval_utmos.py --audio_dir "" --ext wav

## hard
# python eval/eval_seed_hard.py --eval_task wer --lang en --gen_wav_dir "" --gpu_nums 8

# bash eval/evaluation.sh
# ./tests/seedtts_testset/zh