# Evaluate with Seed-TTS testset

import argparse
import json
import os
import sys

sys.path.append(os.getcwd())

import multiprocessing as mp
from importlib.resources import files

import numpy as np
from eval.utils_eval import (
    get_basetts_test,
    run_asr_wer_whisper,
    run_sim,
)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval_task", type=str, default="wer", choices=["sim", "wer"])
    parser.add_argument("-l", "--lang", type=str, default="en", choices=["zh", "en"])
    parser.add_argument("-g", "--gen_wav_dir", type=str, required=True)
    parser.add_argument("-n", "--gpu_nums", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--local", action="store_true", help="Use local custom checkpoint directory")
    return parser.parse_args()

def main():
    args = get_args()
    eval_task = args.eval_task
    lang = args.lang
    gen_wav_dir = args.gen_wav_dir
    ##truth 
    metalst = "eval/data/test_en1.json"

    gpus = list(range(args.gpu_nums))
    test_set = get_basetts_test(metalst, gen_wav_dir, gpus)

    local = args.local
    if local:
        if lang == "zh":
            asr_ckpt_dir = "./ckpts/paraformer-zh"  # paraformer-zh dir under funasr
        elif lang == "en":
            asr_ckpt_dir = "model/whisper-large-v3"
            # asr_ckpt_dir = "./ckpts/hubert-large-ls960-ft"
    else:
        asr_ckpt_dir = ""
    wavlm_ckpt_dir = "./ckpts/wavlm_large_finetune.pth"

    # --------------------------- WER ---------------------------
    wer_result_path = f"{gen_wav_dir}/{lang}_wer_results.jsonl"

    if eval_task == "wer":
        wer_results = []
        wers = []
        cers = []
        ctx = mp.get_context("spawn") 
        with ctx.Pool(processes=len(gpus)) as pool:
            args = [(rank, lang, sub_test_set, asr_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_asr_wer_whisper, args)
            for r in results:
                wer_results.extend(r)

        with open(wer_result_path, "w") as f:
            for line in wer_results:
                wers.append(line["wer"])
                cers.append(line["cer"])
                json_line = json.dumps(line, ensure_ascii=False)
                f.write(json_line + "\n")

        wer = round(np.mean(wers) * 100, 3)
        cer = round(np.mean(cers) * 100, 3)

        print(f"\nTotal {len(wers)} samples")
        print(f"WER      : {wer}%")
        print(f"CER      : {cer}%")
        print(f"Results have been saved to {wer_result_path}")
        with open(wer_result_path, "a") as f:
            f.write(f"Total {len(wers)},WER      :{wer}%, CER      :{cer}%")

    # --------------------------- SIM ---------------------------

    if eval_task == "sim":
        sims = []
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, sub_test_set, wavlm_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_sim, args)
            for r in results:
                sims.extend(r)

        sim = round(sum(sims) / len(sims), 3)
        print(f"\nTotal {len(sims)} samples")
        print(f"SIM      : {sim}")
        print(f"Results have been saved to {wer_result_path}")
        with open(wer_result_path, "a") as f:
            f.write(f"Total {len(sims)},SIM      :{sim}%")

if __name__ == "__main__":
    main()
