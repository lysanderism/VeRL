import json
import re
import os
import random
import string
from pathlib import Path
from torchaudio.transforms import Resample
import torch
import torch.nn.functional as F
import torchaudio
import librosa
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Wav2Vec2Processor, HubertForCTC
import whisper
# from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL


def number_to_words(n):
    units = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    if n == 0:
        return units[0]

    words = []

    # 处理百万
    if n >= 1000000:
        millions = n // 1000000
        words.append(number_to_words(millions) + " million")
        n %= 1000000

    # 处理千位（递归）
    if n >= 1000:
        thousands = n // 1000
        words.append(number_to_words(thousands) + " thousand")
        n %= 1000
        if 0 < n < 100:
            words.append("and")

    # 处理百位
    if n >= 100:
        hundreds = n // 100
        words.append(units[hundreds] + " hundred")
        n %= 100
        if n > 0:
            words.append("and")

    # 处理十位和个位
    if n >= 20:
        words.append(tens[n // 10])
        n %= 10
    elif 10 <= n < 20:
        words.append(teens[n - 10])
        n = 0

    if n > 0:
        words.append(units[n])

    return " ".join(words).replace(" and zero", "").replace("  ", " ")

def replace_mixed_numbers(text):
    parts = re.findall(r'\d+|\D+', text)
    converted = []
    for part in parts:
        if part.isdigit():
            converted.append(number_to_words(int(part)))
        else:

            converted.append(part)

    return re.sub(r'\s+', ' ', ' '.join(converted)).strip()

# seedtts testset metainfo: utt, prompt_text, prompt_wav, gt_text, gt_wav
def get_seedtts_testset_metainfo(metalst):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        if len(line.strip().split("|")) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = line.strip().split("|")
        elif len(line.strip().split("|")) == 4:
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")
            gt_wav = os.path.join(os.path.dirname(metalst), "wavs", utt + ".wav")
            if os.path.exists(gt_wav):
                continue
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)
        # gt_text = replace_mixed_zhnumbers(gt_text)
        metainfo.append((utt, prompt_text, prompt_wav, gt_text, gt_wav))
    return metainfo

# librispeech test-clean metainfo: gen_utt, ref_txt, ref_wav, gen_txt, gen_wav
def get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")

        # ref_txt = ref_txt[0] + ref_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + ".flac")

        # gen_txt = gen_txt[0] + gen_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
        gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")

        metainfo.append((gen_utt, ref_txt, ref_wav, " " + gen_txt, gen_wav))

    return metainfo

# get prompts from metainfo containing: utt, prompt_text, prompt_wav, gt_text, gt_wav
def get_seed_tts_test(metalst, gen_wav_dir, gpus):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        if len(line.strip().split("|")) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = line.strip().split("|")
        elif len(line.strip().split("|")) == 4:
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")

        if not os.path.exists(os.path.join(gen_wav_dir, utt + ".wav")):
            continue
        gen_wav = os.path.join(gen_wav_dir, utt + ".wav")
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)

        test_set_.append((gen_wav, gt_text))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set

def get_basetts_test(json_data, gen_wav_dir, gpus):
    test_set_ = []
    with open(json_data, 'r') as f:
        data = json.load(f)
    for item in tqdm(data):
        text = item['text']
        wav_id = item['id']
        if not os.path.exists(os.path.join(gen_wav_dir, wav_id)):
            continue
        gen_wav = os.path.join(gen_wav_dir, wav_id)
        test_set_.append((gen_wav, text))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))
    return test_set


def load_asr_model(lang, ckpt_dir=""):
    if lang == "zh":
        from funasr import AutoModel
        model = AutoModel(
            model=os.path.join(ckpt_dir, "paraformer-zh"),
            # vad_model = os.path.join(ckpt_dir, "fsmn-vad"),
            # punc_model = os.path.join(ckpt_dir, "ct-punc"),
            # spk_model = os.path.join(ckpt_dir, "cam++"),
            disable_update=True,
        )  # following seed-tts setting
    elif lang == "en":
        from faster_whisper import WhisperModel
        model_size = "large-v3" if ckpt_dir == "" else ckpt_dir
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    return model

# WER Evaluation
def run_asr_wer(args):
    rank, lang, test_set, ckpt_dir = args

    if lang == "zh":
        import zhconv

        torch.cuda.set_device(rank)
    elif lang == "en":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    else:
        raise NotImplementedError(
            "lang support only 'zh' (funasr paraformer-zh), 'en' (faster-whisper-large-v3), for now."
        )

    asr_model = load_asr_model(lang, ckpt_dir=ckpt_dir)


    punctuation_all = string.punctuation
    wer_results = []

    from jiwer import compute_measures

    for gen_wav, prompt_wav, truth in tqdm(test_set):
        if lang == "zh":
            res = asr_model.generate(input=gen_wav, batch_size_s=300, disable_pbar=True)
            hypo = res[0]["text"]
            hypo = zhconv.convert(hypo, "zh-cn")
        elif lang == "en":
            segments, _ = asr_model.transcribe(gen_wav, beam_size=5, language="en")
            hypo = ""
            for segment in segments:
                hypo = hypo + " " + segment.text

        raw_truth = truth
        raw_hypo = hypo

        for x in punctuation_all:
            truth = truth.replace(x, "")
            hypo = hypo.replace(x, "")

        truth = truth.replace("  ", " ")
        hypo = hypo.replace("  ", " ")

        if lang == "zh":
            truth = " ".join([x for x in truth])
            hypo = " ".join([x for x in hypo])
        elif lang == "en":
            truth = truth.lower()
            hypo = hypo.lower()

        measures = compute_measures(truth, hypo)
        wer = measures["wer"]

        # ref_list = truth.split(" ")
        # subs = measures["substitutions"] / len(ref_list)
        # dele = measures["deletions"] / len(ref_list)
        # inse = measures["insertions"] / len(ref_list)

        wer_results.append(
            {
                "wav": Path(gen_wav).stem,
                "truth": raw_truth,
                "hypo": raw_hypo,
                "wer": wer,
            }
        )

    return wer_results

# WER Evaluation
def run_asr_wer_whisper(args, is_ellav=True):
    rank, lang, test_set, ckpt_dir = args

    if lang == "zh":
        import zhconv

        torch.cuda.set_device(rank)
    elif lang == "en":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    else:
        raise NotImplementedError(
            "lang support only 'zh' (funasr paraformer-zh), 'en' (faster-whisper-large-v3), for now."
        )

    processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path = "openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path ="openai/whisper-large-v3").to("cuda")

    punctuation_all =  string.punctuation
    wer_results = []

    from jiwer import cer,wer

    # test_set = test_set[0]
    # rank, test_set = test_set[0], test_set[1]
    for gen_wav, truth in tqdm(test_set):
        if lang == "zh":
            res = asr_model.generate(input=gen_wav, batch_size_s=300, disable_pbar=True)
            hypo = res[0]["text"]
            hypo = zhconv.convert(hypo, "zh-cn")
        elif lang == "en":
            wav, sr = librosa.load(gen_wav, sr=16000)
            # hypo = asr_model.transcribe(
            #     wav,
            #     language="en",
            #     fp16=True
            # )["text"].strip()

            input_features = processor(
                wav, sampling_rate=16000, return_tensors="pt"
            ).input_features
            input_features = input_features.to("cuda")
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language="english", task="transcribe"
            )
            predicted_ids = model.generate(
                input_features, forced_decoder_ids=forced_decoder_ids
            )
            hypo = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

        raw_truth = truth
        raw_hypo = hypo

        truth = truth.replace('-', ' ')
        hypo = hypo.replace('-', ' ')
        truth = truth.replace('“', ' ').replace('”', ' ')
        hypo = hypo.replace('“', ' ').replace('”', ' ')
        import unicodedata, re

        def strip_punctuation(text: str) -> str:
            out = []
            for ch in text:
                cat = unicodedata.category(ch)
                # P* = punctuation, S* = symbol
                out.append(' ' if cat.startswith(('P', 'S')) else ch)
            return re.sub(r'\s+', ' ', ''.join(out)).strip()

        truth = strip_punctuation(truth)
        hypo  = strip_punctuation(hypo)
        # print(truth, hypo)
        
        if lang == "zh":
            truth = " ".join([x for x in truth])
            hypo = " ".join([x for x in hypo])
        elif lang == "en":
            truth = truth.lower()
            hypo = hypo.lower()
        if is_ellav:
            hypo = replace_mixed_numbers(hypo)
        
        wer_pred = wer(truth, hypo,)
        cer_pred = cer(truth, hypo,)

        wer_results.append(
            {
                "wav": Path(gen_wav).stem,
                "truth": raw_truth,
                "hypo": raw_hypo,
                "wer": wer_pred,
                "cer": cer_pred,
            }
        )

    return wer_results

# SIM Evaluation
def run_sim(args):
    rank, test_set, ckpt_dir = args
    device = f"cuda:{rank}"

    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(ckpt_dir, weights_only=True, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"], strict=False)

    use_gpu = True if torch.cuda.is_available() else False
    if use_gpu:
        model = model.cuda(device)
    model.eval()

    sims = []
    for wav1, wav2, truth in tqdm(test_set):
        wav1, sr1 = torchaudio.load(wav1)
        wav2, sr2 = torchaudio.load(wav2)

        resample1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)
        resample2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)
        wav1 = resample1(wav1)
        wav2 = resample2(wav2)

        if use_gpu:
            wav1 = wav1.cuda(device)
            wav2 = wav2.cuda(device)
        with torch.no_grad():
            emb1 = model(wav1)
            emb2 = model(wav2)

        sim = F.cosine_similarity(emb1, emb2)[0].item()
        # print(f"VSim score between two audios: {sim:.4f} (-1.0, 1.0).")
        sims.append(sim)

    return sims
