import os, re, time, warnings, json
from typing import List

import numpy as np
import requests
from jiwer import cer, wer
from requests.adapters import HTTPAdapter
import torch
import unicodedata
# -----------------------------------------------------------------------------
# 1) config
# -----------------------------------------------------------------------------
BASE_PORT    = 8000
SERVER       = f"http://localhost:{BASE_PORT}"
ENDPOINT     = SERVER + "/batch_score"
HEALTH       = SERVER + "/healthz"

MAX_BATCH    = 4
REQ_TIMEOUT  = 200.0
HEALTH_INTVL = 30.0

NUM_GPUS = 8

# -----------------------------------------------------------------------------
# 2) HTTP session
# -----------------------------------------------------------------------------
_last_health = 0.0

def create_session():
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=10, pool_maxsize=100,
        max_retries=3, pool_block=True
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

SESSION = create_session()

def check_health():
    global _last_health
    now = time.time()
    if now - _last_health < HEALTH_INTVL:
        return
    try:
        SESSION.get(HEALTH, timeout=2).raise_for_status()
        _last_health = now
    except Exception as e:
        raise RuntimeError(f"failed: {e}")

# -----------------------------------------------------------------------------
# 3) Token 
# -----------------------------------------------------------------------------
_PARSE = re.compile(r"<\|s_(\d+)\|>")
def parse_ids(token_str: str) -> List[int]:
    return [int(m) for m in _PARSE.findall(token_str)]

# -----------------------------------------------------------------------------
# 4) post HTTP )
# -----------------------------------------------------------------------------
def remote_batch(tokens: List[List[int]], texts: List[str]):
    check_health()
    payload = [{"tokens": t, "text": x} for t, x in zip(tokens, texts)]
    resp = SESSION.post(ENDPOINT, json=payload, timeout=REQ_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list) or len(data) != len(tokens):
        raise ValueError(f"mismatch: {len(data)} != {len(tokens)}")
    return data

# -----------------------------------------------------------------------------
# 5) compute_score： MAX_BATCH×GPU 
# -----------------------------------------------------------------------------
def compute_score(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos:   List[dict],
    *,
    beta_c=3.0, 
    tau_n=3.0, 
    lambda_c=0.6, 
    lambda_n=0.4
) -> np.ndarray:
    N = len(solution_strs)
    chunk_size = MAX_BATCH * NUM_GPUS
    responses = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        toks = [parse_ids(solution_strs[i]) for i in range(start, end)]
        txts = [ground_truths[i]      for i in range(start, end)]
        try:
            chunk = remote_batch(toks, txts)
        except Exception as e:
            warnings.warn(f"[{start}:{end}] failed: {e}")
            chunk = [{"nll": None, "transcript": ""}] * (end - start)
        responses.extend(chunk)

    nlls = np.zeros(N, dtype=float)
    hyps = []
    for i, res in enumerate(responses):
        if res.get("nll") is None:
            nlls[i] = 1e9
            hyps.append("")
        else:
            nlls[i] = float(res.get("nll"))
            hyps.append(res["transcript"])
        # hyps.append(res["transcript"])

    cers = np.array([cer((gt), (hyp) or (gt)) for gt, hyp in zip(ground_truths, hyps)])
    cer_u = 1.0 - np.tanh(beta_c * cers)
    # cal NLL 
    nll_u = np.exp(-nlls / tau_n)
    denom = lambda_c / cer_u + lambda_n / nll_u
    rewards = (lambda_c + lambda_n) / np.where(denom > 0, denom, np.inf)

    print(f"Processed {N} samples | WER {cers.mean():.3f} | Reward {rewards.mean():.4f}")
        #   f"NLL {nlls.mean():.2f} | Reward {rewards.mean():.4f}")
    return np.clip(rewards, 0.0, 1.0)



if __name__ == "__main__":
    data = json.load(open("sample.json", "r", encoding="utf-8"))
    sols, gts = data["solutions"], data["gts"]
    rewards = compute_score(sols, gts)
    print(json.dumps({"reward": rewards.tolist()}, indent=2, ensure_ascii=False))
