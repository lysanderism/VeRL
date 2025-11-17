#!/usr/bin/env python3
from __future__ import annotations
import os, argparse, sys, threading, time
from queue import Queue, Empty
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import uvicorn
import whisper
import whisper.audio as _wa
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch.nn.utils.rnn import pad_sequence
from xcodec2.modeling_xcodec2 import XCodec2Model

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--model", type=str, default="large-v3")
parser.add_argument("--codec", type=str, default="HKUST-Audio/xcodec2")
parser.add_argument("--backend", type=str, default="nccl")
args, _ = parser.parse_known_args()

dist.init_process_group(backend=args.backend)
local_rank = int(os.environ["LOCAL_RANK"])
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

whisper_base = whisper.load_model(
    args.model, download_root="whisper-v3", device=device
).eval()

for name, buf in whisper_base.named_buffers():
    if buf.layout != torch.strided:
        parts = name.split('.')
        parent = whisper_base
        for p in parts[:-1]: parent = getattr(parent, p)
        parent._buffers[parts[-1]] = buf.to_dense().to(device)

model = torch.nn.parallel.DistributedDataParallel(
    whisper_base,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False
)
model.eval()

codec = XCodec2Model.from_pretrained(args.codec,).eval().to(device)
codec = torch.compile(codec, mode="reduce-overhead", fullgraph=True)

TOKENIZER = whisper.tokenizer.get_tokenizer(multilingual=True, task="transcribe")
PAD_ID = TOKENIZER.eot

def _get_mel_bins(m: whisper.Whisper) -> int:
    if hasattr(m.encoder, "conv1"): return m.encoder.conv1.in_channels
    return 128 if m.dims.n_mels == 128 else 80
REQ_BINS = _get_mel_bins(whisper_base)

class ScoreRequest(BaseModel):
    tokens: List[int]
    text: str

class ScoreResponse(BaseModel):
    nll: float
    transcript: str

class Job:
    def __init__(self, data: List[ScoreRequest]):
        self.data = data
        self.result: Optional[List[ScoreResponse]] = None
        self.error: Optional[Exception] = None
        self.event = threading.Event()


@torch.inference_mode()
def process_batch(reqs: List[ScoreRequest]) -> List[ScoreResponse]:
    B = len(reqs)
    # scatter requests into per-rank slice
    chunk = (B + world_size - 1) // world_size
    start = local_rank * chunk
    end = min(start + chunk, B)
    part_reqs = reqs[start:end]

    # decode part tokens -> wavs on each GPU
    if part_reqs:
        toks = [torch.tensor(r.tokens, device=device) for r in part_reqs]
        pad = pad_sequence(toks, batch_first=True, padding_value=0)
        out = codec.decode_code(pad[:, None, :])
        audio_gpu = out[0] if isinstance(out, tuple) else out
        if audio_gpu.ndim == 3:
            audio_gpu = audio_gpu[:, 0]
        part_wavs = [w.cpu() for w in audio_gpu]

        # transcribe
        part_transcripts = []
        for w in part_wavs:
            result = whisper_base.transcribe(audio=w.numpy(), fp16=True)["text"].strip()
            part_transcripts.append(result)
    else:
        part_wavs = []
        part_transcripts = []

    # gather all wavs and transcripts back to each rank
    gathered_wavs = [None] * world_size
    dist.all_gather_object(gathered_wavs, part_wavs)
    wavs = [w for sub in gathered_wavs for w in sub]

    gathered_trans = [None] * world_size
    dist.all_gather_object(gathered_trans, part_transcripts)
    transcripts = [t for sub in gathered_trans for t in sub]

    # build full mel & tgt
    mels = [
        _wa.log_mel_spectrogram(_wa.pad_or_trim(w.numpy()), n_mels=REQ_BINS)
        for w in wavs
    ]
    mel = pad_sequence([torch.as_tensor(m) for m in mels], batch_first=True).to(device)
    tgt_ids = [[TOKENIZER.sot] + TOKENIZER.encode(r.text) + [TOKENIZER.eot] for r in reqs]
    tgt = pad_sequence(
        [torch.tensor(ids, device=device) for ids in tgt_ids],
        batch_first=True,
        padding_value=PAD_ID
    )
    # scatter mel/tgt for parallel nll compute
    part_mel = mel[start:end]
    part_tgt = tgt[start:end]

    if part_mel.shape[0] > 0:
        logits = model(part_mel, part_tgt[:, :-1])
        nll_all = F.cross_entropy(
        logits.flatten(0,1),
        part_tgt[:, 1:].flatten(),
        reduction="none"
        )
        lengths_on_this_rank = [len(ids) - 1 for ids in tgt_ids[start:end]]
        part_nll = []
        local_pos = 0
        for L in lengths_on_this_rank:
            if L > 0:
                nll_slice = nll_all[local_pos : local_pos + L]
                part_nll.append(float(nll_slice.mean()))
                local_pos += L
            else:
                part_nll.append(0.0)
    else:
        part_nll = []

    # gather nll
    gathered_nll = [None] * world_size
    dist.all_gather_object(gathered_nll, part_nll)
    nlls = [x for sub in gathered_nll for x in sub]

    if local_rank == 0:
        return [ScoreResponse(nll=n, transcript=t) for n, t in zip(nlls, transcripts)]
    return []


@torch.inference_mode()
def process_batch_single(reqs: List[ScoreRequest]) -> List[ScoreResponse]:
    scatter_buf: list[Optional[ScoreRequest]] = [None] * world_size
    if local_rank == 0:
        scatter_buf[: len(reqs)] = reqs

    dist.scatter_object_list(
        scatter_buf,           
        scatter_buf if local_rank == 0 else [],
        src=0,
    )

    my_req = scatter_buf[local_rank]          

    if my_req is None:                       
        my_result: Optional[ScoreResponse] = None
    else:
        tok = torch.tensor(my_req.tokens, device=device)[None, None, :]   # (1,1,T)
        wav = codec.decode_code(tok)[0, 0]
        mel = _wa.log_mel_spectrogram(
            _wa.pad_or_trim(wav.cpu().numpy()), n_mels=REQ_BINS
        )
        mel = torch.as_tensor(mel, device=device)[None]              # (1, C, L)
        # 4. token
        tgt = torch.tensor(
            [TOKENIZER.sot] + TOKENIZER.encode(r.text) + [TOKENIZER.eot],
            device=device
        )[None]                                                     # (1, L_t)
        enc    = whisper_base.encoder(mel)
        logits = whisper_base.decoder(tgt[:, :-1], enc)             # (1, L_t-1, V)
        nll_val   = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt[:, 1:].reshape(-1),
            ignore_index=PAD_ID,
            reduction="none"
        ).view(tgt[:, 1:].shape).mean()
        transcript  = whisper_base.transcribe(
            audio=wav.cpu().numpy(),
            fp16=torch.cuda.is_available()
        )["text"].strip()
        my_result   = ScoreResponse(nll=nll_val, transcript=transcript)

    gathered: list[Optional[ScoreResponse]] = [None] * world_size
    dist.all_gather_object(gathered, my_result)

    if local_rank == 0:
        return [res for res in gathered[: len(reqs)] if res is not None]
    else:
        return []

@torch.inference_mode()
def process_batchseq(reqs: List[ScoreRequest]) -> List[ScoreResponse]:
    B = len(reqs)
    my_indices = list(range(local_rank, B, world_size))
    part_results: list[tuple[float, str]] = []
    for idx in my_indices:
        r = reqs[idx]
        tok = torch.tensor(r.tokens, device=device)[None, None, :]   # (1,1,T)
        wav = codec.decode_code(tok)[0, 0]                           # (T,)
        # 2. Whisper
        transcript = whisper.transcribe(
            whisper_base, wav, fp16=True
        )["text"].strip()
        # 3. Mel
        mel = _wa.log_mel_spectrogram(
            _wa.pad_or_trim(wav.cpu().numpy()), n_mels=REQ_BINS
        )
        mel = torch.as_tensor(mel, device=device)[None]              # (1, C, L)
        # 4. token
        tgt = torch.tensor(
            [TOKENIZER.sot] + TOKENIZER.encode(r.text) + [TOKENIZER.eot],
            device=device
        )[None]                                                     # (1, L_t)
        enc    = whisper_base.encoder(mel)
        logits = whisper_base.decoder(tgt[:, :-1], enc)             # (1, L_t-1, V)
        loss   = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt[:, 1:].reshape(-1),
            ignore_index=PAD_ID,
            reduction="none"
        ).view(tgt[:, 1:].shape).mean()                            
        part_results.append((float(loss), transcript))
    gathered: list[list[tuple[float, str]]] = [None] * world_size
    dist.all_gather_object(gathered, part_results)
    if local_rank == 0:
        flat = [it for sub in gathered for it in sub]  # 展平
        return [ScoreResponse(nll=n, transcript=t) for n, t in flat]
    return []

def run_server(queue: Queue):
    app = FastAPI(title=f"Whisper DDP Master (Rank 0)")

    @app.post("/batch_score", response_model=List[ScoreResponse])
    async def batch_score(req: List[ScoreRequest]):
        job = Job(req)
        queue.put(job)
        job.event.wait()
        if job.error:
            raise HTTPException(status_code=500, detail=str(job.error))
        return job.result

    @app.get("/healthz")
    async def healthz():
        return {"rank": local_rank, "world_size": world_size}

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")

def main_loop():
    job_queue = Queue() if local_rank == 0 else None
    dist.barrier()
    if local_rank == 0:
        threading.Thread(target=run_server, args=(job_queue,), daemon=True).start()
    while True:
        job = None
        data = [None]
        if local_rank == 0:
            try:
                job = job_queue.get(block=False)
                data = [job.data]
            except Empty:
                pass
        dist.broadcast_object_list(data, src=0)
        reqs = data[0]
        if reqs:
            res = process_batch(reqs) 
            if local_rank == 0:
                job.result = res
                job.event.set()
        dist.barrier()

if __name__ == "__main__":
    main_loop()
    dist.barrier()
