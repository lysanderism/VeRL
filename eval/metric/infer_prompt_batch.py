import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import torchaudio
import transformers
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import soundfile as sf
from xcodec2.modeling_xcodec2 import XCodec2Model
from accelerate import Accelerator
accelerator = Accelerator()
from accelerate.utils import gather_object
from torch.nn.utils.rnn import pad_sequence

device = f"cuda:{accelerator.process_index}"

@dataclass
class TestArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    model_path: Optional[str] = field(default=None, metadata={"help": "model dir"})
    data_file: Optional[str] = field(default=None, metadata={"help": "test file"})
    audio_dir: Optional[str] = field(default=None, metadata={"help": "audio dir"})
    out_audio: Optional[str] = field(default=None, metadata={"help": "output file for test"})
    batch_size: Optional[int] =  field(default=16, metadata={"help": "batch size"})

    def __post_init__(self):
        if self.model_path is None:
            raise ValueError("config path should not none")

def ids_to_speech_tokens(speech_ids):
 
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def extract_speech_ids(speech_tokens_str):
 
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

def _get_audio(wav_path):
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    audio = waveform#[0]
    print(audio.shape)
    return audio

def _get_message(obj_dict, Codec_model):
    # Encode the prompt wav
    prompt_wav = _get_audio(obj_dict['prompt_audio'])
    vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav)
    print("Prompt Vq Code Shape:", vq_code_prompt.shape )

    vq_code_prompt = vq_code_prompt[0,0,:]
    # Convert int 12345 to token <|s_12345|>
    speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

    formatted_text = f"<|TEXT_UNDERSTANDING_START|>{obj_dict['prompt_text']}{obj_dict['text']}<|TEXT_UNDERSTANDING_END|>"
    # Tokenize the text
    message = [
        # {"role": "system", "content": content},
        {"role": "user", "content": "Convert the text to speech:" + formatted_text},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"+ ''.join(speech_ids_prefix)}
    ]

    return message

def main():
    parser = HfArgumentParser(TestArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    transformers.logging.set_verbosity_info()
    logging.info(data_args)

    out_dir = data_args.out_audio
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(data_args.model_path)
    model = AutoModelForCausalLM.from_pretrained(data_args.model_path)
    model.eval() 
    model.to('cuda')
    
    Codec_model = XCodec2Model.from_pretrained("HKUST-Audio/xcodec2")
    Codec_model.eval().cuda()   

    datas = []
    with open(data_args.data_file, "r") as f:
        datas = json.load(f)

    all_outputs = []
    batch_size = data_args.batch_size
    
    for i in tqdm(range(0, len(datas), batch_size)):
        batch_data = datas[i : i + batch_size]

        batch_messages = []
        batch_ids = []
        batch_prompt_audios = []
        for bd in batch_data:
            batch_messages.append(_get_message(bd, Codec_model))
            batch_ids.append(bd["id"])
            # batch_prompt_audios.append(_get_audio(bd["audio"]).numpy())

        with accelerator.split_between_processes(batch_messages) as messages, accelerator.split_between_processes(batch_ids) as batch_id:
            accelerator.wait_for_everyone()
            # input_ids = [
            #     tokenizer.apply_chat_template(msg, tokenize=True, 
            #         return_tensors='pt', 
            #         continue_final_message=True)
            #     for msg in batch_messages
            # ]
            
            input_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, 
                    return_tensors='pt', 
                    continue_final_message=True).to(model.device)
            speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
            # print(input_ids.shape)
            outputs = model.generate(
                input_ids,
                max_length=2048,  # We trained our model with a max length of 2048
                eos_token_id= speech_end_id,
                do_sample=True,    
                top_p=1,           #  Adjusts the diversity of generated content
                temperature=0.8,   #  Controls randomness in output
            )
            generated_ids = outputs[0][input_ids.shape[1]:-1]
            speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)   
            speech_tokens_raw = [speech_tokens]
            
            speech_tokens = extract_speech_ids(speech_tokens)

            speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)

            # Decode the speech tokens to speech waveform
            response = Codec_model.decode_code(speech_tokens) 
            sf.write(out_dir+'/'+batch_id[0], response[0, 0, :].cpu().numpy(), 16000)

            speech_codes = gather_object(speech_tokens_raw)
            if accelerator.is_main_process:
                print(speech_codes, len(speech_codes))
        all_outputs.extend(speech_codes)

    final_output = []

    for input_example, model_output in zip(datas, all_outputs):
        result = input_example
        result_new = {
            "id": result["id"],
            "text": result["text"],
            "model_prediction": model_output
        }
        final_output.append(result_new)

    # Save results to a JSON file
    output_path = "eval/results.json"
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
