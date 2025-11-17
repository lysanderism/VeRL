from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import soundfile as sf
import json
llasa ='HKUSTAudio_model/Llasa-1B'

tokenizer = AutoTokenizer.from_pretrained(llasa)
model = AutoModelForCausalLM.from_pretrained(llasa)
model.eval() 
model.to('cuda')

from xcodec2.modeling_xcodec2 import XCodec2Model
 
model_path = "HKUST-Audio/xcodec2"  
 
Codec_model = XCodec2Model.from_pretrained(pretrained_model_name_or_path="model/xcodec2")
Codec_model.eval().cuda()   

input_text = [
    "apple",
    "banana",
    "cheese",
    "That",
    "dis",
    "egg",
    "finger",
    "ground",
    "happy",
    "image",
    "Thank you",
    "good",
    "joke",
    "align",
    "day two three one",
    "my name is",
    "I am a student",
  ]
# input_text = '突然，身边一阵笑声。我看着他们，意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道："我身上的肉，是为了掩饰我爆棚的魅力，否则，岂不吓坏了你们呢？"'
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

#TTS start!
total = []
for text in input_text:
    with torch.no_grad():
    
        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"

        # Tokenize the text
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
        ]

        input_ids = tokenizer.apply_chat_template(
            chat, 
            tokenize=True, 
            return_tensors='pt', 
            continue_final_message=True
        )
        input_ids = input_ids.to('cuda')
        speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

        # Generate the speech autoregressively
        outputs = model.generate(
            input_ids,
            max_length=2048,  # We trained our model with a max length of 2048
            eos_token_id= speech_end_id ,
            do_sample=True,    
            top_p=1,           #  Adjusts the diversity of generated content
            temperature=0.8,   #  Controls randomness in output
        )
        # Extract the speech tokens
        generated_ids = outputs[0][input_ids.shape[1]:-1]

        speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)   
        # print(speech_tokens, text)
        # total.append((speech_tokens))
        total.append({
            "speech_tokens": speech_tokens,
            "text": text
        })
        # Convert  token <|s_23456|> to int 23456 
        # speech_tokens = extract_speech_ids(speech_tokens)
        # speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)

        # # Decode the speech tokens to speech waveform
        # gen_wav = Codec_model.decode_code(speech_tokens) 
with open("sample.json", "w") as f:
    json.dump(total, f, indent=4, ensure_ascii=False)

# sf.write("gen.wav", gen_wav[0, 0, :].cpu().numpy(), 16000)
