from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import soundfile as sf

llasa_3b ='HKUSTAudio/Llasa-3B'

tokenizer = AutoTokenizer.from_pretrained(llasa_3b)
model = AutoModelForCausalLM.from_pretrained(llasa_3b)
model.eval() 
model.to('cuda')

from xcodec2.modeling_xcodec2 import XCodec2Model
 
model_path = "HKUSTAudio/xcodec2"  
 
Codec_model = XCodec2Model.from_pretrained(model_path)
Codec_model.eval().cuda()   
# only 16khz speech support!
prompt_wav, sr = sf.read("太乙真人.wav")   # you can find wav in Files
#prompt_wav, sr = sf.read("Anna.wav") # English prompt
prompt_wav = torch.from_numpy(prompt_wav).float().unsqueeze(0)  

prompt_text ="对，这就是我万人敬仰的太乙真人，虽然有点婴儿肥，但也掩不住我逼人的帅气。"
#promt_text = "A chance to leave him alone, but... No. She just wanted to see him again. Anna, you don't know how it feels to lose a sister. Anna, I'm sorry, but your father asked me not to tell you anything."
target_text = '突然，身边一阵笑声。我看着他们，意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道："我身上的肉，是为了掩饰我爆棚的魅力，否则，岂不吓坏了你们呢？"'
#target_text = "Dealing with family secrets is never easy. Yet, sometimes, omission is a form of protection, intending to safeguard some from the harsh truths. One day, I hope you understand the reasons behind my actions. Until then, Anna, please, bear with me."
input_text = prompt_text   + target_text

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
with torch.no_grad():
    # Encode the prompt wav
    vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav)
    print("Prompt Vq Code Shape:", vq_code_prompt.shape )   

    vq_code_prompt = vq_code_prompt[0,0,:]
    # Convert int 12345 to token <|s_12345|>
    speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

    formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

    # Tokenize the text and the speech prefix
    chat = [
        {"role": "user", "content": "Convert the text to speech:" + formatted_text},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
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
        top_p=1,           
        temperature=0.8,
    )
    # Extract the speech tokens
    generated_ids = outputs[0][input_ids.shape[1]-len(speech_ids_prefix):-1]

    speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)   

    # Convert  token <|s_23456|> to int 23456 
    speech_tokens = extract_speech_ids(speech_tokens)

    speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)

    # Decode the speech tokens to speech waveform
    gen_wav = Codec_model.decode_code(speech_tokens) 

    # if only need the generated part
    # gen_wav = gen_wav[:,:,prompt_wav.shape[1]:]

sf.write("gen.wav", gen_wav[0, 0, :].cpu().numpy(), 16000)
