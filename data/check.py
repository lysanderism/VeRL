import json,re 
import unicodedata
with open('data/llasa-tts-rl-s/train_simple.json', 'r') as f:   
    unique = json.load(f)

with open('data/llasa-tts-rl-s/train_raw.json','r') as f:
    data = json.load(f)


def strip_punctuation(text: str) -> str:
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        out.append(' ' if cat.startswith(('P', 'S')) else ch)
    return re.sub(r'\s+', ' ', ''.join(out)).strip()


output_data = []
unique_text= [item['text'] for item in unique]

for item in data:
    if item['text'] in unique_text:
        output_data.append(item)

# clean = []
# for i, item in enumerate(output_data):
#     if str.lower(strip_punctuation(item['text'])) not in unique_text and not any(count_common_words(item['text'], unique_item) > 4 for unique_item in unique_text):
#         if len(item['text'].split())>6:
#             clean.append(item)

with open('data/train_clean_part.json', 'w') as f:   
    json.dump(output_data, f, indent=4, ensure_ascii=False)