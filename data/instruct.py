import os
import pyarrow.parquet as pq
import pandas as pd
import time
import io
import wave
import numpy as np
from tqdm import tqdm
import json

with open('data/llasa-tts-rl-s/train_clean_part.json','r') as f:
    data = json.load(f)

result = []
i = 0

for row in data:
    if "?" in row['text']:
        type_text = "QUESTIONS"
    elif "emotion" in row['text']:
        type_text = "EMOTION"
    elif "<" in row['text'] or "(" in row['text'] or ".." in row['text']:
        type_text = "PUNCTUATION"
    elif "that" in row['text'] or 'beacuse' in row['text']:
        type_text = "SYNTACTIC_COMPLEXITY"
    elif "-" in row['text']:
        type_text = "COMPOUND_NOUNS"
    else:
        type_text = "EMOTION"
    item = {
        'id': row['id']+'.wav',
        'text': row['text'],
        'type': type_text,

    }
    result.append(item)
    i+=1

with open('data/train_clean_part.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

