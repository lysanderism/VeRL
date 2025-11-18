import pandas as pd
import json

# 路径按实际修改
parquet_file = 'data/llasa-tts-rl-s/train.parquet'
output_json = 'data/llasa-tts-rl-s/train_simple.json'

type_column = 'role'   # 如果你想用 style/emotion/ability，改这里

# 读取 parquet
df = pd.read_parquet(parquet_file)

# 若 id 字段实际是 'index' 或 'split'，请根据实际更换
out_list = []
for _, row in df.iterrows():
    out_list.append({
        'text': row['text'],
        'type': "type",#row['extra_info']['type'],
        'id': str(row['extra_info']['index'])+'.wav',  
    })

# 保存为 json
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(out_list, f, ensure_ascii=False, indent=2)

print(f"写入完成：{output_json}")