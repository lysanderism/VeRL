import os
import pyarrow.parquet as pq
import pandas as pd
import time
import io
import wave
import numpy as np
from tqdm import tqdm
import uuid 
## seen
def see_parquet_files(parquet_files):
    for file in tqdm(parquet_files):
        
        parquet_file = pq.ParquetFile(file)
        columns = parquet_file.schema.names  # 获取列名
        print(f"Columns in {file}: {columns}")
        df_batch = parquet_file.read().to_pandas().head(10)
        print(df_batch)

if __name__ == "__main__":
    # parquet_files = [os.path.join(folder_path, f) for f in parquet_files]
    parquet_files = ['data/llasa-tts-rl-s/test.parquet']
    see_parquet_files(parquet_files)
