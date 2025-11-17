import argparse
import os
import json
import datasets
from verl.utils.hdfs_io import copy, makedirs
import random
random.seed(42)
def preprocess_json_to_parquet(data_source, local_dir, hdfs_dir=None):
    """
    Preprocess the Text to Speech dataset to parquet format from JSON files.
    Args:
        data_source (str): Directory where JSON files are stored.
        local_dir (str): Directory to save the processed data locally.
        hdfs_dir (str, optional): Directory to copy the processed data to HDFS.
    """
    # Load the datasets from the JSON files
    train_file = os.path.join(data_source, 'train_part.json')
    test_file = os.path.join(data_source, 'test.json')
    with open(train_file, 'r') as f:
        train_data = json.load(f)

    with open(test_file, 'r') as f:
        test_data = json.load(f)
    # Convert list of dicts into a datasets.Dataset format
    train_dataset = datasets.Dataset.from_dict({"text": [item["text"] for item in train_data],
                                                 "type": [item["type"] for item in train_data],
                                                 "Pitch": [item['Pitch'] for item in train_data ],
                                                 "Volume": [item['Volume'] for item in train_data ],
                                                 "Speed": [item['Speed'] for item in train_data ],
                                                 "Emotion": [item['Emotion'] for item in train_data ]
                                                 })
    
    test_dataset = datasets.Dataset.from_dict({"text": [item["text"] for item in test_data],
                                                "type": [item["type"] for item in test_data],
                                                 "Pitch": [item['Pitch'] for item in test_data ],
                                                 "Volume": [item['Volume'] for item in test_data ],
                                                 "Speed": [item['Speed'] for item in test_data ],
                                                 "Emotion": [item['Emotion'] for item in test_data ],
                                                })

    # Define a function to process the text data and add additional columns
    def make_map_fn(split):
        def process_fn(example, idx):
            text = example["text"]
            # Format the text
            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
            question = "Convert the text to speech:" + formatted_text

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    },
                    {
                        "role": "assistant",
                        "content": "<|SPEECH_GENERATION_START|>",
                    },
                ],
                "ability": "text-to-speech",
                "reward_model": {"style": "rule", "ground_truth": text},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "text": text,
                    "type": example.get("type", "unknown"),
                    "Pitch":example.get("Pitch", "unknown"),
                    "Volume":example.get("Volume", "unknown"),
                    "Speed":example.get("Speed", "unknown"),
                    "Emotion":example.get("Emotion", "unknown"),
                },
            }
            return data

        return process_fn

    # Map the processing function to the dataset
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Save the processed data to parquet
    train_output_path = os.path.join(local_dir, "train.parquet")
    test_output_path = os.path.join(local_dir, "test.parquet")
    train_dataset.to_parquet(train_output_path)
    test_dataset.to_parquet(test_output_path)

    print(f"Train data saved to {train_output_path}")
    print(f"Test data saved to {test_output_path}")

    # Optionally, copy the parquet files to HDFS
    if hdfs_dir:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"Data copied to HDFS at {hdfs_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="data/llasa-tts-rl-wer", help="Path to the JSON data source")
    parser.add_argument("--local_dir", default="data/llasa-tts-rl-wer", help="Local directory to store parquet files")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to store parquet files")

    args = parser.parse_args()

    preprocess_json_to_parquet(args.data_source, args.local_dir, args.hdfs_dir)