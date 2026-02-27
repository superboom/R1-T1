""" Preprocess dataset for knights and knaves logic task """

import os
import time
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
from random import shuffle

import json

language = {
    'zh': 'Chinese',
    'en': 'English',
    'fr': 'French',
    'de': 'German',
    'ru': 'Russian',
    'ja': 'Japanese'
}


def make_prefix(dp, template_type):
    source_text = dp['source_text']
    source_language = language[dp['source_lang']]
    target_language = language[dp['target_lang']]

    quiz = f"Please translate the following {source_language} Source sentence into {target_language}.\nSource sentence: {source_text}"

    if template_type == 'base':
        prefix = f"""The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to translate a given text. After thinking, when you reach a conclusion, provide the translated text within <answer> </answer> tags. i.e., <answer> translated text here </answer>.\n\nUser:{quiz}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\n You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the translation. The reasoning process and translation are enclosed within <think> </think> and <answer> </answer> tags, respectively. i.e., <think> reasoning process here </think><answer> translated text here </answer>. Now the user asks you to translate a given text. After thinking, when you reach a conclusion, provide the translated text within <answer> </answer> tags. i.e., <answer> translated text here </answer>. <|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""

    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', default='')
    parser.add_argument('--train_size', type=int, default=1783)
    parser.add_argument('--test_size', type=int, default=198)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')

    args = parser.parse_args()

    data_source = 'translate'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size


    # Load custom JSON dataset
    def load_json(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert isinstance(data, list), "Input JSON must be a list of samples."
        return data


    raw_data = load_json(args.data_path)
    raw_dataset = Dataset.from_list(raw_data)
    print(len(raw_dataset))

    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE

    # 打乱数据集
    shuffled_dataset = raw_dataset.shuffle(seed=42)
    train_dataset = shuffled_dataset.select(range(TRAIN_SIZE))
    print(len(train_dataset))
    test_dataset = shuffled_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
    print(len(test_dataset))


    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target_text": example['target_text'],
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "translation",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                }
            }
            return data

        return process_fn


    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)