#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-process a bilingual JSON dataset into HuggingFace Parquet files that
follow a <think>/<answer> prompt template for translation tasks.

Usage Example:
--------------
python preprocess_translation.py \
    --data_path /path/to/data.json \
    --local_dir ~/data/COT \
    --train_size 900 \
    --test_size 100 \
    --template_type qwen-instruct \
    --hdfs_dir hdfs:///data/COT
"""

import argparse
import json
import os
from typing import Dict

from datasets import Dataset
from verl.utils.hdfs_io import copy, makedirs

# ---------------------------------------------------------------------------- #
# Constants
# ---------------------------------------------------------------------------- #
LANGUAGE: Dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "ja": "Japanese",
}

# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #
def load_json(path: str) -> Dataset:
    """Load a standard JSON file (array of objects) into a HuggingFace Dataset."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return Dataset.from_list(data)


def make_prefix(example: dict, *, template_type: str) -> str:
    """Build the prompt prefix for one sample."""
    source_lang = LANGUAGE[example["source_lang"]]
    target_lang = LANGUAGE[example["target_lang"]]
    source_sentence = example["source_text"]

    quiz = (
        f"Please translate the following {source_lang} Source sentence into "
        f"{target_lang}.\nSource sentence: {source_sentence}"
    )

    if template_type == "base":
        return (
            "The user asks a question, and the Assistant solves it. The assistant "
            "first thinks about the reasoning process in the mind and then "
            "provides the user with the final answer. The reasoning process and "
            "answer are enclosed within <think></think> and <answer></answer> "
            "tags, respectively. Now the user asks you to translate a given text. "
            "After thinking, when you reach a conclusion, provide the translated "
            "text within <answer></answer> tags.\n\n"
            f"User:{quiz}\nAssistant: <think>"
        )

    # Default: qwen-instruct
    return (
        "<|im_start|>system You are a helpful assistant. The assistant first thinks "
        "about the reasoning process in the mind and then provides the translation. "
        "The reasoning process and translation are enclosed within <think></think> "
        "and <answer></answer> tags, respectively. Now the user asks you to "
        "translate a given text. After thinking, when you reach a conclusion, "
        "provide the translated text within <answer></answer> tags. <|im_end|>\n"
        "<|im_start|>user\n"
        f"{quiz}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n<think>"
    )


def build_record(example: dict, template_type: str) -> dict:
    """Convert a raw example to the training-ready format."""
    prompt = make_prefix(example, template_type=template_type)
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "translation",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target_text": example["target_text"]},
        },
    }


# ---------------------------------------------------------------------------- #
# Main script
# ---------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="Path to the input JSON file (array format)")
    parser.add_argument("--local_dir", default="~/data/COT",
                        help="Directory to write Parquet files")
    parser.add_argument("--hdfs_dir", default=None,
                        help="Optional HDFS directory to sync output")
    parser.add_argument("--train_size", type=int, default=900)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--template_type", choices=["base", "qwen-instruct"],
                        default="qwen-instruct")
    args = parser.parse_args()

    # Resolve directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Load dataset
    raw_ds = load_json(args.data_path)
    if len(raw_ds) < args.train_size + args.test_size:
        raise ValueError(
            f"Dataset size {len(raw_ds)} < required {args.train_size + args.test_size}"
        )

    # Train/Test split
    train_ds = raw_ds.select(range(args.train_size))
    test_ds = raw_ds.select(range(args.train_size, args.train_size + args.test_size))

    # Transform each split
    train_ds = train_ds.map(
        lambda ex: build_record(ex, args.template_type), num_proc=4)
    test_ds = test_ds.map(
        lambda ex: build_record(ex, args.template_type), num_proc=4)

    # Save
    train_path = os.path.join(local_dir, "train.parquet")
    test_path = os.path.join(local_dir, "test.parquet")
    train_ds.to_parquet(train_path)
    test_ds.to_parquet(test_path)
    print(f"✓ Saved train → {train_path}")
    print(f"✓ Saved test  → {test_path}")

    # HDFS sync
    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"✓ Copied to HDFS → {args.hdfs_dir}")


if __name__ == "__main__":
    main()
