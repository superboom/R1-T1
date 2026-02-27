# R1-T1: Fully Incentivizing Translation Capability in LLMs via Reasoning Learning

## Overview

**R1-T1** is a reasoning-enhanced machine translation framework built upon the [VERL](https://github.com/facebookresearch/VERL) repository. This project aims to fully incentivize large language models (LLMs) to perform high-quality translations through explicit reasoning learning, leveraging Chain-of-Thought (CoT) style supervision and customized reward modeling.

## Features

- **General-Purpose Reasoning-Based Machine Translation**  
  R1-T1 extends reasoning-based translation beyond specific sub-tasks, applying it to general machine translation across diverse domains such as legal, medical, and idiomatic expressions. This approach covers six languages and multiple translation directions, demonstrating adaptability to various translation scenarios.

- **Human-Aligned Chain-of-Thought (CoT) Templates**  
  The framework formalizes six expert-curated CoT templates that mirror hybrid human translation strategies, including context-aware paraphrasing and back translation. These templates guide the model to emulate structured, multi-layered reasoning processes used by professional translators.

- **Self-Evolving Translation CoTs via Reinforcement Learning**  
  R1-T1 employs reinforcement learning with specially designed rewards to enable the model to autonomously discover and evolve novel CoT trajectories for unseen translation tasks. This self-evolving capability enhances the model's adaptability and performance without relying solely on supervised fine-tuning.

- **Enhanced Translation Performance on Low-Resource Languages**  
  Experimental results indicate that R1-T1 achieves steady translation performance improvements in 10+ languages and 40+ translation directions on the Flores-101 test set, 4 domain-specific transaltion tasks. This demonstrates the model's generalization ability on  broader MT scenarios.


## Installation

```bash
git clone https://github.com/superboom/R1-T1.git
cd R1-T1
pip install -r requirements.txt
# For sglang-based reasoning models
pip install -r requirements_sglang.txt
```

## Data Preparation

We provide a curated [2K corpus](https://github.com/superboom/R1-T1/blob/main/data/corpus.json) that can be used for two purposes:

- Generating CoT-based SFT data for cold-start initialization
- Preparing training data directly for reinforcement learning

### Generate Cold-Start SFT Data

We utilize an advanced LLM API to construct the cold-start data. You can either:

- Use our **Human-Aligned Chain-of-Thought (CoT) Templates** described in the paper to generate your own data, or  
- Directly download our prepared CoT-annotated corpus [here](https://github.com/superboom/R1-T1/blob/main/data/CoT_fata.json).


### Generate RL Training Data
We provide the RL training data in ready-to-use parquet format.
You may either:

Directly use our released parquet data for training reproduction [here](https://github.com/superboom/R1-T1/blob/main/data/train.parquet); or

Generate your own RL data from a custom parallel corpus using the preprocessing script below.

```bash
cd data
python data_prepare.py \
    --data_path /path/to/corpus.json \
    --local_dir ~/data/COT_RL \
    --train_size 1783 \
    --test_size 198 \
    --template_type qwen-instruct  
```

## Training

To start reinforcement learning training with GRPO, simply run:

```bash
bash main_grpo.sh
```

### Reward API Configuration
R1-T1 uses COMET as the core reward model to evaluate translation quality. Before running the training, you must set up your own reward inference API and configure the endpoint in the reward module:

Edit the following line in reward/translate.py:

```python
COMET_API_URL = "http://<internal_ip>:<port>/score"
```
Replace <internal_ip>:<port> with the address of your COMET scoring service. We use COMET-DA models to compute sentence-level translation quality during reward estimation.

Ensure your COMET server is accessible and returns a valid JSON response like:

```json
{ "system_score": 3.42 }

```

This COMET-based reward guides the model to align with human-level translation preferences.

### Model Path Configuration

In `main_grpo.sh`, you must modify the following line to specify the correct initial model:

```bash
--actor_rollout_ref.model.path=$MODEL_PATH
```

Replace $MODEL_PATH with one of the following options:

- R1-T1 — if you want to continue training from our CoT-SFT checkpoint

- R1-T1-ZERO — if you want to start from the original Qwen-7B-Instruct model without post CoT sft

## Evaluation

We provide an evaluation script to test multilingual translation models on:

- **FLORES Benchmark** — General multilingual translation across 21 languages.
- **Domain-Specific Benchmarks** — Challenging translations involving literature, Common sense etc.


### Step 1: Prepare Input Data

Create the following directories and place evaluation files inside:

./data/flores_raw/

./data/hard_raw/

- For FLORES, files should be named like `flores.zh`, `flores.en`, etc.
- For hard tests, include files like `commonmt_la.zh`, `literature.en`, etc.

You can customize the input format according to your own test corpora if needed.

### Step 2: Run Evaluation

```bash
python evaluation/evaluate.py \
  --model_dir /path/to/your/model \
  --model_alias r1-t1 \
  --tp 4 \
  --gpu_mem 0.9 \
  --method 1
  ```
  Arguments:
  
  --model_dir: Path to your vLLM-compatible or HuggingFace-style model (e.g., R1-T1, R1-T1-ZERO)
  
  --model_alias: Custom tag for naming result folders (e.g., r1-t1)
  
  --tp: Tensor parallel size (default 4 for multi-GPU setups)
  
  --gpu_mem: Per-GPU memory usage (e.g., 0.9 for 90%)
  
  --method: Prompting method:
  
  1: Reasoning-based CoT prompt
  
  2: System + user prompt
  
  4: ParroT-style instruction

### Step 3: Output Results
All generated translations are saved to:

```bash
./outputs/flores_results/{model_alias}/flores.{src}2{tgt}.out

./outputs/hard_results/{model_alias}/{domain}.{src}2{tgt}.out
```
Each file is a .jsonl list of:

```json
{ "prompt": "...", "res": "..." }
```
These outputs can be used for further evaluation (e.g., BLEU, COMET, human annotation).


## Citation

If you find this project helpful, please cite:

[//]: # (```bibtex)

[//]: # (@article{he2025r1,)

[//]: # (  title={R1-T1: Fully incentivizing translation capability in LLMs via reasoning learning},)

[//]: # (  author={He, Minggui and Liu, Yilun and Tao, Shimin and Luo, Yuanchang and Zeng, Hongyong and Su, Chang and Zhang, Li and Ma, Hongxia and Wei, Daimeng and Meng, Weibin and others},)

[//]: # (  journal={arXiv preprint arXiv:2502.19735},)

[//]: # (  year={2025})

[//]: # (})

[//]: # (```)

## Acknowledgements
Our training parallel corpus comes from the foloowing sources:
- [Verl](https://github.com/facebookresearch/VERL)
- [WMT](https://machinetranslate.org/wmt)
- [UN Corpus](https://www.un.org/dgacm/zh/content/uncorpus/)
- [ZHIJI Corpus](https://www.jizhi-dataset.top/index/category/detail/15)
- [Opensource ZH_CN Corpus](https://huggingface.co/datasets/joefox/newstest-2017-2019-ru_zh)
## License
