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

We provide a curated 2K corpus that can be used for two purposes:

- Generating CoT-based SFT data for cold-start initialization
- Preparing training data directly for reinforcement learning

### Generate Cold-Start SFT Data

We utilize an advanced LLM API to construct the cold-start data. You can either:

- Use our **Human-Aligned Chain-of-Thought (CoT) Templates** described in the paper to generate your own data, or  
- Directly download our prepared CoT-annotated corpus [here](https://huggingface.co/datasets/superboom/r1t1-cot-corpus).


### Generate RL Training Data

```bash
cd data
python data_prepare.py
```


## Citation

If you find this project helpful, please cite:

```bibtex
@article{he2025r1,
  title={R1-T1: Fully incentivizing translation capability in LLMs via reasoning learning},
  author={He, Minggui and Liu, Yilun and Tao, Shimin and Luo, Yuanchang and Zeng, Hongyong and Su, Chang and Zhang, Li and Ma, Hongxia and Wei, Daimeng and Meng, Weibin and others},
  journal={arXiv preprint arXiv:2502.19735},
  year={2025}
}
```

## Acknowledgements
Our training parallel corpus comes from the foloowing sources:
- [Verl](https://github.com/facebookresearch/VERL)
- [WMT](https://machinetranslate.org/wmt)
- [UN Corpus](https://www.un.org/dgacm/zh/content/uncorpus/)
- [ZHIJI Corpus](https://www.jizhi-dataset.top/index/category/detail/15)
- [Opensource ZH_CN Corpus](https://huggingface.co/datasets/joefox/newstest-2017-2019-ru_zh)
## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.