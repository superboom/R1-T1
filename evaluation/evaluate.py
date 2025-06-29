#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, logging, torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from textwrap import dedent

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model

# Configurable paths
FLORES_DIR  = Path("./data/flores_raw/")
HARD_DIR    = Path("./data/hard_raw/")
OUTPUT_ROOT = {
    "flores": Path("./outputs/flores_results"),
    "hard":   Path("./outputs/hard_results"),
}

BATCH_SIZE = 1024
SAMPLING_KWARGS = dict(top_k=64, top_p=0.75, temperature=0.1, max_tokens=8192)

LANG_TAGS  = ['ar','de','es','fr','it','ja','kr','pt','ru','th','tr',
              'en','zh','nl','pl','ms','id','el','cs','vi','tl']
LANG_NAMES = ['Arabic','German','Spanish','French','Italian',
              'Japanese','Korean','Portuguese','Russian','Thai','Turkey',
              'English','Chinese','Dutch','Polish','Malaysian','Indonesian',
              'Greek','Czech','Vietnamese','Tagalog']
LANG_MAP = dict(zip(LANG_TAGS, LANG_NAMES))

PARROT_LANG = {
    'de': {'de':"Deutsch",'en':"Englisch",'ja':"Japanisch",'zh':"Chinesisch"},
    'en': {'de':"German",'en':"English",'ja':"Japanese",'zh':"Chinese"},
    'ja': {'de':"ドイツ語",'en':"英語",'ja':"日本語",'zh':"中国語"},
    'zh': {'de':"德语",'en':"英语",'ja':"日语",'zh':"中文"},
}


def read_lines(p: Path) -> List[str]:
    return p.read_text(encoding="utf-8").splitlines()

def write_jsonl(p: Path, rows: List[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")

@dataclass
class Translator:
    model_dir: str; gpu_mem: float; tp_size: int; method: int
    tokenizer: AutoTokenizer = field(init=False)
    model:     object        = field(init=False)
    sampling:  SamplingParams= field(init=False)
    vllm_backend: bool       = field(init=False)

    def __post_init__(self):
        self._load_model()
        self.sampling = SamplingParams(**SAMPLING_KWARGS)

    def _load_model(self):
        try:
            self.model = LLM(self.model_dir, gpu_memory_utilization=self.gpu_mem,
                             tensor_parallel_size=self.tp_size, trust_remote_code=True)
            self.vllm_backend = True
            logging.info("Loaded with vLLM.")
        except Exception as e:
            logging.warning("vLLM failed (%s). Falling back to HF.", e)
            with init_empty_weights():
                base = AutoModelForCausalLM.from_pretrained(
                    self.model_dir, torch_dtype=torch.float16,
                    trust_remote_code=True)
            max_mem={i:f"{int(self.gpu_mem*100)}GiB" for i in range(torch.cuda.device_count())}
            dm = infer_auto_device_map(base, max_memory=max_mem)
            self.model = dispatch_model(base, dm)
            self.vllm_backend=False
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, use_fast=("bloom" in self.model_dir.lower()),
            trust_remote_code=True)

    def _default_template(self)->str:
        return dedent("""
        {% for m in messages -%}
        <|im_start|>{{ m['role'] }}
        {{ m['content'] }}<|im_end|>
        {% endfor -%}
        {% if add_generation_prompt -%}
        <|im_start|>assistant
        {% endif %}
        """)

    def build_prompt(self, txt:str, src:str, tgt:str)->str:
        if self.method==4:
            src_h = PARROT_LANG['en'].get(src, src)
            tgt_h = PARROT_LANG['en'].get(tgt, tgt)
            instruction = f"We are translating the following sentences from {src_h} to {tgt_h}."
            return ( "Below is an instruction that describes a task, paired with an input that provides further context. "
                     "Write a response that appropriately completes the request.\n\n"
                     f"### Instruction:\n{instruction}\n\n"
                     f"### Input:\n{txt}\n\n"
                     "### Response:" )

        if self.method==1:
            user_msg=(f"You are a helpful assistant. The assistant first thinks ... "
                      f"Translate the following sentence from {src} to {tgt} and explain your reasoning:\n{txt}")
            msgs=[{"role":"user","content":user_msg}]
        elif self.method==2:
            sys_msg=("You are a helpful assistant. The assistant first thinks about the reasoning process ...")
            user_msg=(f"Please translate the following {src} source sentence into {tgt}.\nSource sentence: {txt}")
            msgs=[{"role":"system","content":sys_msg},{"role":"user","content":user_msg}]
        else:
            msgs=[{"role":"user","content":f"Translate the following {src} source text to {tgt}:\n{src}:{txt}\n\n{tgt}:"}]

        try:
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except ValueError:
            if self.tokenizer.chat_template is None:
                self.tokenizer.chat_template=self._default_template()
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def translate_batch(self, lines:List[str], s:str, t:str)->List[dict]:
        prompts=[self.build_prompt(l, LANG_MAP[s], LANG_MAP[t]) for l in lines]
        if self.vllm_backend:
            outs=self.model.generate(prompts,sampling_params=self.sampling)
            gens=[o.outputs[0].text for o in outs]
        else:
            tok=self.tokenizer(prompts,return_tensors="pt",padding=True,
                               truncation=True).to("cuda:0")
            ids=self.model.generate(**tok,max_new_tokens=self.sampling.max_tokens,
                                    top_k=self.sampling.top_k,top_p=self.sampling.top_p,
                                    temperature=self.sampling.temperature)
            gens=self.tokenizer.batch_decode(ids,skip_special_tokens=True)
        return [{"prompt":p,"res":g} for p,g in zip(prompts,gens)]

PIVOT_LANGS = ["en", "zh"]
ALL_LANGS   = LANG_TAGS

def flores_task(tr: Translator, alias: str):
    pivots   = set(PIVOT_LANGS)
    others   = [l for l in ALL_LANGS if l not in pivots]
    pairs = [(src, tgt) for src in others for tgt in pivots] + \
            [(src, tgt) for src in pivots for tgt in others] + \
            [("zh", "en"), ("en", "zh")]
    for s, t in pairs:
        logging.info("FLORES %s → %s", s, t)
        src_lines = read_lines(FLORES_DIR / f"flores.{s}")
        out_rows  = tr.translate_batch(src_lines, s, t)
        out_path  = OUTPUT_ROOT["flores"] / alias / f"flores.{s}2{t}.out"
        write_jsonl(out_path, out_rows)

def hard_task(tr: Translator, alias: str):
    HARD_CASES = [
        ("commonmt_la.zh",  'zh','en','commonmt_la'),
        ("commonmt_cl_sa.zh",'zh','en','commonmt_cl_sa'),
        ("commonmt_ct_sa.zh",'zh','en','commonmt_ct_sa'),
        ("term.en",         'en','de','term'),
        *[(f"cul_en_{l}.en",'en',l,'cul') for l in ['es','fr','zh']],
        *[(f"cul_en_{l}.{l}",l,'en','cul') for l in ['es','fr','zh']],
        ("literature.en",'en','zh','literature'),
        ("literature.zh",'zh','en','literature'),
    ]
    for file, s, t, tag in HARD_CASES:
        logging.info("HARD %s→%s (%s)", s, t, tag)
        src_lines = read_lines(HARD_DIR / file)
        out_rows  = tr.translate_batch(src_lines, s, t)
        out_path  = OUTPUT_ROOT["hard"] / alias / f"{tag}.{s}2{t}.out"
        write_jsonl(out_path, out_rows)

TASKS = [flores_task, hard_task]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir",   required=True, help="Path to the base model directory")
    ap.add_argument("--model_alias", default="r1-t1", help="Alias for saving outputs")
    ap.add_argument("--tp",          type=int,   default=4,  help="Tensor parallel size for vLLM")
    ap.add_argument("--gpu_mem",     type=float, default=0.9,help="Per-GPU memory upper bound (0-1)")
    ap.add_argument("--method",      type=int,   default=6,  choices=[0,1,2,4])
    return ap.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    translator = Translator(args.model_dir, args.gpu_mem, args.tp, args.method)
    for task in TASKS:
        task(translator, args.model_alias)

if __name__ == "__main__":
    main()