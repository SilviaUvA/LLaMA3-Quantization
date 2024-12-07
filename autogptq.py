"""
Adapted from examples/quantization/basic_usage_wikitext2.py
from the AutoGPTQ repo.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.utils import Perplexity

from transformers import AutoTokenizer
import numpy as np
import torch
import random

from datautils import get_loaders


def load_model(args) -> tuple[AutoGPTQForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    quantize_config = BaseQuantizeConfig(
        bits=args.wbits,
        group_size=args.group_size
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        args.model, quantize_config=quantize_config)

    return model, tokenizer


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append(
            {"input_ids": inp, "attention_mask": attention_mask})
    return traindataset, testenc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str,
                        help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="../log/",
                        type=str, help="direction of logging file")
    parser.add_argument("--save_dir", default=None, type=str,
                        help="direction for saving quantization model")
    parser.add_argument("--calib_dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "ptb", "c4", "mix", "pile"],
                        help="Where to extract calibration data from.",
                        )
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Number of calibration data samples.")
    parser.add_argument("--seed", type=int, default=2,
                        help="Seed for sampling the calibration data.")
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=None)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model, tokenizer = load_model(args)

    dataloader, testloader = get_loaders(
        args.calib_dataset,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )

    traindataset, testenc = get_wikitext2(args.nsamples, args.seed,
                                          model.seqlen, args.model)
    model.quantize(traindataset, use_triton=False)
    model.save_quantized(args.save_dir)
    model.save_quantized(args.save_dir, use_safetensors=True)
