"""
GPTQ quantization using AutoGPTQ implicitly in transformers module
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig

from transformers import AutoTokenizer
import numpy as np
import torch
import random


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    gptq_config = GPTQConfig(
        bits=args.wbits,
        group_size=args.group_size,
        tokenizer=tokenizer,
        model_seqlen=args.seqlen,
        dataset=args.calib_dataset
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=gptq_config,
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        trust_remote_code=True
    )

    return model, tokenizer


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
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Sequence length of model.")
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
    model.save_pretrained(args.save_dir, use_safetensors=True)
    tokenizer.save_pretrained(args.save_dir)
