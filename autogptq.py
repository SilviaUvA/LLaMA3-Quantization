from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig, GPTQConfig
from auto_gptq.utils import Perplexity

from transformers import AutoTokenizer
import numpy as np
import torch
import random


def load_model(args) -> AutoGPTQForCausalLM:
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    quantize_config = BaseQuantizeConfig(
        bits=args.wbits,
        group_size=args.group_size,
        tokenizer=tokenizer,
        dataset="wikitext2"
    )

    # gptq_config = GPTQConfig(
    #     bits=args.wbits,
    #     dataset="wikitext2",
    #     tokenizer=tokenizer
    # )

    model = AutoGPTQForCausalLM.from_pretrained(
        args.model, quantization_config=quantize_config)

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
    parser.add_argument("--seed", type=int, default=2,
                        help="Seed for sampling the calibration data.")
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=None)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model, _ = load_model(args)
    model.quantize()
    model.save_quantized(args.save_dir, use_safetensors=True)
