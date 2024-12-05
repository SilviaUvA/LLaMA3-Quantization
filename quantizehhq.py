import argparse
import torch
# from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from hqq.models.hf.llama import LlamaHQQ  # Import for LLaMA-specific quantization
from hqq.core.quantize import *
from hqq.models.hf.base import AutoHQQHFModel
# from hqq.models.hf.base import BaseQuantizeConfig  # Quantization config

# Login to Hugging Face
with open("huggingface_access_token.txt") as f:
    access_token = f.readline().strip()

login(token=access_token)

def OLDload_llama_model_and_tokenizer(model_name, device="cuda"):
    """Load LLaMA model and tokenizer from Hugging Face."""
    print(f"Loading LLaMA model {model_name}...")
    # Use LlamaHQQ for loading and quantization
    model = LlamaHQQ.from_pretrained(
        model_name, 
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model.to(device), tokenizer

def load_llama_model_and_tokenizer(model_name, device="cuda"):
    """Load LLaMA3 model and tokenizer from Hugging Face."""
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model.to(device), tokenizer

def quantize_llama_model(model, num_bits, group_size, compute_dtype, device):
    """Quantize a LLaMA model using LlamaHQQ."""
    print(f"Quantizing LLaMA model with {num_bits}-bit precision and group size {group_size}...")
    quant_config = BaseQuantizeConfig(nbits=num_bits, group_size=group_size)
    quantized_model = LlamaHQQ.quantize_model(
        model, 
        quant_config=quant_config, 
        compute_dtype=compute_dtype, 
        device=device
    )
    return quantized_model

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path from Hugging Face")
    parser.add_argument("--bits", type=int, default=4, help="Number of bits for quantization (default: 4)")
    parser.add_argument("--group_size", type=int, default=64, help="Group size for quantization (default: 64)")
    parser.add_argument("--dtype", type=str, default="torch.float16", help="Compute data type (default: torch.float16)")
    args = parser.parse_args()

    # Load LLaMA model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = eval(args.dtype)  # Convert string to dtype
    model, tokenizer = load_llama_model_and_tokenizer(args.model, device)

    # Quantize the LLaMA model
    model = quantize_llama_model(model, args.bits, args.group_size, compute_dtype, device)

    
    save_path = f"quantized-llama-hqq-{args.model.split('/')[-1]}-{args.bits}bit"
    print(f"Saving quantized LLaMA model to {save_path}...")
    # Save the quantized model
    quantized_model = AutoHQQHFModel.save_quantized(model, save_path)

    # quantized_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    main()