#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=eval_q_model
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=eval_q_model_output_%A.out
#SBATCH --error=eval_q_model_error_%A.out

module purge
module load 2022
module load Anaconda3/2022.05
module load cuDNN/8.6.0.163-CUDA-11.8.0

source ~/.bashrc
cd $HOME/LLaMA3-Quantization

conda activate llama

pip uninstall transformers
pip install transformers==4.37.2
pip install protobuf==3.20.2


llama3_8b="meta-llama/Meta-Llama-3-8B"  # Llama 3 8B
llama3_70b="meta-llama/Meta-Llama-3-70B"  # Llama 3 70B
llava_next_8b="lmms-lab/llama3-llava-next-8b" # Llava Next 8B
model=${llama3_8b}

gptq="LLaMA-3-8B-GPTQ-4bit-b128" # GPTQ
awq="LLaMA-3-8B-AWQ-4bit-b128" # AWQ
quip="LLaMA-3-8B-QuIP-2bit" # QuIP
dbllm="LLaMA-3-8B-DB-LLM-2bit-fake" # DB-LLM
pbllm="LLaMA-3-8B-PB-LLM-1.7bit-fake" # PB-LLM
billm="LLaMA-3-8B-BiLLM-1.1bit-fake " # BiLLM
smoothquant4="LLaMA-3-8B-SmoothQuant-4bit-4bit" # SmoothQuant 4bit
smoothquant8="LLaMA-3-8B-SmoothQuant-8bit-8bit" # SmoothQuant 8bit

quantization_model=${awq}

tasks_commonsenseQA="piqa,arc_easy,arc_challenge,hellaswag,winogrande"

# Evaluate with quantized model from Efficient-ML's huggingface

# AWQ
python3 main.py --model "Efficient-ML/"${awq} --quant_method awq  --wbits 4 --epochs 0 --eval_ppl --output_dir ./log/${awq} --lwc --net "llama-7b" --group_size 128 --tasks ${tasks_commonsenseQA}

# SmoothQuant 4-bit
python3 main.py --model "Efficient-ML/"${smoothquant4} --quant_method smoothquant4  --wbits 4 --epochs 0 --eval_ppl --output_dir ./log/${smoothquant4} --lwc --net "llama-7b" --group_size 128 --tasks ${tasks_commonsenseQA}

# SmoothQuant 8-bit
python3 main.py --model "Efficient-ML/"${smoothquant8} --quant_method smoothquant8  --wbits 8 --epochs 0 --eval_ppl --output_dir ./log/${smoothquant8} --lwc --net "llama-7b" --group_size 128 --tasks ${tasks_commonsenseQA}
