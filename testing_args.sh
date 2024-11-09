#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=testing_args
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:01:00
#SBATCH --output=testingargs__output_%A.out
#SBATCH --error=testing_args_error_%A.out

module purge
module load 2022
module load Anaconda3/2022.05
module load cuDNN/8.6.0.163-CUDA-11.8.0

source ~/.bashrc
cd $HOME/LLaMA3-Quantization
conda activate llama

wbits=4

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

gptq70B="LLaMA-3-70B-GPTQ-4bit-b128" # GPTQ for LLama 3 70B

irqlora="LLaMA-3-8B-IR-QLoRA" # IR-QLoRA

quantization_model=${gptq}

tasks_commonsenseQA="piqa,arc_easy,arc_challenge,hellaswag,winogrande"
tasks=${tasks_commonsenseQA}

python3 main.py --model ${model} --peft "Efficient-ML/"${quantization_model} --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/${quantization_model} --wbits ${wbits} --tasks ${tasks}
