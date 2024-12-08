#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=autogptq_run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=autogptq_run_output_%A.out
#SBATCH --error=autogptq_run_error_%A.out

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

pip install toml
pip install triton==2.0.0

pip install optimum==1.23.3

# pt_file="../GPTQ-for-LLaMa/tensors/llama-3-8b-4bit-128g/llama-3-8b-4bit-128g.pt"

llama3_8b="meta-llama/Meta-Llama-3-8B"  # Llama 3 8B

quantized_model=" ./quantized_models/autogptq-llama-3-8b-4bit-128g"

# Quantizing Llama model from running code using AutoGPTQ
CUDA_VISIBLE_DEVICES=0 python3 autogptq.py --model ${llama3_8b} --save_dir ${quantized_model} --output_dir ./log/gptq --wbits 4 --group_size 128

echo "Done for llama3-8B 4bit 128g"