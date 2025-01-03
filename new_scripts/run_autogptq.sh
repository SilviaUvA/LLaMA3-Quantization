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

llama3_8b="Meta-Llama-3-8B"  # Llama 3 8B
llama2_7b="Llama-2-7b-hf"  # Llama 2 7B

cur_llama=${llama3_8b}
llama_model="meta-llama/${cur_llama}"


wbits=4

model_name="autogptq-${cur_llama}-${wbits}bit-128g"
quantized_model=" ./quantized_models/${model_name}"

# Quantizing Llama model from running code using AutoGPTQ
CUDA_VISIBLE_DEVICES=0 python3 autogptq.py --model ${llama_model} --save_dir ${quantized_model} --output_dir ./log/${model_name} --wbits ${wbits} --group_size 128

echo "Done for ${cur_llama} ${wbits}bit 128g"
