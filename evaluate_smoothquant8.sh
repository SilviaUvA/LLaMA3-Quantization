#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=sq8_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=sq8_eval_output_%A.out
#SBATCH --error=sq8_eval_error_%A.out

module purge
module load 2022
module load Anaconda3/2022.05
module load cuDNN/8.6.0.163-CUDA-11.8.0

source ~/.bashrc
cd $HOME/LLaMA3-Quantization
conda activate llama

smoothquant8="LLaMA-3-8B-SmoothQuant-8bit-8bit"

tasks_commonsenseQA="piqa,arc_easy,arc_challenge,hellaswag,winogrande"

# Evaluate with quantized model from Efficient-ML's huggingface

# SmoothQuant 8-bit
python3 main.py --model "Efficient-ML/"${smoothquant8} --quant_method smoothquant8  --wbits 8 --epochs 0 --eval_ppl --output_dir ./log/${smoothquant8} --lwc --net "llama-7b" --group_size 128 --tasks ${tasks_commonsenseQA}
