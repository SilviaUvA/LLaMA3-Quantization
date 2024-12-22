#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=sq4_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=sq4_eval_output_%A.out
#SBATCH --error=sq4_eval_error_%A.out

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


smoothquant4="LLaMA-3-8B-SmoothQuant-4bit-4bit"

tasks_commonsenseQA="piqa,arc_easy,arc_challenge,hellaswag,winogrande"

# Evaluate with quantized model from Efficient-ML's huggingface

# SmoothQuant 4-bit
python3 main.py --model "Efficient-ML/"${smoothquant4} --quant_method smoothquant4  --wbits 4 --epochs 0 --eval_ppl --output_dir ./log/${smoothquant4} --lwc --net "llama-7b" --group_size 128 --tasks ${tasks_commonsenseQA}
