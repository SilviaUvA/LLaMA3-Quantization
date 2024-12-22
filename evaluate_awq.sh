#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=awq_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=awq_eval_output_%A.out
#SBATCH --error=awq_eval_error_%A.out

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


awq="LLaMA-3-8B-AWQ-4bit-b128"

tasks_commonsenseQA="piqa,arc_easy,arc_challenge,hellaswag,winogrande"

# Evaluate with quantized model from Efficient-ML's huggingface

# AWQ
python3 main.py --model "Efficient-ML/"${awq} --quant_method awq  --wbits 4 --epochs 0 --eval_ppl --output_dir ./log/${awq} --lwc --net "llama-7b" --group_size 128 --tasks ${tasks_commonsenseQA}
