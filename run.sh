#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=Run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --output=run_%A.out
#SBATCH --error=error_%A.out

module purge
module load 2022
module load Anaconda3/2022.05
module load cuDNN/8.6.0.163-CUDA-11.8.0

source ~/.bashrc
cd $HOME/LLaMA3-Quantization
#conda init bash 
conda activate llama

# python3 main.py --eval_ppl --model meta-llama/Meta-Llama-3-8B --peft Efficient-ML/LLaMA-3-8B-IR-QLoRA --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/llama-3-8b-irqlora-mmlu --wbits 4 --tasks "hendrycksTest-*" #winogrande
# python3 main.py --model meta-llama/Meta-Llama-3-8B --peft Efficient-ML/LLaMA-3-8B-IR-QLoRA --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/llama-3-8b-irqlora-mmlu --wbits 4 --tasks "hendrycksTest-*" #piqa,arc_easy,arc_challenge,hellaswag,winogrande
python3 main.py --model "Efficient-ML/LLaMA-3-8B-SmoothQuant-8bit-8bit" --quant_method gptq --eval_ppl --epochs 0 --output_dir ./log/LLaMA-3-8B-SmoothQuant-8bit-8bit --wbits 4
