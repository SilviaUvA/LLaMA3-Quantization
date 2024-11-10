#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=Run_eval_fake_ptq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00
#SBATCH --output=run_eval_fake_ptq_%A.out
#SBATCH --error=error_eval_fake_ptq_%A.out

module purge
module load 2022
module load Anaconda3/2022.05
module load cuDNN/8.6.0.163-CUDA-11.8.0

source ~/.bashrc
cd $HOME/LLaMA3-Quantization
#conda init bash 
conda activate llama

model_path='Efficient-ML/LLaMA-3-8B-BiLLM-1.1bit-fake'
python main.py --model ${model_path} --epochs 0 --output_dir ./log/--tasks 'hellaswag,piqa,winogrande,arc_easy,arc_challenge' --wbits 16 --abits 16 --eval_ppl --multigpu
