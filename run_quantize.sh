#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=hhq_quantize8b_3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=hhq_quantize8b_3.out

module purge
module load 2022
module load Anaconda3/2022.05
module load cuDNN/8.6.0.163-CUDA-11.8.0

source ~/.bashrc
cd $HOME/LLaMA3-Quantization
#conda init bash 
conda activate llama

python quantizehqq.py --model "meta-llama/Meta-Llama-3-8B" --bits 3 --group_size 128