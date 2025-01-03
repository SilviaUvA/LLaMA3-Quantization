#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=Env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=install_env_%A.out

module purge
module load 2022
module load Anaconda3/2022.05

source ~/.bashrc
cd $HOME/LLaMA3-Quantization

conda env create -f environment.yml
