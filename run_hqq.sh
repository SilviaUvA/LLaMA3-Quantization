#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=hqq_quantize
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=hqq_quantize.out

module purge
module load 2022
module load Anaconda3/2022.05
module load cuDNN/8.6.0.163-CUDA-11.8.0

source ~/.bashrc
cd $HOME/LLaMA3-Quantization
#conda init bash
conda activate llama

bits=4

llama3_8b="Meta-Llama-3-8B"  # Llama 3 8B
llama2_7b="Llama-2-7b-hf"  # Llama 2 7B

cur_llama=${llama3_8b}
llama_model="meta-llama/${cur_llama}"


python quantizehqq.py --model ${cur_llama} --bits ${bits} --group_size 128


echo "Done for ${cur_llama} ${bits}bit 128g"
