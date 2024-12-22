#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=hqq_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=hqq_eval_output_%A.out
#SBATCH --error=hqq_eval_error_%A.out

module purge
module load 2022
module load Anaconda3/2022.05
module load cuDNN/8.6.0.163-CUDA-11.8.0

source ~/.bashrc
cd $HOME/LLaMA3-Quantization
conda activate llama

bits=4

llama2="Llama-2-7b-hf"
llama3="Meta-Llama-3-8B"
cur_llama=${llama3}

hqq=quantized-llama-hqq-${cur_llama}-${bits}bit

python main.py --model ${pwd}"./"${hqq} --quant_method hqq --eval_ppl --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/${hqq} --wbits ${bits} --abits ${bits} --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande

echo "Done for ${cur_llama} ${bits}bit"
