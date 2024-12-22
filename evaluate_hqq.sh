#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=hqq_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=hqq_eval_output_%A.out
#SBATCH --error=hqq_eval_error_%A.out

module load 2024

conda activate llama

bits=4
hqq=quantized-llama-hqq-Meta-Llama-3-8B-${bits}bit

python main.py --model ${pwd}"./"${hqq} --quant_method hqq --eval_ppl --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/${hqq} --wbits ${bits} --abits ${bits} --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande
