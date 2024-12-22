#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=MTEB-STS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --output=run_mteb_sts_%A_hqq_2bit.out
#SBATCH --error=run_mteb_sts_error_%A_hqq_2bit.out

module purge
module load 2022
module load Anaconda3/2022.05
module load cuDNN/8.6.0.163-CUDA-11.8.0

source ~/.bashrc
cd $HOME/LLaMA3-Quantization
conda activate llama

# Define STS tasks
STS_TASKS=(
    "STS12"
    "STS13"
    "STS14"
    "STS15"
    "STS16"
    "STSBenchmark"
    "SICK-R"
)

# Convert array to space-separated string
TASKS_STR=$(IFS=' ' ; echo "${STS_TASKS[*]}")

# Run evaluation
#baseline
# python3 benchmark_mteb.py \
#     --model meta-llama/Meta-Llama-3-8B \
#     --quant_method None \
#     --tau_range 0.1 \
#     --tau_n 100 \
#     --blocksize2 256 \
#     --epochs 0 \
#     --output_dir ./mteb_sts_results_baseline \
#     --batch_size 32 \
#     --sts_tasks $TASKS_STR \
#     --languages eng

# hqq-quantized-8b-3bit
python3 benchmark_mteb.py \
    --model quantized-llama-hqq-Meta-Llama-3-8B-2bit \
    --quant_method hqq \
    --tau_range 0.1 \
    --tau_n 100 \
    --blocksize2 256 \
    --epochs 0 \
    --output_dir ./mteb_sts_results_hqq_2bit \
    --batch_size 32 \
    --sts_tasks $TASKS_STR \
    --languages eng