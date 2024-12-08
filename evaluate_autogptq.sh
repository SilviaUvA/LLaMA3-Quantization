#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=autogptq_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=autogptq_eval_output_%A.out
#SBATCH --error=autogptq_eval_error_%A.out

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

pip install toml
pip install triton==2.0.0

pip install optimum==1.23.3

tasks_commonsenseQA="piqa,arc_easy,arc_challenge,hellaswag,winogrande"

llama3_8b="meta-llama/Meta-Llama-3-8B"  # Llama 3 8B

wbits=4

quantized_model=" ./quantized_models/autogptq-llama-3-8b-${wbits}bit-128g"

# Evaluating GPTQ model from running code using AutoGPTQ

python3 main.py --model ${quantized_model} --quant_method gptq  --wbits ${wbits} --epochs 0 --eval_ppl --output_dir ./log/${quantized_model} --lwc --net "llama-7b" --group_size 128 --tasks ${tasks_commonsenseQA}


echo "Done for llama3-8B ${wbits}bit 128g"
