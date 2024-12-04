#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=gptq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=gptq_output_%A.out
#SBATCH --error=gptq_error_%A.out

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



gptq_custom_4bit="../GPTQ-for-LLaMa/llama8b-4bit-128g"
gptq_custom_2bit="../GPTQ-for-LLaMa/llama8b-2bit-128g"

tasks_commonsenseQA="piqa,arc_easy,arc_challenge,hellaswag,winogrande"


# Evaluating GPTQ model from running code from GPTQ-for-LLaMa repo
 python3 main.py --model ${gptq_custom_4bit} --quant_method gptq --eval_ppl --epochs 0 --output_dir ./log/gptq --wbits 4  --lwc --net "llama-7b" --group_size 128 --model_type LlamaForCausalLM --tasks ${tasks_commonsenseQA}
