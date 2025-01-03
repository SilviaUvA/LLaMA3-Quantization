#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=autogptq_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=06:00:00
#SBATCH --output=autogptq_eval_output_%A.out
#SBATCH --error=autogptq_eval_error_%A.out

module purge
module load 2022
module load Anaconda3/2022.05
module load cuDNN/8.6.0.163-CUDA-11.8.0

source ~/.bashrc
cd $HOME/LLaMA3-Quantization
conda activate llama

tasks_commonsenseQA="piqa,arc_easy,arc_challenge,hellaswag,winogrande"


llama3_8b="Meta-Llama-3-8B"  # Llama 3 8B
llama2_7b="Llama-2-7b-hf"  # Llama 2 7B

cur_llama=${llama3_8b}

wbits=4

model_name="autogptq-${cur_llama}-${wbits}bit-128g"

quantized_model="./quantized_models/${model_name}"

# Evaluating GPTQ model from running code using AutoGPTQ

python3 main.py --model ${quantized_model} --quant_method gptq  --wbits ${wbits} --epochs 0 --eval_ppl --output_dir ./log/${model_name} --lwc --net "llama-7b" --group_size 128 --tasks ${tasks_commonsenseQA}


echo "Done for ${cur_llama} ${wbits}bit 128g"
