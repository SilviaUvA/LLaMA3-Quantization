#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=llama3_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=model_llama3_eval_output_%A.out
#SBATCH --error=model_llama3_eval_error_%A.out

module purge
module load 2022
module load Anaconda3/2022.05
module load cuDNN/8.6.0.163-CUDA-11.8.0

source ~/.bashrc
cd $HOME/LLaMA3-Quantization

conda activate llama

pip uninstall transformers
pip install transformers==4.37.0
pip install protobuf==3.20.2

llama3_8b="meta-llama/Meta-Llama-3-8B"  # Llama 3 8B
llama3_70b="meta-llama/Meta-Llama-3-70B"  # Llama 3 70B
llava_next_8b="lmms-lab/llama3-llava-next-8b" # Llava Next 8B
model=${llama3_8b}


tasks_commonsenseQA="piqa,arc_easy,arc_challenge,hellaswag,winogrande"

# Evaluate original Llama 3 model
python3 main.py --model ${model} --quant_method none  --wbits 4 --epochs 0 --eval_ppl --output_dir ./log/${model} --lwc --net "llama-7b" --group_size 128 --tasks ${tasks_commonsenseQA}
