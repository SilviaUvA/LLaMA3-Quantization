#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=llama2_quantization
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=qllama2_output_%A.out
#SBATCH --error=qllama2_error_%A.out

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


llama2_7b="meta-llama/Llama-2-7b-hf"  # Llama 2 7B

tasks_commonsenseQA="piqa,arc_easy,arc_challenge,hellaswag,winogrande"

# LLaMa 2
python3 main.py --model ${llama2_7b} --quant_method gptq --eval_ppl --epochs 1 --output_dir ./log/${llama2_7b} --wbits 4 --lwc --net "llama-7b" --group_size 128  --model_type LlamaForCausalLM --tokenizer_class LlamaTokenizer --tasks ${tasks_commonsenseQA}
