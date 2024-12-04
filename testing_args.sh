#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=testing_args
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=args_testing_output_%A.out
#SBATCH --error=args_testing_error_%A.out

module purge
module load 2022
module load Anaconda3/2022.05
module load cuDNN/8.6.0.163-CUDA-11.8.0

source ~/.bashrc
cd $HOME/LLaMA3-Quantization
pip uninstall transformers
pip install transformers==4.37.2
conda activate llama
pip install protobuf==3.20.2

wbits=4


llama2_7b="meta-llama/Llama-2-7b-hf"  # Llama 2 7B

llama3_8b="meta-llama/Meta-Llama-3-8B"  # Llama 3 8B
llama3_70b="meta-llama/Meta-Llama-3-70B"  # Llama 3 70B
llava_next_8b="lmms-lab/llama3-llava-next-8b" # Llava Next 8B
model=${llama3_8b}

gptq_custom_4bit="../GPTQ-for-LLaMa/llama8b-4bit-128g"
gptq_custom_2bit="../GPTQ-for-LLaMa/llama8b-2bit-128g"

gptq="LLaMA-3-8B-GPTQ-4bit-b128" # GPTQ
awq="LLaMA-3-8B-AWQ-4bit-b128" # AWQ
quip="LLaMA-3-8B-QuIP-2bit" # QuIP
dbllm="LLaMA-3-8B-DB-LLM-2bit-fake" # DB-LLM
pbllm="LLaMA-3-8B-PB-LLM-1.7bit-fake" # PB-LLM
billm="LLaMA-3-8B-BiLLM-1.1bit-fake " # BiLLM
smoothquant4="LLaMA-3-8B-SmoothQuant-4bit-4bit" # SmoothQuant 4bit
smoothquant8="LLaMA-3-8B-SmoothQuant-8bit-8bit" # SmoothQuant 8bit

quantization_model=${awq}

gptq70B="LLaMA-3-70B-GPTQ-4bit-b128" # GPTQ for LLama 3 70B

irqlora="LLaMA-3-8B-IR-QLoRA" # IR-QLoRA


tasks_commonsenseQA="piqa,arc_easy,arc_challenge,hellaswag,winogrande"
tasks=${tasks_commonsenseQA}

# LLaMa 2
python3 main.py --model ${llama2_7b} --quant_method gptq --eval_ppl --epochs 1 --output_dir ./log/${llama2_7b} --wbits 4 --lwc --net "llama-7b" --group_size 128  --model_type LlamaForCausalLM


# Evaluating GPTQ model from running code from GPTQ-for-LLaMa repo
#  python3 main.py --model ${gptq_custom_4bit} --quant_method gptq --eval_ppl --epochs 0 --output_dir ./log/gptq --wbits 4  --lwc --net "llama-7b" --group_size 128 --model_type LlamaForCausalLM

# Evaluating IR-QLoRA
# python3 main.py --model ${model} --peft "Efficient-ML/"${irqlora} --quant_method irqlora --tau_range 0.1 --tau_n 100 --blocksize2 256 --epochs 0 --output_dir ./log/${irqlora} --wbits ${wbits} --abits ${wbits} --tasks ${tasks}

# Quantization with 1 epoch with model from Efficient-ML's huggingface
# python3 main.py --model "Efficient-ML/"${quantization_model} --quant_method gptq --eval_ppl --epochs 1 --output_dir ./log/${quantization_model} --wbits 8 --lwc --net "llama-7b" --group_size 128
