#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=Beir
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=runBeir_%A.out
#SBATCH --error=errorBeir_%A.out

module purge
module load 2022
module load Anaconda3/2022.05
module load cuDNN/8.6.0.163-CUDA-11.8.0

source ~/.bashrc
cd $HOME/LLaMA3-Quantization
#conda init bash 
conda activate llama

# installing Elasticsearch for BEIR
ES_VERSION="7.9.1"
ES_DIR="elasticsearch-$ES_VERSION"
ES_TAR="$ES_DIR-linux-x86_64.tar.gz"
ES_DOWNLOAD_URL="https://artifacts.elastic.co/downloads/elasticsearch/$ES_TAR"

# check if downloaded
if [ ! -d "$ES_DIR" ]; then
    echo "Downloading Elasticsearch $ES_VERSION..."
    wget $ES_DOWNLOAD_URL -O $ES_TAR

    echo "Extracting Elasticsearch..."
    tar -xzf $ES_TAR
    rm $ES_TAR
fi

# data and logs dirs for Elasticsearch
ES_DATA_DIR="$ES_DIR/data"
ES_LOGS_DIR="$ES_DIR/logs"
mkdir -p $ES_DATA_DIR
mkdir -p $ES_LOGS_DIR

# start Elasticsearch server
echo "Starting Elasticsearch..."
./$ES_DIR/bin/elasticsearch > $ES_LOGS_DIR/elasticsearch.log 2>&1 &
ES_PID=$!

cleanup() {
    echo "Stopping Elasticsearch..."
    kill $ES_PID
}

# kill Elasticsearch server after finishing job just to be sure
trap cleanup EXIT

# run actual python file
# python3 benchmark_beir.py --model meta-llama/Meta-Llama-3-8B --peft Efficient-ML/LLaMA-3-8B-IR-QLoRA --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/beir/llama-3-8b-irqlora --wbits 4 #--batch_size 128 #--tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande
python3 benchmark_beir.py --model "Efficient-ML/LLaMA-3-8B-SmoothQuant-8bit-8bit" --quant_method gptq --eval_ppl --epochs 0 --output_dir ./log/beir/LLaMA-3-8B-SmoothQuant-8bit-8bit --wbits 8 --abits 8 --ce --be
