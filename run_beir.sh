#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=Beir
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
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
# python3 benchmark_beir.py --model meta-llama/Meta-Llama-3-8B --peft Efficient-ML/LLaMA-3-8B-IR-QLoRA --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/beir/llama-3-8b-irqlora --wbits 4 #--batch_size 128 
# python3 benchmark_beir.py --model "Efficient-ML/LLaMA-3-8B-SmoothQuant-8bit-8bit" --quant_method gptq --epochs 0 --output_dir ./log/beir/LLaMA-3-8B-SmoothQuant-8bit-8bit --wbits 8 --abits 8 --ce --be
# python3 benchmark_beir.py --model "Efficient-ML/LLaMA-3-8B-AWQ-4bit-b128" --quant_method gptq --epochs 1 --output_dir ./log/beir/LLaMA-3-8B-AWQ-4bit-b128-epoch1 --wbits 4 --group_size 128 --lwc --net "llama-7b" --be
# python3 benchmark_beir.py --model "Efficient-ML/LLaMA-3-8B-AWQ-4bit-b128" --quant_method gptq --epochs 1 --output_dir ./log/beir/LLaMA-3-8B-AWQ-4bit-b128-epoch1 --wbits 4 --group_size 128 --lwc --net "llama-7b" --ce
# python3 benchmark_beir.py --model "Efficient-ML/LLaMA-3-8B-AWQ-4bit-b128" --quant_method gptq --epochs 0 --output_dir ./log/beir/LLaMA-3-8B-AWQ-4bit-b128-epoch0 --wbits 4 --group_size 128 --ce --be

# python3 benchmark_beir.py --model meta-llama/Meta-Llama-3-8B --quant_method None --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/beir/llama-3-8b-irqlora --ce --be #--wbits 4 #--batch_size 128 
# python3 benchmark_beir.py --model meta-llama/Meta-Llama-3-8B --quant_method None --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/beir/llama-3-8b-irqlora --ce #--wbits 4 #--batch_size 128 
# python3 benchmark_beir.py --model meta-llama/Meta-Llama-3-8B --quant_method None --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/beir/llama-3-8b-upr --upr #--wbits 4 #--batch_size 128 
# python3 benchmark_beir.py --model meta-llama/Meta-Llama-3-8B --quant_method None --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/beir/llama-3-8b-upr-hotpotqa --upr --beirdata hotpotqa #--wbits 4 #--batch_size 128 
# python3 benchmark_beir.py --model meta-llama/Meta-Llama-3-8B --quant_method None --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/beir/test-batchsize1-uprdefaultbase --topk 10 --upr #--wbits 4 #--batch_size 128 



# python3 benchmark_beir.py --model meta-llama/Meta-Llama-3-8B-Instruct --quant_method None --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/beir/LLaMA38B-16bit-upr --topk 100 --upr
# python3 benchmark_beir.py --model meta-llama/Meta-Llama-3-8B --quant_method None --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/beir/LLaMA38B-16bit-upr --topk 100 --upr 
# python3 benchmark_beir.py --model bigscience/T0_3B --quant_method None --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/beir/T03B-16bit-upr --topk 100 --upr #--seqlen 512




#### HQQ & scifact ####
# python benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-4bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/Meta-Llama-3-8B_4bits --wbits 4 --abits 4 --group_size 128 --upr 
# python benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-3bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/Meta-Llama-3-8B_3bits --wbits 3 --abits 3 --group_size 128 --upr 
# python benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-2bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/Meta-Llama-3-8B_2bits --wbits 2 --abits 2 --group_size 128 --upr
# python3 benchmark_beir.py --model meta-llama/Meta-Llama-3-8B --quant_method None --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/beir/LLaMA38B-16bit-upr --upr 

#### HQQ & climate-fever ####
# python benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-4bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/Meta-Llama-3-8B_4bits --wbits 4 --abits 4 --group_size 128 --upr --beirdata climate-fever &&
# python benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-3bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/Meta-Llama-3-8B_3bits --wbits 3 --abits 3 --group_size 128 --upr --beirdata climate-fever &&
# python3 benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-2bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --batch_size 1 --output_dir ./log/Meta-Llama-3-8B_2bits --wbits 2 --abits 2 --group_size 128 --upr --beirdata climate-fever &&
# python3 benchmark_beir.py --model meta-llama/Meta-Llama-3-8B --quant_method None --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --batch_size 1 --output_dir ./log/beir/LLaMA38B-16bit-upr --upr --beirdata climate-fever

#### HQQ & quora #### #TODO change this dataset, takes more than 20 hours for one method... could do arguana, trec-covid or nfcorpus instead
# python3 benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-4bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/Meta-Llama-3-8B_4bits --wbits 4 --abits 4 --group_size 128 --upr --beirdata quora &&
# python3 benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-3bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/Meta-Llama-3-8B_3bits --wbits 3 --abits 3 --group_size 128 --upr --beirdata quora &&
# python3 benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-2bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/Meta-Llama-3-8B_2bits --wbits 2 --abits 2 --group_size 128 --upr --beirdata quora &&
# python3 benchmark_beir.py --model meta-llama/Meta-Llama-3-8B --quant_method None --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/beir/LLaMA38B-16bit-upr --upr --beirdata quora

python3 benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-4bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/Meta-Llama-3-8B_4bits --wbits 4 --abits 4 --group_size 128 --upr --beirdata trec-covid &&
python3 benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-3bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/Meta-Llama-3-8B_3bits --wbits 3 --abits 3 --group_size 128 --upr --beirdata trec-covid &&
python3 benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-2bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/Meta-Llama-3-8B_2bits --wbits 2 --abits 2 --group_size 128 --upr --beirdata trec-covid &&
python3 benchmark_beir.py --model meta-llama/Meta-Llama-3-8B --quant_method None --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/beir/LLaMA38B-16bit-upr --upr --beirdata trec-covid

#### HQQ & fiqa ####
# python3 benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-4bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/Meta-Llama-3-8B_4bits --wbits 4 --abits 4 --group_size 128 --upr --beirdata fiqa &&
# python3 benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-3bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/Meta-Llama-3-8B_3bits --wbits 3 --abits 3 --group_size 128 --upr --beirdata fiqa &&
# python3 benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-2bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/Meta-Llama-3-8B_2bits --wbits 2 --abits 2 --group_size 128 --upr --beirdata fiqa &&
# python3 benchmark_beir.py --model meta-llama/Meta-Llama-3-8B --quant_method None --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/beir/LLaMA38B-16bit-upr --upr --beirdata fiqa

#### HQQ & webis-touche2020 ####
# python benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-4bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/Meta-Llama-3-8B_4bits --wbits 4 --abits 4 --group_size 128 --upr --beirdata webis-touche2020 &&
# python benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-3bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/Meta-Llama-3-8B_3bits --wbits 3 --abits 3 --group_size 128 --upr --beirdata webis-touche2020 &&
# python benchmark_beir.py --model ./quantized-llama-hqq-Meta-Llama-3-8B-2bit --quant_method hqq --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/Meta-Llama-3-8B_2bits --wbits 2 --abits 2 --group_size 128 --upr --beirdata webis-touche2020 &&
# python3 benchmark_beir.py --model meta-llama/Meta-Llama-3-8B --quant_method None --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/beir/LLaMA38B-16bit-upr --upr --beirdata webis-touche2020

