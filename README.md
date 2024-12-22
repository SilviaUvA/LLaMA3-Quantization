# LLaMA3-Quantization For Information Retrieval

This is the fork containing the code for the reproducibility study of How Good Are Low-bit Quantized LLAMA3 Models?
An Empirical Study [[PDF](https://arxiv.org/abs/2404.14047)]. It also extends the original paper by expanding it to the Information Retrieval domain.

## Setting Up
To clone the fork, run:
```
git clone https://github.com/SilviaUvA/LLaMA3-Quantization.git
cd LLaMA3-Quantization
```
Everything was run with scripts on SURF's Snellius. To install the environment, run the following:
```shell
sbatch install_env.sh
```
This will create the `llama` environment, which takes roughly 30 minutes to an hour. 
Now to login to Hugging Face, create a login token with [your Hugging Face account](https://huggingface.co/docs/hub/security-tokens) and paste the token into `huggingface_access_token.txt`.

## Creating the quantized models
This section discusses how to obtain the HQQ and GPTQ models.

### HQQ
First create the HQQ models. Within `run_quantize.sh` change `--bits x` to the required number of bits (default script is 2). Then run:
```
sbatch run_quantize.sh
```
After the script finished running, a folder named `quantized-llama-hqq-Meta-Llama-3-8B-xbit` should be visible, which contains the quantized model.

### GPTQ
...

## Evaluating the quantized models
After creating the models, they can be evaluated. Follow the corresponding instructions per section depending on what should be run.

### Reproducibility Evaluation
...

### Cross-encoder Evaluation
This only used the original, full bit LLaMA3-8B model and the HQQ versions.
Modify the commandline arguments in `run_beir.sh` depending on what model it should be run for. If trying to run the full model, change it to:
```
model=${"meta-llama/Meta-Llama-3-8B"}
quantmethod=${"None"} 
beirdata=${"trec-covid"} # choose from: "trec-covid", "fiqa", "scifact", "climate-fever" and "webis-touche2020"

python3 benchmark_beir.py --model ${model} --quant_method ${quantmethod} --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/beir/${model} --upr --beirdata ${beirdata}
```

If trying to run on a quantized model with _x_ bits, instead change it to:
```
nbits=${x} # change to number of bits, must align with model name
model=${"./quantized-llama-hqq-Meta-Llama-3-8B-xbit"}
quantmethod=${"hqq"} 
beirdata=${"trec-covid"} # choose from: "trec-covid", "fiqa", "scifact", "climate-fever" and "webis-touche2020"

python3 benchmark_beir.py --model ${model} --quant_method ${quantmethod} --let --tau_range 0.1 --tau_n 100 --blocksize 256 --epochs 0 --output_dir ./log/${model} --wbits ${nbits} --abits ${nbits} --group_size 128 --upr --beirdata ${beirdata}
```

Then run the following:
```
sbatch run_beir.sh
```

### Bi-encoder Evaluation
This only used the original, full bit LLaMA3-8B model and the HQQ versions.
Modify the commandline arguments in `run_mteb_sts.sh` depending on what model it should be run for. If trying to run the full model, change it to:
```
python3 benchmark_mteb.py \
    --model meta-llama/Meta-Llama-3-8B \
    --quant_method None \
    --tau_range 0.1 \
    --tau_n 100 \
    --blocksize2 256 \
    --epochs 0 \
    --output_dir ./mteb_sts_results_baseline \
    --batch_size 32 \
    --sts_tasks $TASKS_STR \
    --languages eng
```

If trying to run on a quantized model with _x_ bits, instead change it to:
```
python3 benchmark_mteb.py \
    --model quantized-llama-hqq-Meta-Llama-3-8B-xbit \ # modify bits here
    --quant_method hqq \
    --tau_range 0.1 \
    --tau_n 100 \
    --blocksize2 256 \
    --epochs 0 \
    --output_dir ./mteb_sts_results_hqq_xbit \ # modify bits here
    --batch_size 32 \
    --sts_tasks $TASKS_STR \
    --languages eng
```
Please ensure `--output_dir` does not exist already. 
Then run the following:
```
sbatch run_mteb_sts.sh
```


## Note
We built on top of the already existing code. However, files that we did not need to modify are not edited to contain more comments or so. This was due to the challenges we ran into. The code created by us is documented, though. 
