# LLaMA3-Quantization

This is the fork containing the code for the reproducibility study of How Good Are Low-bit Quantized LLAMA3 Models?
An Empirical Study [[PDF](https://arxiv.org/abs/2404.14047)].

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
After the script finished running, a folder named `quantized-llama-hqq-Meta-Llama-3-8B-xbit` should be visible.

### GPTQ
...

## Evaluating the quantized models
After creating the models, they can be evaluated. Follow the corresponding instructions per section.

### Reproducibility Evaluation
...

### Cross-encoder Evaluation
This only used the original, full bit LLaMA3-8B model and the HQQ versions.
Run the following:
```
sbatch run_beir.sh
```

### Bi-encoder Evaluation
This only used the original, full bit LLaMA3-8B model and the HQQ versions.
Run the following:
```
sbatch run_mteb_sts.sh
```


## Note
We built on top of the already existing code. However, files that we did not need to modify are not edited to contain more comments or so. This was due to the challenges we ran into. The code created by us is documented, though. 
