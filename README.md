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

## Per Experiment Commands
Depending on what experiments you want to run, follow the remaining instructions in the corresponding section.

### GPTQ Experiments
...

### HQQ Experiments
...

### Cross-encoder Experiments
...

### Bi-encoder Experiments
...

## Note
We built on top of the already existing code. However, files that we did not need to modify are not edited to contain more comments or so. This was due to the challenges we ran into. The code created by us is documented, though. 

## Related Project

[QUIP](https://github.com/Cornell-RelaxML/QuIP)

[GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://github.com/IST-DASLab/gptq)

[AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)

[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://github.com/mit-han-lab/llm-awq)

[RPTQ: Reorder-Based Post-Training Quantization for Large Language Models](https://github.com/hahnyuan/RPTQ4LLM)

[OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://github.com/OpenGVLab/OmniQuant)

[PB-LLM: Partially Binarized Large Language Models](https://github.com/hahnyuan/PB-LLM)

[BiLLM: Pushing the Limit of Post-Training Quantization for LLMs](https://github.com/Aaronhuang-778/BiLLM)

[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://github.com/mit-han-lab/smoothquant)

[QLoRA: Efficient Finetuning of Quantized LLMs](https://github.com/artidoro/qlora)

[IR-QLoRA: Accurate LoRA-Finetuning Quantization of LLMs via Information Retention](https://github.com/htqin/IR-QLoRA)


