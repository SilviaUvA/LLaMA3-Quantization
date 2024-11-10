import os
import torch
import sys

#TODO should incorporate these into environment later (beir & elasticsearch) -> Successfully installed Pillow-11.0.0 beir-2.0.0 elasticsearch-7.9.1 (this one goes via sh file) faiss_cpu-1.9.0 pytrec_eval-0.5 sentence-transformers-3.2.1
try:
    from beir import util, LoggingHandler
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.search.lexical import BM25Search as BM25
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "beir"])
    from beir import util, LoggingHandler
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.search.lexical import BM25Search as BM25

import random
import numpy as np
from models.LMClass import LMClass
from models.IRQLoRALMClass import IRQLoRALMClass
import time
from datautils import get_loaders
from lm_eval import evaluator
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quant.omniquant import omniquant
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories

from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quant.int_linear import QuantLinear

import pdb

from huggingface_hub import login

with open("huggingface_access_token.txt") as f:
    access_token = f.readline().strip()

login(token=access_token)

torch.backends.cudnn.benchmark = True
print("Using device: ", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "llava-llama-2-13b-chat-lightning-preview",
    "falcon-180b",
    "falcon-7b",
    "mixtral-8x7b"
]

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="../log/", type=str, help="direction of logging file")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--real_quant", default=False, action="store_true", help="real quantization, which can see memory reduce. Note that due to the limitations of AutoGPTQ kernels, the real quantization of weight-only quantization can only lead memory reduction, but with slower inference speed.")
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix","pile"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_lr", type=float, default=5e-3)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--let",default=False, action="store_true",help="activate learnable equivalent transformation")
    parser.add_argument("--lwc",default=False, action="store_true",help="activate learnable weight clipping")
    parser.add_argument("--aug_loss", default=False, action="store_true", help="calculate additional loss with same input")
    parser.add_argument("--symmetric",default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--disable_zero_point",default=False, action="store_true", help="quantization without zero_point")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--multigpu", action="store_true", help="at eval, map model to multiple gpus")
    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)

    parser.add_argument("--quant_method", type=str, default='irqlora')
    parser.add_argument("--peft", type=str, default='./')
    parser.add_argument("--tau_range", type=float, default=0.1)
    parser.add_argument("--tau_n", type=int, default=100)
    parser.add_argument("--blocksize2", type=int, default=256)

    args = parser.parse_args()
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    # # check
    # if args.epochs > 0:
    #     assert args.lwc or args.let
        
    # if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
    #     args.deactive_amp = True

    # # init logger
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # if args.cache_dir:
    #     Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    # if args.save_dir:
    #     Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    # output_dir = Path(args.output_dir)
    # logger = utils.create_logger(output_dir)
    # logger.info(args)
    
    # # load model
    # if args.net is None:
    #     args.net = args.model.split('/')[-1]
    # # assert args.net in net_choices
    # args.model_family = args.net.split('-')[0]
    # if args.quant_method in ['irqlora', 'qlora']:
    #     lm = IRQLoRALMClass(args)
    # else:
    #     lm = LMClass(args)

    # lm.seqlen = 2048
    # lm.model.eval()
    # for param in lm.model.parameters():
    #     param.requires_grad = False

    

    # args.weight_quant_params = {
    #     "n_bits": args.wbits,
    #     "per_channel_axes": [0],
    #     "symmetric": args.symmetric,
    #     "dynamic_method": args.w_dynamic_method,
    #     "group_size": args.group_size,
    #     "lwc":args.lwc,
    #     "disable_zero_point": args.disable_zero_point
    # }
    # args.act_quant_params = {
    #     "n_bits":  args.abits,
    #     "per_channel_axes": [],
    #     "symmetric": False,
    #     "dynamic_method": args.a_dynamic_method,
    # }
    # args.q_quant_params = {
    #     "n_bits": args.abits,
    #     "per_channel_axes": [],
    #     "symmetric": False,
    #     "dynamic_method": args.a_dynamic_method,
    # }
    # args.k_quant_params = {
    #     "n_bits": args.abits,
    #     "per_channel_axes": [],
    #     "symmetric": False,
    #     "dynamic_method": args.a_dynamic_method,
    # }
    # args.v_quant_params = {
    #     "n_bits": args.abits,
    #     "per_channel_axes": [],
    #     "symmetric": False,
    #     "dynamic_method": args.a_dynamic_method,
    # }
    # args.p_quant_params = {
    #     "n_bits": 16,
    #     "metric": "fix0to1",
    # }

    # if args.multigpu:
    #     gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
    #     lm._device = f"cuda:{gpu_id}"
    #     logger.info(f"set quantization in gpu {gpu_id}")

    # # act scales and shifts
    # if args.act_scales is None:
    #     args.act_scales = f'./act_scales/{args.net}.pt'
    # if args.act_shifts is None:
    #     args.act_shifts = f'./act_shifts/{args.net}.pt'

    # # quantization
    # if (args.wbits < 16 or args.abits < 16) and (args.epochs > 0):
    #     logger.info("=== start quantization ===")
    #     tick = time.time()     
    #     # load calibration dataset
    #     cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
    #     if os.path.exists(cache_dataloader):
    #         dataloader = torch.load(cache_dataloader)
    #         logger.info(f"load calibration from {cache_dataloader}")
    #     else:
    #         dataloader, _ = get_loaders(
    #             args.calib_dataset,
    #             nsamples=args.nsamples,
    #             seed=args.seed,
    #             model=args.model,
    #             seqlen=lm.seqlen,
    #         )
    #         torch.save(dataloader, cache_dataloader)    
    #     act_scales = None
    #     act_shifts = None
    #     if args.let:
    #         act_scales = torch.load(args.act_scales)
    #         act_shifts = torch.load(args.act_shifts)
    #     omniquant(
    #         lm,
    #         args,
    #         dataloader,
    #         act_scales,
    #         act_shifts,
    #         logger,
    #     )
    #     logger.info(time.time() - tick)

    import pathlib, os
    import logging

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    #### Download scifact.zip dataset and unzip the dataset
    dataset = "scifact"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    #### Provide the data path where scifact has been downloaded and unzipped to the data loader
    # data folder would contain these files: 
    # (1) scifact/corpus.jsonl  (format: jsonlines)
    # (2) scifact/queries.jsonl (format: jsonlines)
    # (3) scifact/qrels/test.tsv (format: tsv ("\t"))

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    #### Lexical Retrieval using Bm25 (Elasticsearch) ####
    #### Provide a hostname (localhost) to connect to ES instance
    #### Define a new index name or use an already existing one.
    #### We use default ES settings for retrieval
    #### https://www.elastic.co/

    hostname = "localhost" #localhost
    index_name = dataset # scifact

    #### Intialize #### 
    # (1) True - Delete existing index and re-index all documents from scratch 
    # (2) False - Load existing index
    initialize = True # False

    #### Sharding ####
    # (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1 
    # SciFact is a relatively small dataset! (limit shards to 1)
    number_of_shards = 1
    model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

    # (2) For datasets with big corpus ==> keep default configuration
    # model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
    retriever = EvaluateRetrieval(model)

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)


if __name__ == "__main__":
    print(sys.argv)
    main()