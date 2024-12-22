import os
import pathlib
import logging
import torch
import sys
from typing import List, Dict, Tuple

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking import Rerank

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

# read hugging face token
with open("huggingface_access_token.txt") as f:
    access_token = f.readline().strip()

# login with token
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


def set_seed(seed):
    """
    Helper function that sets all seeds, just in case.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_model(args):
    """
    Function that utilizes the original authors' code to load in the model.
    In this manner, we can apply all of their models into this IR setting.
    """
    # check
    if args.epochs > 0:
        assert args.lwc or args.let
        
    if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
        args.deactive_amp = True

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]
    if args.quant_method in ['irqlora', 'qlora']:
        lm = IRQLoRALMClass(args)
    else:
        lm = LMClass(args)

    lm.seqlen = args.seqlen
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc":args.lwc,
        "disable_zero_point": args.disable_zero_point
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")

    # act scales and shifts
    if args.act_scales is None:
        args.act_scales = f'./act_scales/{args.net}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'./act_shifts/{args.net}.pt'

    # quantization
    if (args.wbits < 16 or args.abits < 16) and (args.epochs > 0):
        logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            torch.save(dataloader, cache_dataloader)    
        act_scales = None
        act_shifts = None
        if args.let:
            act_scales = torch.load(args.act_scales)
            act_shifts = torch.load(args.act_shifts)
        omniquant(
            lm,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
        )
        logger.info(time.time() - tick)
    if "opt" in args.net.lower():
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
    elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
        lm.model = lm.model.to(lm.device)
    elif "falcon" in args.net.lower():
        lm.model.transformer = lm.model.transformer.to(lm.device)

    return lm, logger


class QLlamaUPRModel:
    def __init__(self, args, **kwargs):
        """
        Class for performing Unbiased Passage Ranking (UPR) using a (HQQ) quantized LLaMA3 model.

        This class wraps a (HQQ) quantized LLaMA3 model and provides functionality to compute nll
        scores for a list of query-document pairs. The model uses a specified header and instruction to 
        construct prompts for the ranking task.
        """
        self.model, self.logger = load_model(args)
        self.args = args
        self.model.tokenizer.pad_token =  self.model.tokenizer.eos_token
        self.kwargs = kwargs
    
    def predict(self, sentences: List[Tuple[str,str]], batch_size: int, **kwargs) -> List[float]:
        # prepare prompts
        separated = list(zip(*sentences))
        header = self.kwargs["header"]
        instruction = self.kwargs["instruction"]
        cond_input = [f"{header} {doc} {instruction}" for doc in separated[1]]
        queries = separated[0]

        nlls = []
        # iter through docs in batches
        for i in range(0, len(sentences), batch_size):
            batch_cond_input = cond_input[i: i + batch_size]
            batch_queries = queries[i: i + batch_size]

            # tokenize passage+instruction (context) and queries separately
            tok_contexts = self.model.tok_encode_batch(batch_cond_input) #dict object, with key input_ids (padded) and attention_mask
            context_input_ids, context_attn_mask = tok_contexts.input_ids.to(self.model.device), tok_contexts.attention_mask.to(self.model.device)
            tok_queries = self.model.tok_encode_batch(batch_queries)
            tar_input_ids, tar_attn_mask = tok_queries.input_ids.to(self.model.device), tok_queries.attention_mask.to(self.model.device)

            # combine context and queries
            combi_input_ids = torch.cat([context_input_ids, tar_input_ids], dim=1)
            combi_attn_mask = torch.cat([context_attn_mask, tar_attn_mask], dim=1)

            # mask context out for loss calc
            labels = combi_input_ids.clone()
            labels[:, :context_input_ids.shape[1]] = -100 
            labels = labels

            self.model.model.config.use_cache = False
            self.model.model.eval()
            self.model.model = self.model.model.to(self.model.device)

            # forward pass
            with torch.no_grad():
                outputs = self.model.model(input_ids=combi_input_ids, attention_mask=combi_attn_mask, labels=labels)

            loss = outputs.loss
            # - because beir does maximization
            nlls.append(-loss.item())

        return nlls


def main():
    import argparse

    # command args to load model with original paper's code
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

    # beir specific command args
    parser.add_argument("--upr", action="store_true", help="Rerank with UPR crossencoder")
    parser.add_argument("--beirdata", type=str, default="scifact", choices=["scifact", "msmarco", "trec-covid", "nfcorpus", "nq", "hotpotqa", "fiqa", "arguana", "webis-touche2020", "cqadupstack", "quora", "dbpedia-entity", "scidocs", "fever", "climate-fever"])
    parser.add_argument("--header", type=str, default="Passage:", help="Text preprended to document for UPR prompt")
    parser.add_argument("--instruction", type=str, default="Please write a question based on this passage.", help="Instruction text appended to document for UPR prompt")
    parser.add_argument("--topk", type=int, default=100, help="# documents to rerank")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length of model")

    # parse args and set seed just in case
    args = parser.parse_args()
    set_seed(args.seed)

    # debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    # download and unzip dataset
    dataset = args.beirdata
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    # first-stage retrieval with BM25
    hostname = "localhost" #server/host
    index_name = dataset
    initialize = True
    number_of_shards = 50
    model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)
    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(corpus, queries)

    # second stage retrieval with UPR
    if args.upr:
        reranker = Rerank(QLlamaUPRModel(args, header=args.header, instruction=args.instruction), batch_size=args.batch_size)
        # rerank top-100 results using the reranker provided
        rerank_results = reranker.rerank(corpus, queries, results, top_k=args.topk)

        # evaluation UPR
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)
        logging.info(f"UPR metrics. NDCG: {ndcg}, MAP: {_map}, RECALL: {recall}, PRECISION: {precision}")

    # evaluation BM25 vanilla baseline
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    logging.info(f"BM25 metrics (baseline). NDCG: {ndcg}, MAP: {_map}, RECALL: {recall}, PRECISION: {precision}")


if __name__ == "__main__":
    print(sys.argv)
    main()
