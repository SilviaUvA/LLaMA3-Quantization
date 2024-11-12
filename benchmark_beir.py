import os
import pathlib
import logging
import torch
import sys
from typing import List, Dict, Tuple

#TODO should incorporate these into environment later (beir & elasticsearch) -> Successfully installed Pillow-11.0.0 beir-2.0.0 elasticsearch-7.9.1 (this one goes via sh file) faiss_cpu-1.9.0 pytrec_eval-0.5 sentence-transformers-3.2.1
try:
    from beir import util, LoggingHandler
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.search.lexical import BM25Search as BM25
    from beir.reranking import Rerank
    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "beir"])
    from beir import util, LoggingHandler
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.search.lexical import BM25Search as BM25
    from beir.reranking import Rerank
    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_model(args):
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
    # output_dir = Path(args.output_dir)
    # self.logger = utils.create_logger(output_dir)
    # self.logger.info(args)
    
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]
    if args.quant_method in ['irqlora', 'qlora']:
        lm = IRQLoRALMClass(args)
    else:
        lm = LMClass(args)

    lm.seqlen = 2048
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
        # self.logger.info(f"set quantization in gpu {gpu_id}")

    # act scales and shifts
    if args.act_scales is None:
        args.act_scales = f'./act_scales/{args.net}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'./act_shifts/{args.net}.pt'

    # quantization
    if (args.wbits < 16 or args.abits < 16) and (args.epochs > 0):
        # self.logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            # self.logger.info(f"load calibration from {cache_dataloader}")
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
            # self.logger,
        )
        # self.logger.info(time.time() - tick)
    return lm


class QLlamaDEModel:
    def __init__(self, args):
        self.model = load_model(args) # ---> HERE Load your custom model
    
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        all_hidden_states = []
        print("QUERIES: ", len(queries), len(queries[0]), queries[0])

        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            # self.model.tok_encode_batch(batch_sentences)
            # batch_query_docs = [f"{q}\n\n{doc}" for q, doc in batch_sentences] # #TODO prepend query to document maybe change it later in some way or something idk what would be a good way
            # print("BATCH Q DOC: ", batch_query_docs)
            self.model.tokenizer.pad_token =  self.model.tokenizer.eos_token
            loaded = self.model.tok_encode_batch(batch_queries) #dict object, with key input_ids (padded) and attention_mask. I should probably normalize by using the attentionmask?
            testenc = loaded["input_ids"]
            attn_mask = loaded["attention_mask"]

            print("TESTENC WHAT ARE YOU: ", testenc.shape)
            nsamples = testenc.numel() // self.model.seqlen
            if nsamples == 0:
                nsamples += 1
            print("NSAMPLES: ", nsamples, "TESTENC.NUMEL: ", testenc.numel())
            self.model.model.config.use_cache = False
            self.model.model.eval()
            for j in tqdm(range(nsamples)):
                batch = testenc[:, (j * self.model.seqlen) : ((j + 1) * self.model.seqlen)].to(self.model.device)
                print("BATCH SHAPE: ", batch.shape)
                # if "opt" in args.net.lower():
                    # outputs = lm.model.model.decoder(batch)
                # elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
                outputs = self.model.model.model(batch) #TODO probably need to adjust the if/elif etc as well for full compatibility with the rest of their code...
                print("OUTPUTS: ", outputs)
                # elif "falcon" in args.model:
                    # outputs = lm.model.transformer(batch)
                hidden_states = outputs[0]
                all_hidden_states.append(hidden_states)
            
        return all_hidden_states
    
    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        pass

class QLlamaCEModel:
    def __init__(self, args):   
        self.model = load_model(args) # ---> HERE Load your custom model
        self.args = args

        # self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False,legacy=False)

    
    # Write your own score function, which takes in query-document text pairs and returns the similarity scores
    def predict(self, sentences: List[Tuple[str,str]], batch_size: int, **kwags) -> List[float]:
        # pass # return only the list of float scores
        # print(sentences) #TODO are these already tokenized? Otherwise need to check if lm has tokenizer?
        print("LEN SENTENCES: ", len(sentences)) #TODO why 3k?
        ppls = []

        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            # self.model.tok_encode_batch(batch_sentences)
            batch_query_docs = [f"{q}\n\n{doc}" for q, doc in batch_sentences] # #TODO prepend query to document maybe change it later in some way or something idk what would be a good way
            print("BATCH Q DOC: ", batch_query_docs)
            self.model.tokenizer.pad_token =  self.model.tokenizer.eos_token
            loaded = self.model.tok_encode_batch(batch_query_docs) #dict object, with key input_ids (padded) and attention_mask. I should probably normalize by using the attentionmask?
            testenc = loaded["input_ids"]
            test_attn_mask = loaded["attention_mask"]

            print("TESTENC WHAT ARE YOU: ", testenc.shape)
            nsamples = testenc.numel() // self.model.seqlen
            if nsamples == 0:
                nsamples += 1
            print("NSAMPLES: ", nsamples, "TESTENC.NUMEL: ", testenc.numel())
            use_cache = self.model.model.config.use_cache
            self.model.model.config.use_cache = False
            self.model.model.eval()
            nlls = []
            for j in tqdm(range(nsamples)):
                batch = testenc[:, (j * self.model.seqlen) : ((j + 1) * self.model.seqlen)].to(self.model.device)
                print("BATCH SHAPE: ", batch.shape)
                # if "opt" in args.net.lower():
                    # outputs = lm.model.model.decoder(batch)
                # elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
                outputs = self.model.model.model(batch) #TODO probably need to adjust the if/elif etc as well for full compatibility with the rest of their code...
                print("OUTPUTS: ", outputs)
                # elif "falcon" in args.model:
                    # outputs = lm.model.transformer(batch)
                hidden_states = outputs[0] #TODO perhaps just do this one lmaooooo
                print("HIDDEN STATES: ", hidden_states.shape, hidden_states)
                print("LM HEAD", self.model.model.lm_head)
                logits = self.model.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (j * self.model.seqlen) : ((j + 1) * self.model.seqlen)][
                    :, 1:
                ].to(self.model.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * self.model.seqlen
                print("NLLS: ", nlls, neg_log_likelihood)
                nlls.append(neg_log_likelihood) #TODO might have to multiply either here or below with -1 to maximize log likelihood
                if j == self.args.limit:
                    break
    
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * self.model.seqlen)) #TODO might have to multiply it here or above with -1 to maximize log likelihood
            self.logger.info(f'?? : {ppl.item()}') #TODO
            self.model.model.config.use_cache = use_cache #TODO when summing NLL / LL, remove the values from the query? idk
            ppls.append(ppl.item())
        print("PPLS", ppls)
        return ppls



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
    set_seed(args.seed)

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

    ################################################
    #### (2) RERANK Top-100 docs using Cross-Encoder
    ################################################

    #### Reranking using Cross-Encoder models #####
    #### https://www.sbert.net/docs/pretrained_cross-encoders.html
    # cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-electra-base') #TODO load custom CE in here
    # reranker = Rerank(cross_encoder_model, batch_size=128)
    reranker = Rerank(QLlamaCEModel(args), batch_size=args.batch_size)
    # reranker = Rerank(QLlamaCEModel(args), batch_size=128)
    # Rerank top-100 results using the reranker provided
    rerank_results = reranker.rerank(corpus, queries, results, top_k=10) #TODO changed top_k=100 to 10
    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)
    print("NDCG, MAP, RECALL, PRECISION, HOLE: ", ndcg, _map, recall, precision)


    # #### Retrieve dense results (format of results is identical to qrels)
    # model = DRES(QLlamaDEModel(args), batch_size=args.batch_size)
    # dense_retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values=[1,3,5,10]) #TODO removed 100 for now
    # rerank_results = dense_retriever.rerank(corpus, queries, results, top_k=10)

    # #### Evaluate your retrieval using NDCG@k, MAP@K ...
    # ndcg, _map, recall, precision, hole = dense_retriever.evaluate(qrels, rerank_results, retriever.k_values)
    # print("NDCG, MAP, RECALL, PRECISION, HOLE: ", ndcg, _map, recall, precision, hole)



    # #### Print top-k documents retrieved ####
    # top_k = 10

    # query_id, ranking_scores = random.choice(list(rerank_results.items()))
    # scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    # logging.info("Query : %s\n" % queries[query_id])

    # # for rank in range(top_k):
    # #     doc_id = scores_sorted[rank][0]
    # #     # Format: Rank x: ID [Title] Body
    # #     logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))

    ### Evaluate your retrieval using NDCG@k, MAP@K ...
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    # ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)


if __name__ == "__main__":
    print(sys.argv)
    main()