import os
import logging
import torch
import torch.nn as nn
import sys
import time
import mteb
from mteb import MTEB
from pathlib import Path
import utils
from models.LMClass import LMClass
from models.IRQLoRALMClass import IRQLoRALMClass
import random
import numpy as np
from tqdm import tqdm
from quant.omniquant import omniquant
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
from datautils import get_loaders
from typing import List, Dict, Tuple

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking import Rerank
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

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

# class STSEvalModel:
#     def __init__(self, model):
#         self.model = model
#         self.device = model.device if hasattr(model, 'device') else 'cuda'
        
#     def encode(self, sentences, batch_size=32, **kwargs):
#         """Encode sentences into embeddings"""
#         self.model.eval()
#         embeddings = []
        
#         with torch.no_grad():
#             for i in tqdm(range(0, len(sentences), batch_size)):
#                 batch = sentences[i:i + batch_size]
                
#                 # Get model outputs
#                 outputs = self.model(batch)
                
#                 # Use last hidden state as sentence embedding
#                 # Adjust this based on your model's output format
#                 if isinstance(outputs, torch.Tensor):
#                     last_hidden = outputs
#                 else:
#                     last_hidden = outputs.last_hidden_state
                
#                 # Mean pooling over sequence length
#                 sentence_embeddings = torch.mean(last_hidden, dim=1)
                
#                 embeddings.append(sentence_embeddings.cpu())
                
#         return torch.cat(embeddings, dim=0).numpy()

class STSEvalModel:
    def __init__(self, model):
        self.model = model
        self.device = model.device if hasattr(model, 'device') else 'cuda'
        # Set padding token to be the same as EOS token
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        
    def encode(self, sentences, batch_size=32, **kwargs):
        """Encode sentences into embeddings"""
        # The model is already set to eval mode in LMClass.__init__
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sentences), batch_size)):
                batch = sentences[i:i + batch_size]
                
                # Get model outputs using the model's tokenizer and forward pass
                encoded = self.model.tok_encode_batch(batch)
                input_ids = encoded["input_ids"].to(self.model.device)
                attention_mask = encoded["attention_mask"].to(self.model.device)
                
                if "llama" in self.model.args.net.lower() or "mixtral" in self.model.args.net.lower():
                    outputs = self.model.model.model(input_ids, attention_mask=attention_mask)
                elif "opt" in self.model.args.net.lower():
                    outputs = self.model.model.model.decoder(input_ids, attention_mask=attention_mask)
                elif "falcon" in self.model.args.model:
                    outputs = self.model.model.transformer(input_ids, attention_mask=attention_mask)
                else:
                    raise NotImplementedError(f"Model type {self.model.model_name} not supported")
                
                # Use last hidden state
                last_hidden = outputs[0]
                
                # Mean pooling over sequence length (using attention mask)
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sentence_embeddings = torch.sum(last_hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                
                embeddings.append(sentence_embeddings.cpu())
                
        return torch.cat(embeddings, dim=0).numpy()

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

class QLlamaDEModel:
    def __init__(self, args):
        self.model, self.logger = load_model(args) # ---> HERE Load your custom model
        self.model.tokenizer.pad_token =  self.model.tokenizer.eos_token
        self.args = args
    
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        all_hidden_states = []

        queries = [query + self.model.tokenizer.eos_token for query in queries]

        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            loaded = self.model.tok_encode_batch(batch_queries)
            input = loaded["input_ids"].to(self.model.device)
            attn_mask = loaded["attention_mask"].to(self.model.device)
            eos_token_ids = torch.sum(attn_mask, 1) - 1
            # eos_token_ids = torch.sum(attn_mask, 1) - 2 #TODO now taking last word of document but that makes no sense
            self.model.model.config.use_cache = False
            self.model.model.eval()

            with torch.no_grad():
                if "opt" in self.args.net.lower():
                    outputs = self.model.model.model.decoder(input)
                elif "llama" in self.args.net.lower() or "mixtral" in self.args.net.lower():
                    outputs = self.model.model.model(input)
                elif "falcon" in self.args.model:
                    outputs = self.model.model.transformer(input)

            hidden_states = outputs[0] #TODO hidden states from lm_head..?

            for j in range(hidden_states.shape[0]):
                all_hidden_states.append(hidden_states[j, eos_token_ids[j]].type(torch.float32).numpy(force=True))

        all_hidden_states = np.array(all_hidden_states)
        # all_hidden_states = torch.tensor(all_hidden_states)

        # Batch size x embedding dim (B, 4096) 
        return all_hidden_states

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        all_docs = []
        title_text_corpus = [f"{i['title']}\n\n{i['text']} {self.model.tokenizer.eos_token}" for i in corpus]

        for i in range(0, len(title_text_corpus), batch_size):
            batch_docs = title_text_corpus[i: i + batch_size]
            loaded = self.model.tok_encode_batch(batch_docs)
            input = loaded["input_ids"].to(self.model.device)
            attn_mask = loaded["attention_mask"].to(self.model.device)
            eos_token_ids = torch.sum(attn_mask, 1) - 1
            # eos_token_ids = torch.sum(attn_mask, 1) - 2 #TODO now taking last word of document but that makes no sense
            self.model.model.config.use_cache = False
            self.model.model.eval()

            with torch.no_grad():
                if "opt" in self.args.net.lower():
                    outputs = self.model.model.model.decoder(input)
                elif "llama" in self.args.net.lower() or "mixtral" in self.args.net.lower():
                    outputs = self.model.model.model(input)
                elif "falcon" in self.args.model:
                    outputs = self.model.model.transformer(input)

            hidden_states = outputs[0]
            for j in range(hidden_states.shape[0]):
                all_docs.append(hidden_states[j, eos_token_ids[j]].type(torch.float32).numpy(force=True))
        
        all_docs = np.array(all_docs)
        # all_docs = torch.tensor(all_docs)
        # Batch size x embedding dim (B, 4096)
        return all_docs

class QLlamaCEModel:
    def __init__(self, args):   
        self.model, self.logger = load_model(args) # ---> HERE Load your custom model
        self.args = args
        self.model.tokenizer.pad_token =  self.model.tokenizer.eos_token
    
    # Write your own score function, which takes in query-document text pairs and returns the similarity scores
    def predict(self, sentences: List[Tuple[str,str]], batch_size: int, **kwargs) -> List[float]:
        # return only the list of float scores
        ppls = []

        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            batch_query_docs = [f"{q}\n\n{doc}" for q, doc in batch_sentences]
            loaded = self.model.tok_encode_batch(batch_query_docs) #dict object, with key input_ids (padded) and attention_mask
            testenc = loaded["input_ids"].to(self.model.device)
            # test_attn_mask = loaded["attention_mask"].to(self.model.device)

            nsamples = testenc.numel() // self.model.seqlen
            if nsamples == 0:
                nsamples += 1

            use_cache = self.model.model.config.use_cache
            self.model.model.config.use_cache = False
            self.model.model.eval()
            nlls = []
            for j in tqdm(range(nsamples)):
                batch = testenc[:, (j * self.model.seqlen) : ((j + 1) * self.model.seqlen)].to(self.model.device)

                with torch.no_grad():
                    if "opt" in self.args.net.lower():
                        outputs = self.model.model.model.decoder(batch)
                    elif "llama" in self.args.net.lower() or "mixtral" in self.args.net.lower():
                        outputs = self.model.model.model(batch)
                    elif "falcon" in self.args.model:
                        outputs = self.model.model.transformer(batch)

                hidden_states = outputs[0]
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
                nlls.append(neg_log_likelihood)
                if j == self.args.limit:
                    break
    
            ppl = -torch.exp(torch.stack(nlls).sum() / (nsamples * self.model.seqlen)) # - because it picks the highest scores
            self.model.model.config.use_cache = use_cache
            ppls.append(ppl.item())
        return ppls


class QLlamaUPRModel:
    def __init__(self, args, **kwargs):   
        self.model, self.logger = load_model(args) # ---> HERE Load your custom model
        self.args = args
        self.model.tokenizer.pad_token =  self.model.tokenizer.eos_token
        self.kwargs = kwargs
        self.epsilon = 0.001
    
    # Write your own score function, which takes in query-document text pairs and returns the similarity scores
    def predict(self, sentences: List[Tuple[str,str]], batch_size: int, **kwargs) -> List[float]:
        # return only the list of float scores
        separated = list(zip(*sentences))
        header = self.kwargs["header"]
        instruction = self.kwargs["instruction"]
        cond_input = [f"{header} {doc} {instruction}" for doc in separated[1]]
        queries = separated[0]

        nlls = []
        for i in range(0, len(sentences), batch_size):
            batch_cond_input = cond_input[i: i + batch_size]
            batch_queries = queries[i: i + batch_size]

            tok_contexts = self.model.tok_encode_batch(batch_cond_input) #dict object, with key input_ids (padded) and attention_mask
            context_input_ids, context_attn_mask = tok_contexts.input_ids.to(self.model.device), tok_contexts.attention_mask.to(self.model.device)
            tok_queries = self.model.tok_encode_batch(batch_queries)
            tar_input_ids, tar_attn_mask = tok_queries.input_ids.to(self.model.device), tok_queries.attention_mask.to(self.model.device)

            combi_input_ids = torch.cat([context_input_ids, tar_input_ids], dim=1)
            combi_attn_mask = torch.cat([context_attn_mask, tar_attn_mask], dim=1)

            # mask context out for loss calc
            labels = combi_input_ids.clone()
            labels[:, :context_input_ids.shape[1]] = -100 

            self.model.model.config.use_cache = False
            self.model.model.eval()

            with torch.no_grad():
                if "llama" in self.args.net.lower() or "mixtral" in self.args.net.lower():
                    outputs = self.model.model(input_ids=combi_input_ids, attention_mask=combi_attn_mask, labels=labels)
                else:
                    raise NotImplementedError
            loss = outputs.loss
            nlls.append(-loss.item())
            # nlls.append(loss.item())

        # nlls_array = np.array(nlls)
        # max_nll = np.max(nlls_array)
        # nlls_array = max_nll - nlls_array + self.epsilon
        # return nlls_array.tolist()
        return nlls

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Add your existing arguments from benchmark_beir.py
    # parser.add_argument("--model", type=str, help="model name or path")
    # parser.add_argument("--cache_dir", default="./cache", type=str)
    # parser.add_argument("--output_dir", default="../log/", type=str)
    # parser.add_argument("--save_dir", default=None, type=str)
    # parser.add_argument("--wbits", type=int, default=4)
    # parser.add_argument("--abits", type=int, default=16)
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
    # parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
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
    # Add other arguments from your original implementation

    parser.add_argument("--ce", action="store_true", help="Rerank with crossencoder")
    parser.add_argument("--be", action="store_true", help="Rerank with biencoder")
    parser.add_argument("--upr", action="store_true", help="Rerank with UPR crossencoder")
    parser.add_argument("--topk", type=int, default=100, help="Top-k documents to rerank")
    parser.add_argument("--header", type=str, default="", help="Header for UPR model")
    parser.add_argument("--instruction", type=str, default="", help="Instruction for UPR model")
    
    # # Add STS specific arguments
    parser.add_argument("--sts_tasks", type=str, nargs="+", 
                        default=["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICK-R"],
                        help="STS tasks to evaluate on")

    # # Remove STS specific arguments
    # parser.remove_argument("--sts_tasks")
    
    # Add retrieval specific arguments
    # parser.add_argument("--retrieval_tasks", type=str, nargs="+", 
    #                 default=["MSMARCO", "TREC-COVID", "FiQA-2018", "SciFact"],
    #                 help="Retrieval tasks to evaluate on")
    parser.add_argument("--languages", type=str, nargs="+",
                    default=["eng"],
                    help="Languages to evaluate on")
    parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--max_corpus_size", type=int, default=None,
    #                help="Maximum number of documents to use from corpus")
    # parser.add_argument("--max_queries", type=int, default=None,
    #                help="Maximum number of queries to evaluate")

    
    args = parser.parse_args()
    set_seed(args.seed)

    # Load and prepare model
    model, logger = load_model(args)
    eval_model = STSEvalModel(model)

    # Initialize MTEB evaluator with specified STS tasks
    logger.info(f"Evaluating on tasks: {args.sts_tasks}")
    evaluation = MTEB(tasks=args.sts_tasks)

    # Initialize MTEB evaluator with specified retrieval tasks
    # if args.retrieval_tasks:
    #     tasks = args.retrieval_tasks
    # else:
    #     tasks = mteb.get_tasks(task_types=["Retrieval"], languages=args.languages)
    
    # logger.info(f"Evaluating on tasks: {tasks}")
    # evaluation = MTEB(tasks=tasks)
    # results = evaluation.run(model, output_folder=args.output_dir)
    
    # Run evaluation
    results = evaluation.run(eval_model, output_folder=args.output_dir)
    
    # Log results
    logger.info("=== Evaluation Results ===")
    if isinstance(results, dict):
        # Handle dictionary results
        for task, metrics in results.items():
            logger.info(f"\nTask: {task}")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value}")
    else:
        # Handle list results
        for result in results:
            if isinstance(result, dict):
                logger.info("\nTask Results:")
                for metric, value in result.items():
                    logger.info(f"{metric}: {value}")
            else:
                logger.info(f"\nResult: {result}")

    # Additional evaluations (UPR, BE, CE)
    # if args.upr or args.be or args.ce:
    #     for task in tasks:
    #         corpus, queries, qrels = task.load_data()
            
    #         # BM25 baseline (if needed)
    #         bm25 = BM25(index_name=task.name)
    #         retriever = EvaluateRetrieval(bm25, k_values=[1, 3, 5, 10, 100])
    #         results = retriever.retrieve(corpus, queries)

    #         if args.ce:
    #             reranker = Rerank(QLlamaCEModel(args), batch_size=args.batch_size)
    #             rerank_results = reranker.rerank(corpus, queries, results, top_k=args.topk)
    #             ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)
    #             logger.info(f"CE metrics for {task.name}. NDCG: {ndcg}, MAP: {_map}, RECALL: {recall}, PRECISION: {precision}")
            
    #         if args.upr:
    #             reranker = Rerank(QLlamaUPRModel(args, header=args.header, instruction=args.instruction), batch_size=args.batch_size)
    #             rerank_results = reranker.rerank(corpus, queries, results, top_k=args.topk)
    #             ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)
    #             logger.info(f"UPR metrics for {task.name}. NDCG: {ndcg}, MAP: {_map}, RECALL: {recall}, PRECISION: {precision}")

    #         if args.be:
    #             dense_model = DRES(QLlamaDEModel(args), batch_size=args.batch_size)
    #             dense_retriever = EvaluateRetrieval(dense_model, score_function="cos_sim", k_values=[1,3,5,10,100])
    #             rerank_results = dense_retriever.rerank(corpus, queries, results, top_k=args.topk)
    #             ndcg, _map, recall, precision = dense_retriever.evaluate(qrels, rerank_results, retriever.k_values)
    #             logger.info(f"BE metrics for {task.name}. NDCG: {ndcg}, MAP: {_map}, RECALL: {recall}, PRECISION: {precision}")

if __name__ == "__main__":
    main() 