from __future__ import absolute_import

import argparse, logging, os, sys, random, jsonlines, shutil, time, inspect
import ujson as json

from io import open
from collections import namedtuple, defaultdict
from pathlib import Path
from tqdm import tqdm, trange
from itertools import chain
import numpy as np
import torch
#torch.backends.cudnn.deterministic = True  # can slow down code
#torch.backends.cudnn.benchmark = False

from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.file_utils import TRANSFORMERS_CACHE

# sys.path.insert(1, os.path.join(sys.path[0], '..'))  # to import from parent dir
from local_utils import read_file, write_file, getGiB
from squad_eval import f1_score, exact_match_score, metric_max_over_ground_truths
from modeling import T5DenseReluDense

# suppress warnings
import warnings 
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],)
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", default='', type=Path,
                        help="Path to pre-processed jsonl.")
    parser.add_argument("--model_name_or_path", default='', type=str, help="T5 pre-trained model (t5-base,etc) or path to dir with a saved model.")
    parser.add_argument("--output_dir", default='./out_temp', type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")    
    parser.add_argument('--ff_proj_topk', type=int, default=-1, help="topk mask before FF intermediate activation")
    parser.add_argument('--cache_dir', type=Path, default=TRANSFORMERS_CACHE, help="transformers cache")
    parser.add_argument("--local_files_only", action='store_true', help="Disable downloading.")
    parser.add_argument("--num_samples", type=int, default=-1, help="for debugging.")
    
    args = parser.parse_args()
    
    logger.info(f'CACHE DIR: {args.cache_dir}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    logger.info(f'customizing FF module: for each tok only top ({args.ff_proj_topk}) FF inner prods kept out of {model.config.d_ff}')
    T5DenseReluDense.replace_ff_with_custom(model, config=model.config, ff_proj_topk=args.ff_proj_topk)
    
    max_seq_len = args.max_seq_length
    device, args.n_gpu = torch.device("cuda"), torch.cuda.device_count()
    
    assert model.config.num_layers % args.n_gpu == 0
    per_gpu_num_layers = model.config.num_layers // args.n_gpu
    logger.info(f'model parallel: placing {per_gpu_num_layers}+{per_gpu_num_layers} layers on each one of {args.n_gpu} devices')
    device_map = {i: list(range(i*per_gpu_num_layers, (i+1)*per_gpu_num_layers)) for i in range(args.n_gpu)}
    model.parallelize(device_map)
    
    logger.info(f'reading {args.file_path} ...')
    data = read_file(args.file_path)              # {'input', 'ans', 'dataset', 'split', 'num_t5_toks'}
    
#     datasets = ['squad1_1', 'squad2', 'boolq', 'narrativeqa', 'mctest_corrected_the_separator', 
#                 'race_string', 'openbookqa', 'arc_easy', 'arc_hard', "ai2_science_middle", "ai2_science_elementary"]
#     data = list(filter(lambda d: d['dataset'] in datasets, data))
#     logger.info(f'only kept {len(data)} instances in {datasets}')
    
    logger.info(f"mean input len: {np.mean([d['num_t5_toks'] for d in data])}")
    logger.info(f'note that instances > {max_seq_len-1} will be truncated!')
    
    if args.num_samples > 0:
        logger.info(f'keeping only first {args.num_samples} of {len(data)} samples!')
        data = data[:args.num_samples]
    
    logger.info(f"sorting data by input len (will save time and expose a possible OOM at start itself) ...")
    data.sort(key=lambda d: d['num_t5_toks'], reverse=True)
    
    model.eval()
    preds = []
    batch_size = args.eval_batch_size
    
    pbar = tqdm(range(len(data) // batch_size + 1))
    for i in pbar:
        input_strings = [d['input'] for d in data[batch_size*i:batch_size*(i+1)]]
        if not input_strings:
            continue
        inputs = tokenizer.batch_encode_plus(input_strings, return_tensors="pt", max_length=max_seq_len, 
                                             truncation=True, padding='longest').convert_to_tensors()
        # input_ids, attention_mask
        assert inputs.input_ids.shape[-1] <= 512, f'{inputs.input_ids.shape}'
        res = model.generate(inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device))
        preds += tokenizer.batch_decode(res, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        pbar.set_description(f'[mem {torch.cuda.max_memory_allocated()*2**-30:.1f}Gi]')

    assert len(preds) == len(data)

    em, f1, split, split_size = defaultdict(list), defaultdict(list), dict(), defaultdict(int)
    for pred, d in tqdm(zip(preds, data)):
        f1[d['dataset']].append(metric_max_over_ground_truths(f1_score, pred, d['all_answers']) * 100)
        em[d['dataset']].append(metric_max_over_ground_truths(exact_match_score, pred, d['all_answers']) * 100)
        split[d['dataset']] = d['split']
        split_size[d['dataset']] = d['split_size']
    logger.info(f'!em, f1 are from SQuAD1.1 eval script!')
    logger.info(f'!max metric is used if multiple answers are given in the corresponding <split>_ans.jsonl file!')
    for dataset, _ in em.items():
        size = split_size[dataset] if args.num_samples < 1 else len(f1[dataset]) 
        print(f'f1 {np.sum(f1[dataset])/size:.1f}, \t em {np.sum(em[dataset])/size:.1f}, \t {dataset}, \t {split[dataset]}, \t ff {model.config.d_ff}, \t k {args.ff_proj_topk}')
    
    preds = [{'pred':pred, 'id':d['id'], 'dataset':d['dataset']} for pred, d in zip(preds, data)]
    
    os.makedirs(args.output_dir, exist_ok=True)
    preds_file = args.output_dir+'/preds.jsonl'
    logger.info(f'writing preds to {preds_file}.')
    write_file(preds, preds_file)


if __name__ == "__main__":
    main()

'''
Note: bsz affects metrics as possibly padding has an effect
source activate torch17

CUDA_VISIBLE_DEVICES=0 python run_unifiedqa.py --model_name_or_path allenai/unifiedqa-t5-3b --file_path data/combined_dev.jsonl --output_dir out_preds/all_3b_-1 --eval_batch_size 32 --num_samples -1 --ff_proj_topk -1

CUDA_VISIBLE_DEVICES=0 python run_unifiedqa.py --model_name_or_path allenai/unifiedqa-t5-base --file_path data/ropes/combined_dev_unifiedqa.jsonl --output_dir out_temp --eval_batch_size 32 --num_samples -1 --ff_proj_topk -1

CUDA_VISIBLE_DEVICES=0 python run_unifiedqa.py --model_name_or_path allenai/unifiedqa-t5-base --file_path data/commonsenseqa/combined_dev_unifiedqa.jsonl --output_dir out_temp --eval_batch_size 32 --num_samples -1 --ff_proj_topk -1

(evaluating finetuned T5 models)
CUDA_VISIBLE_DEVICES=0 python run_unifiedqa.py --model_name_or_path out_csqa_256_nocache  --file_path data/commonsenseqa/combined_dev_t5.jsonl --output_dir out_temp --eval_batch_size 32 --ff_proj_topk 256
'''
