from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
import collections, jsonlines, pickle
from multiprocessing import Pool
from functools import partial
from collections import Counter, defaultdict


import random, sys, csv, argparse, os
from random import randrange, randint, shuffle, choice, sample
import numpy as np
import jsonlines

# suppress warnings
import warnings 
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from transformers import AutoTokenizer
from local_utils import read_file, write_file

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],)
logger.setLevel(logging.INFO)


def func_num_toks(d, tokenizer):
    return len(tokenizer.tokenize(d['input']))
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data/preprocessed_datasets/', type=Path, 
                        help="direc with unified qa formatted datasets.")
    parser.add_argument("--out_dir", default='data/', type=Path, help="direc for output file.")
    parser.add_argument("--split", default='dev', type=str,
                        help="train/dev in case dev doesn't exist test data will be used.")
    parser.add_argument("--tokenizer", default='allenai/unifiedqa-t5-base', type=str,
                        help="tokenizer.")
    parser.add_argument("--cores", default=-1, type=int, help="num cpu cores for mp.")
    parser.add_argument("--weighted_duplicate_to", default=-1, type=int, help="each dataset is duplicated to this size.")
    parser.add_argument("--datasets", default='', type=str, help="comma-seperated list of datasets to process / `all`")
    parser.add_argument("--add_special_words", action='store_true', help="add extra words like 'question', 'context' to inputs for easier T5 finetuning")
    
    args = parser.parse_args()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    direc = str(args.data_dir)
    split = args.split
    pool = Pool(args.cores) if args.cores > 1 else None
    
    if args.datasets == 'all':
        logger.info(f'processing all datasets.')
    else:
        datasets = ['commonsenseqa', 'squad1_1', 'squad2', 'boolq', 'narrativeqa', 'mctest_corrected_the_separator', 
                    'race_string', 'openbookqa', 'arc_easy', 'arc_hard', 'ai2_science_middle', 'ai2_science_elementary']
        if args.datasets != '':  # commma-separated list of datasets
            datasets = [x.strip() for x in args.datasets.split(',')]
        logger.info(f'only processing samples in {datasets}.')
    
    all_data = []
    for dataset in os.listdir(direc):
        if args.datasets != 'all' and dataset not in datasets:
            continue
        path = f"{direc}/{dataset}/{split}.tsv"
        split_used = split
        if split == 'dev' and not os.path.exists(path):
            path = f"{direc}/{dataset}/test.tsv"
            split_used = 'test'
        answers_file = f"{direc}/{dataset}/{split}_ans.jsonl"  # some datasets have multiple correct answers
        all_answers = read_file(answers_file) if os.path.exists(answers_file) else None 
        data = []
        with open(path) as fd:
            rd = csv.reader(fd, delimiter="\t")
            n_kept, n_ignored = 0, 0
            for id, row in enumerate(rd):
                if len(row) != 2:
                    n_ignored += 1
                    continue
                n_kept += 1
                input = row[0]
                if args.add_special_words:  # similar to T5 squad format
                    sep_idx = row[0].index('\\n')
                    input = f"question: {row[0][:sep_idx]} context: <yes> <no> <No Answer> {row[0][sep_idx+2:]}" 
                data.append({'dataset': dataset, 'split': split_used, 
                             'id': id, 'input': input, 'ans': row[1], 'all_answers': all_answers[id] if all_answers else [row[1]]}
                           )
            func_num_toks_ = partial(func_num_toks, tokenizer=tokenizer)
            n_toks = pool.map(func_num_toks_, data) if pool is not None else list(map(func_num_toks_, data))
            for d, nt in zip(data, n_toks):
                d['num_t5_toks'] = nt
                d['split_size'] = len(all_answers) if all_answers else n_kept + n_ignored  # there are discrepencies in unifiedqa data
            logger.info(f'{split_used} \t  kept: {n_kept} \t ignored: {n_ignored} \t length_98pcl: {np.percentile(n_toks, q=98):.0f} \t {dataset}')
        
        all_data += data
    
    format = 'unifiedqa' if not args.add_special_words else 't5'
    out_file = f'{args.out_dir}/combined_{split}_{format}.jsonl'
    logger.info(f'writing {len(all_data)} samples to {out_file}...')
    os.makedirs(args.out_dir, exist_ok=True)
    write_file(all_data, out_file)
    logger.info('Done.')
    
    if args.weighted_duplicate_to > 0:
        # creating weighted set
        logger.info(f'Duplicating each dataset to size >= {args.weighted_duplicate_to}...')
        dataset_data = defaultdict(list)
        for d in all_data:
            dataset_data[d['dataset']].append(d)
        mixture = {}
        for k, v in dataset_data.items():
            mixture[k] = v*max(1, args.weighted_duplicate_to // len(v))
        for k, v in mixture.items():
            print(f'{k}: {len(dataset_data[k])} -> {len(v)}')
        out_file = f'{args.out_dir}/weighted_{split}_{format}.jsonl'
        logger.info(f'writing {sum([len(v) for v in mixture.values()])} samples to {out_file}...')
        write_file(sum(mixture.values(), []), out_file)
        logger.info('Done.')
    
if __name__ == "__main__":
    main()

'''
# unifiedqa
python prepare_data.py --data_dir data/preprocessed_datasets --out_dir data --split dev --cores 8

# t5 finetuning
python prepare_data.py --data_dir data/preprocessed_datasets --out_dir data --split train --cores 8 --weighted_duplicate_to 30000 --add_special_words
python prepare_data.py --data_dir data/preprocessed_datasets --out_dir data --split dev --cores 8 --add_special_words


# bash
for k in commonsenseqa squad1_1 squad2 boolq narrativeqa mctest_corrected_the_separator race_string openbookqa arc_easy arc_hard ai2_science_middle ai2_science_elementary ropes
do
    python prepare_data.py --data_dir data/preprocessed_datasets --out_dir data/$k --dataset $k --split train --cores 8 --add_special_words
    python prepare_data.py --data_dir data/preprocessed_datasets --out_dir data/$k --dataset $k --split dev --cores 8 --add_special_words
    python prepare_data.py --data_dir data/preprocessed_datasets --out_dir data/$k --dataset $k --split dev --cores 8
done
'''
