# read lines and calculate F1/EM
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
from pathlib import Path
from local_utils import read_file

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ans_jsonl", default="data/preprocessed_datasets/squad1_1/dev_ans.jsonl", 
                        type=Path, help="Path to pre-processed ans jsonl.")
    parser.add_argument("--preds_jsonl", default="out_preds/squad/preds.jsonl", type=Path, help="Path to pre-processed jsonl.")
    parser.add_argument("--dataset", default="squad1_1", type=str, help="squad1_1 / squad2")
    args = parser.parse_args()
    
#     preds = read_file(args.preds_jsonl)
#     # group by dataset
#     preds = list(filter(lambda d: args.dataset == d['dataset'], data))
    
    gold = []
    with open(args.ans_jsonl) as f:
        for i, l in enumerate(f.readlines()):
            gold.append(json.loads(l.replace("\n", "")))

    preds = read_file(args.preds_jsonl)
    preds = list(filter(lambda d: args.dataset == d['dataset'], preds))
    
    f1 = exact_match = 0
    for i, prediction in enumerate(preds):
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction['pred'].replace("\n", ""), gold[prediction['id']])
        f1 += metric_max_over_ground_truths(
            f1_score, prediction['pred'].replace("\n", ""), gold[prediction['id']])

    exact_match = 100.0 * exact_match / len(gold)
    f1 = 100.0 * f1 / len(gold)

    print({'exact_match': exact_match, 'f1': f1})

if __name__ == "__main__":
    main()
    

'''python squad_eval.py --preds_jsonl out_preds/squad/preds.jsonl --dataset squad1_1
'''