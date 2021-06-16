import argparse, logging, os, sys, random, jsonlines, shutil, time, inspect
import ujson as json
from tqdm import tqdm
import numpy as np

import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from torch import nn

from modeling import BenchmarkSelfAttention, BenchmarkFF

sys.path.insert(1, os.path.join(sys.path[0], '..'))  # to import from parent dir
from unifiedqa.local_utils import write_file


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default='./out_benchmarking', type=str,
                        help="The output directory where the results are stored.")
    parser.add_argument("--attn_variant", required=True, type=str, 
                        choices=['vanilla', 'Q-chunking', 'performer', 'topk-sparse'], help="implementation")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--n_iters', type=int, default=3, help="number of iters to average upon")
    parser.add_argument('--batch_size', type=int, required=True, help="batch size")
    parser.add_argument('--n_heads', type=int, required=True, help="num heads")
    parser.add_argument('--n_queries', type=int, required=True, help="num queries")
    parser.add_argument('--n_keys', type=int, required=True, help="num keys")
    parser.add_argument('--head_size', type=int, required=True, help="head_size = hidden_size / n_heads")
    parser.add_argument('--topk', type=int, required=True, help="-1 means off, should be <= n_keys")
    parser.add_argument('--layer', type=str, required=True, choices=['no-mask-self-attn', 'causal-self-attn', 'ff'], help="")
    parser.add_argument('--backward', action='store_true', help="also perform backward pass")
    parser.add_argument('--Q_chunk_size', type=int, required=True, help="query chunk size")
    parser.add_argument('--K_chunk_size', type=int, required=True, help="key chunk size")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    N, H, L_Q, L_K, d, topk = args.batch_size, args.n_heads, args.n_queries, args.n_keys, args.head_size, args.topk
    attn_variant = args.attn_variant 
    extra_args = {'topk':topk, 'Q_chunk_size': args.Q_chunk_size, 'K_chunk_size':args.K_chunk_size}
    causal = 'causal' in args.layer
    mask = None
    
    def get_inputs():
        if 'self-attn' in args.layer:
            assert L_Q == L_K
            Q = torch.randn(N, L_Q, H*d, device='cuda', requires_grad=True)
            model = BenchmarkSelfAttention(H, d)
        
        elif 'ff' in args.layer:
            assert not causal
            Q = torch.randn(N, L_Q, d, device='cuda', requires_grad=True)
            model = BenchmarkFF(L_K, d)
        else:
            assert False
        return Q, model.cuda()
    
    assert L_K >= topk
    
    if attn_variant == 'vanilla' and causal:
        rq, rk = torch.arange(L_Q, device='cuda'), torch.arange(L_K, device='cuda')
        mask = rq.unsqueeze(1) < rk
        del rq, rk
        
    times, max_mem_alloc, max_mem = [], 0, 0

    for i in tqdm(range(args.n_iters)):
        Q, model = get_inputs()
        model.train() if args.backward else None
        torch.cuda.synchronize()
        # following https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964/2
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = None
        with (torch.enable_grad() if args.backward else torch.no_grad()):
            if 'self-attn' in args.layer:
                out = model(attn_variant, Q, Q, Q, mask=mask, causal=causal, extra_args=extra_args)
            elif 'ff' in args.layer:
                out = model(attn_variant, Q, extra_args=extra_args)
            else:
                exit()
            if args.backward:
                loss = out.mean()
        if args.backward:
            loss.backward()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end)*1e-3)  # secs
        max_mem = max(max_mem, torch.cuda.max_memory_reserved()*2**-30)
        max_mem_alloc = max(max_mem_alloc, torch.cuda.max_memory_allocated()*2**-30)
#         print(loss.item(), Q.grad.norm().item())
        del Q, model, out, loss
    print(N, H, L_Q, L_K, d, topk, ':', max_mem_alloc, max_mem, np.mean(times))
    metrics = vars(args)
    metrics.update({'max_mem_alloc':max_mem_alloc, 'max_mem': max_mem, 'time': float(np.mean(times))})
    metrics = [metrics]
    args.output_dir = args.output_dir + f"/single-{args.layer}"
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = args.output_dir + f"/{attn_variant}_{N}_{H}_{L_Q}_{L_K}_{topk}_{causal}_{args.backward}_{extra_args['Q_chunk_size']}.jsonl"
    write_file(metrics, out_file)


if __name__ == "__main__":
    main()


'''
CUDA_VISIBLE_DEVICES=0 python benchmarking_single_layer.py  --output_dir out_benchmarking  --batch_size 1  --n_heads 12  --layer causal-self-attn  --head_size 64  --Q_chunk_size -1 --K_chunk_size -1 --topk -1  --backward   --n_queries 512  --n_keys 512  --attn_variant Q-chunking 

CUDA_VISIBLE_DEVICES=0 python benchmarking_single_layer.py  --output_dir out_benchmarking --batch_size 512 --n_heads 0  --layer ff  --head_size 768  --backward   --n_queries 512  --Q_chunk_size 1024 --K_chunk_size -1 --topk -1  --n_keys 512  --attn_variant Q-chunking 
'''