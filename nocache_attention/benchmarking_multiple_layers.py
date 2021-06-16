import argparse, logging, os, sys, random, jsonlines, shutil, time, inspect
import ujson as json
from tqdm import tqdm
import numpy as np

import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from torch import nn

from modeling import BenchmarkTransformer

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
                        choices=['vanilla', 'Q-chunking', 'performer'], help="implementation")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--n_iters', type=int, default=3, help="number of iters to average upon")
    parser.add_argument('--n_heads', type=int, required=True, help="num heads")
    parser.add_argument('--head_size', type=int, required=True, help="head_size = hidden_size / n_heads")
    parser.add_argument('--ff_dim', type=int, required=True, help="ff dim")
    parser.add_argument('--n_layers', type=int, required=True, help="number of Transformer blocks")
    parser.add_argument('--batch_size', type=int, required=True, help="batch size")
    parser.add_argument('--n_queries', type=int, required=True, help="seq len")
    parser.add_argument('--causal', action='store_true', help='causal mask')
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--Q_chunk_size_attn', type=int, required=True, help="query chunk size")
    parser.add_argument('--Q_chunk_size_ff', type=int, required=True, help="query chunk size")
    parser.add_argument('--topk_attn', type=int, required=True, help="-1 means off, should be <= n_queries")
    parser.add_argument('--topk_ff', type=int, required=True, help="-1 means off, should be <= ff_dim")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    N, H, L_Q, d_ff, d, n_layers = args.batch_size, args.n_heads, args.n_queries, args.ff_dim, args.head_size, args.n_layers
    topk_attn, topk_ff = args.topk_attn, args.topk_ff
    attn_args = {'topk':topk_attn, 'Q_chunk_size': args.Q_chunk_size_attn}
    ff_args = {'topk':topk_ff, 'Q_chunk_size': args.Q_chunk_size_ff}
    causal = args.causal
    attn_variant = args.attn_variant 
    mask = None
    
    def get_inputs():
        Q = torch.randn(N, L_Q, H*d, device='cuda', requires_grad=True)
        model = BenchmarkTransformer(H, d, d_ff, n_layers)
        return Q, model.cuda()
        
    if attn_variant == 'vanilla' and causal:
        rq = torch.arange(L_Q, device='cuda')
        mask = rq.unsqueeze(1) < rq
        del rq
        
    times, max_mem_alloc, max_mem = [], 0, 0

    for i in tqdm(range(args.n_iters)):
        Q, model = get_inputs()
        torch.cuda.synchronize()
        # following https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964/2
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = None
        with (torch.enable_grad() if args.backward else torch.no_grad()):
            out = model(Q, attn_variant, mask=mask, causal=causal, attn_args=attn_args, ff_args=ff_args)
            if args.backward:
                loss = out.mean()
        if args.backward:
            loss.backward()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end)*1e-3)  # secs
        max_mem = max(max_mem, torch.cuda.max_memory_reserved()*2**-30)
        max_mem_alloc = max(max_mem_alloc, torch.cuda.max_memory_allocated()*2**-30)
#         print(loss.item(), Q.grad.norm()) 
        del Q, model, out, loss
    print(N, L_Q, d_ff, n_layers, args.Q_chunk_size_attn, topk_attn, args.Q_chunk_size_ff, topk_ff, ':', max_mem_alloc, max_mem, np.mean(times))
    metrics = vars(args)
    metrics.update({'max_mem_alloc':max_mem_alloc, 'max_mem': max_mem, 'time': float(np.mean(times))})
    metrics = [metrics]
    args.output_dir = args.output_dir + f"/multi"
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = args.output_dir + f"/{attn_variant}_{N}_{H}_{L_Q}_{d_ff}_{n_layers}_{topk_attn}_{topk_ff}_{causal}_{args.backward}_{attn_args['Q_chunk_size']}_{ff_args['Q_chunk_size']}.jsonl"
    write_file(metrics, out_file)


if __name__ == "__main__":
    main()


'''
CUDA_VISIBLE_DEVICES=0 python benchmarking_multiple_layers.py  --output_dir out_benchmarking  --batch_size 1 --n_heads 12 --head_size 64  --ff_dim 3072 --n_layers 12  --causal --backward --Q_chunk_size_attn 1024 --Q_chunk_size_ff 4096 --topk_attn 64 --topk_ff -1 --n_queries 2048 --attn_variant Q-chunking 

'''