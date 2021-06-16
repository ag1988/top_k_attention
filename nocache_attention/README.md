### Memory-efficient implementation of vanilla and top-k attention via query chunking

- Usage:

```python
import torch
from torch import nn
from nocache_attention import AttentionNoCache

bs, nh, L_Q, L_K, d = 2, 12, 16384, 16384, 64
f, drop = nn.Softmax(-1), nn.Dropout(0.1)

activation = lambda x: drop(f(x * d**-0.5)) 
args = {'topk':128, 'Q_chunk_size':2048}

Q = torch.randn(bs, nh, L_Q, d, device='cuda', requires_grad=True)
K, V = (torch.randn(bs, nh, L_K, d, device='cuda', requires_grad=True) for _ in range(2))
out = AttentionNoCache(activation)(Q, K, V, causal_masking=True, args=args) 
loss = out.mean()
loss.backward()
```  
To use `torch.sparse`-based implementation of top-k attention, use `AttentionNoCacheSparse` instead. <br/><br/>

- Benchmarking single bert-base self-attention layer with causal mask
```bash
for L in 2048 {2048..65536..2048}
do
    CUDA_VISIBLE_DEVICES=0 python benchmarking_single_layer.py  --output_dir out_benchmarking  --K_chunk_size -1 --batch_size 1  --n_heads 12  --layer causal-self-attn  --head_size 64  --Q_chunk_size 1024 --topk -1  --backward   --n_queries $L  --n_keys $L  --attn_variant <attn>
done
```
where `<attn>` can be `vanilla`, `Q-chunking`, `performer` or `topk-sparse`. For `vanilla`, `performer` hyperparams like `--Q_chunk_size`, etc are ignored. The `topk-sparse` is same as `Q_chunking` except it uses `torch.sparse` to implement sparse x dense matrix products as discussed in appendix of our paper. You can use `--topk 128` for using the top-k attention. For performer, you need to install the [pytorch port](https://github.com/lucidrains/performer-pytorch) using 
```bash
pip install  performer-pytorch  pytorch-fast-transformers>=0.3.0
``` 
<br/> 

- Benchmarking single FF layer 
```bash
for L in 2048 {2048..65536..2048}
do
    CUDA_VISIBLE_DEVICES=0 python benchmarking_single_layer.py  --output_dir out_benchmarking2  --n_heads 0 --K_chunk_size -1  --layer ff  --head_size 768  --backward   --n_queries 512  --batch_size 512  --topk -1 --n_keys $L  --Q_chunk_size 16384 --attn_variant <attn>
done
```
where `<attn>` can be `vanilla`, `Q-chunking`, `topk-sparse`. You can use `--topk 512` for using the top-k attention.  


- Benchmarking 12-layer model  
```bash
for L in 2048 {2048..32768..2048}
do
    CUDA_VISIBLE_DEVICES=0 python benchmarking_multiple_layers.py  --output_dir out_benchmarking3  --batch_size 1 --n_heads 12 --head_size 64  --ff_dim 3072 --n_layers 12  --causal --backward --Q_chunk_size_attn 1024 --Q_chunk_size_ff 4096 --topk_attn -1 --topk_ff -1  --n_queries $L --attn_variant <attn>
```
where `<attn>` can be `vanilla`, `Q-chunking`, `topk-sparse`. You can use `--topk_attn 64` for using the top-k attention at self-attention layers.

