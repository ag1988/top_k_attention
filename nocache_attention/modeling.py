import copy
import math
import os
import warnings
import logging 

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

from nocache_attention import AttentionNoCache, AttentionNoCacheSparse, max_neg_value
from performer_pytorch import FastAttention


class BenchmarkSelfAttention(nn.Module):
    def __init__(self, H, d):
        super().__init__()
        self.H, self.d = H, d
        self.wQ, self.wK, self.wV, self.wO = (nn.Linear(H*d, H*d) for _ in range(4))  
        f, drop = nn.Softmax(-1), nn.Dropout(0.1)
        self.activation = lambda x: drop(f(x * d**-0.5))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.H, self.d)
        return x.view(*new_x_shape).permute(0, 2, 1, 3)

    def forward(self, attn_variant, Q, K, V, mask=None, causal=False, extra_args=None):
        in_shape = Q.shape
        Q = self.transpose_for_scores(self.wQ(Q))
        K = self.transpose_for_scores(self.wK(K))
        V = self.transpose_for_scores(self.wV(V))
        activation = self.activation
        
        if attn_variant == 'vanilla':
            out = Q.matmul(K.transpose(-1, -2))
            if mask is not None:
                out.masked_fill_(mask, max_neg_value(out))
            #--------------
#             topk = extra_args['topk']
#             assert topk > 0
#             print('vanilla topk', topk)
#             top_dots, top_inds = out.topk(topk, dim=-1, sorted=False)
#             out = out.zero_().scatter_(-1, top_inds, activation(top_dots)).matmul(V) 
            #--------------
            out = activation(out).matmul(V)
        elif attn_variant == 'Q-chunking':
            out = AttentionNoCache(activation)(Q, K, V, causal_masking=causal, args=extra_args)
        elif attn_variant == 'performer':
            out = FastAttention(dim_heads=self.d, nb_features=256, causal=causal)(Q, K, V)
        elif attn_variant == 'topk-sparse':
            assert Q.ndim == 4
            shape_4 = Q.shape
            Q, K, V = map(lambda x: x.reshape((-1,) + x.shape[2:]), (Q, K, V))
            assert Q.ndim == 3
            out = AttentionNoCacheSparse(activation)(Q, K, V, causal_masking=causal, args=extra_args)
            out = out.reshape(shape_4)
        else:
            assert False, 'not implemented'
        
        return self.wO(out.permute(0, 2, 1, 3).reshape(in_shape))

    
class BenchmarkFF(nn.Module):
    def __init__(self, L_K, d):
        super().__init__()
        self.wi = nn.Linear(d, L_K, bias=False)
        self.wo = nn.Linear(L_K, d, bias=False)
        f, drop = nn.ReLU(), nn.Dropout(0.1)
        self.activation = lambda x: drop(f(x))

    def forward(self, attn_variant, Q, extra_args=None):
        in_shape = Q.shape
        Q = Q.view(-1, in_shape[-1])                          #  [b*t,d]
        K = self.wi.weight                                    #  [ff,d]
        V = self.wo.weight.t()                                #  [ff,d]
        activation = self.activation
        
        if attn_variant == 'vanilla':
            out = activation(Q.matmul(K.transpose(-1, -2))).matmul(V)
        elif attn_variant == 'Q-chunking':
            out = AttentionNoCache(activation)(Q, K, V, args=extra_args)
        elif attn_variant == 'topk-sparse':
            out = AttentionNoCacheSparse(activation)(Q, K, V, args=extra_args)
        else:
            assert False, 'not implemented'
        
        return out.view(in_shape)
    

class BenchmarkTransformerBlock(nn.Module):
    def __init__(self, n_heads, d_head, d_ff):
        super().__init__()
        d_model = n_heads*d_head
        self.attn = BenchmarkSelfAttention(n_heads, d_head)
        self.norm_a = nn.LayerNorm(d_model)
        self.ff = BenchmarkFF(d_ff, d_model)
        self.norm_f = nn.LayerNorm(d_model)
    
    def forward(self, Q, attn_variant, mask=None, causal=False, attn_args=None, ff_args=None):
        in_shape = Q.shape
        Q = self.norm_a(Q + self.attn(attn_variant, Q, Q, Q, mask=mask, causal=causal, extra_args=attn_args))
        assert in_shape == Q.shape
        attn_variant = 'vanilla' if attn_variant == 'performer' else attn_variant
        Q = self.norm_f(Q + self.ff(attn_variant, Q, extra_args=ff_args))
        assert in_shape == Q.shape
        return Q


class BenchmarkTransformer(nn.Module):
    def __init__(self, n_heads, d_head, d_ff, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([BenchmarkTransformerBlock(n_heads, d_head, d_ff) for _ in range(n_layers)])
        
    def forward(self, Q, attn_variant, mask=None, causal=False, attn_args=None, ff_args=None):
        for layer in self.layers:
            Q = layer(Q, attn_variant, mask=mask, causal=causal, attn_args=attn_args, ff_args=ff_args)
        return Q


    