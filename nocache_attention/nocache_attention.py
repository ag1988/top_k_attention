import copy
import math
import os
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import get_device_states, set_device_states

import logging 
logger = logging.getLogger(__name__)


# --------- helper functions -----------

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


# following example for saving and setting rng here https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
# credit: @lucidrains
class Deterministic(nn.Module):
    def __init__(self, net):
        '''ensures the second forward pass inside backward pass is identical to original forward pass under stochasticity.
        '''
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng = False, set_rng = False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)


#---------------------------------------------------------------------------


class _AttentionNoCache(torch.autograd.function.Function):
    @staticmethod
    def forward(ctx, Q, K, V, activation, mask=None, args=None):
        '''Computes (args['activation'](Q.K^T)).V
           -----------
           activation: nn.Softmax(-1), nn.ReLU(), etc. (default nn.ReLU())
           NOTE: for vanilla softmax attention remember to supply activation with scaling by d**-0.25 + softmax.
           mask: bool (default None)
           args is a dict() with following optional keys:
               - topk: (default -1)
        '''
        assert isinstance(activation, Deterministic), 'activation must be wrapped in Deterministic'
        assert mask is None or (mask.dtype == torch.bool and not mask.requires_grad)
        
        topk = -1
        if args is not None and 'topk' in args and args['topk'] > 0:
            topk = args['topk']
        args.update({'topk':topk})
        
        dots = Q.matmul(K.transpose(-1, -2))  # [Lq, Lk]
        if mask is not None: 
            dots.masked_fill_(mask, max_neg_value(dots))
        
        top_dots, top_inds = None, None
        if topk > 0:
            mask = None
            top_dots, top_inds = dots.topk(topk, dim=-1, sorted=False)
            attn = dots.zero_().scatter_(-1, top_inds, activation(top_dots, record_rng=activation.training))
            # we're not caching dots so ok to overwrite
        else:
            attn = activation(dots, record_rng=activation.training)  # [Lq, Lk]
        del dots
        
        out = attn.matmul(V)
        ctx.activation = activation
        ctx.args = args
        ctx.save_for_backward(Q, K, V, mask, top_dots, top_inds)
        return out    # [Lq, d]    
    
    @staticmethod
    def backward(ctx, d_out):
        Q, K, V, mask, top_dots, top_inds = ctx.saved_tensors
        args = ctx.args
        activation, topk = ctx.activation, args['topk']
        matmul_x_t_y = _AttentionNoCache.matmul_x_t_y
        
        d_attn = d_out.matmul(V.transpose(-2,-1))  # [Lq, Lk] == [Lq, d] x [d, Lk]
        # didn't cache attn so recompute it
        # alternatively we could've cached a sparse (top-k) rep of dots & avoided recomputing it.
        # this doesn't help as the heavy ops are products involving large [Lq, Lk] matrices
        # must explore block-sparse x dense products
        if topk > 0:
            d_top_attn = d_attn.gather(-1, top_inds)
            # recompute d_top_dots later used for d_dots
            with torch.enable_grad():
                top_dots.requires_grad = True
                top_attn = activation(top_dots, set_rng=True)  # [Lq, topk]
            top_attn.backward(d_top_attn)
            d_top_dots = top_dots.grad
            del top_dots, d_top_attn
            top_attn = top_attn.detach()
            
            # compute attn
            attn = d_attn.zero_().scatter_(-1, top_inds, top_attn)  # [Lq, Lk]
            d_V = matmul_x_t_y(attn, d_out)                         # [Lk, d]  == [Lk, Lq] x [Lq, d]
            del top_attn, d_out
            # compute d_dots
            d_dots = d_attn.scatter_(-1, top_inds, d_top_dots)
            del top_inds, d_top_dots
        else:
            # recompute attn and d_dots
            dots = Q.matmul(K.transpose(-1, -2))                    # [Lq, Lk]
            if mask is not None: 
                dots.masked_fill_(mask, max_neg_value(dots))
            with torch.enable_grad():
                dots.requires_grad = True
                attn = activation(dots, set_rng=True)               # [Lq, Lk]
            attn.backward(d_attn)
            d_dots = dots.grad
            del dots, d_attn
            d_V = matmul_x_t_y(attn, d_out)                         # [Lk, d] == [Lk, Lq] x [Lq, d]
            del attn, d_out
        
        d_Q = d_dots.matmul(K)                                      # [Lq, d] == [Lq, Lk] x [Lk, d]
        d_K = matmul_x_t_y(d_dots, Q)                               # [Lk, d] == [Lk, Lq] x [Lq, d]
        return d_Q, d_K, d_V, None, None, None

    @staticmethod
    def matmul_x_t_y(x, y):
        '''compute x^T.y'''
        a, b, c = x.shape[-1], x.shape[-2], y.shape[-1]
        if b*a <= b*c + c*a:
            return x.transpose(-2,-1).matmul(y)                 # [a, c] = [a, b] x [b, c] 
        return y.transpose(-2,-1).matmul(x).transpose(-2,-1)    # [a, c] = ([c, b] x [b, a])^T

    
class AttentionNoCache(nn.Module):
    def __init__(self, activation):
        '''activation: nn.Softmax(-1), nn.ReLU(), etc.
           NOTE: for vanilla softmax attention remember to supply activation with scaling by d**-0.5 + softmax.
           If topk will be used later dont use dropout
        '''
        super().__init__()
        self.activation = Deterministic(activation)
    
    def forward(self, Q, K, V, mask=None, causal_masking=False, args=None):
        '''Computes self.activation(Q.K^T).V
           -----------
           Q, K, V: [...,Lq,d], [...,Lk,d], [...,Lk,d]
           mask: bool - must be broadcastable. True's are masked (default None)
           causal_masking: will apply causal masking (mask should be None)
           args is a dict() with following optional keys:
               - Q_chunk_size: queries are chunked and looped over to limit max mem usage (default Lq)
               - topk: (default -1)
        '''
        assert not causal_masking or mask is None, 'mask should not be provided with causal masking'
        Q_chunks, Lq = 1, Q.shape[-2] 
        if args is not None and 'Q_chunk_size' in args:
            Q_chunk_size = args['Q_chunk_size'] if args['Q_chunk_size'] > 0 else Lq
            Q_chunks = max(1, Lq // Q_chunk_size)
        
        out = Q.new_zeros(Q.shape[:-1] + (V.shape[-1],))
        for chunk_ids in torch.arange(Lq, device=Q.device).chunk(Q_chunks):
            chunk_mask = None
            if mask is not None:
                # we cant realize a large Lq x Lk mask so we must realize it after chunking 
                assert mask.shape[-2] in [1, Lq]
                chunk_mask = mask if mask.shape[-2] == 1 else mask[...,chunk_ids,:]
            elif causal_masking:
                assert Q.shape[-2] == K.shape[-2]
                chunk_mask = torch.triu(torch.ones(len(chunk_ids), K.shape[-2], device=Q.device, dtype=torch.bool),
                                        diagonal=1+chunk_ids[0])  # [Cq, Lk]
            out[...,chunk_ids,:] = _AttentionNoCache.apply(Q[...,chunk_ids,:], K, V, self.activation, chunk_mask, args)  # [Cq, d]
        return out  # [Lq,d]


class _AttentionNoCacheSparse(torch.autograd.function.Function):
    @staticmethod
    def get_sparse_coo(top_vals, top_inds, dense_size, indices=None):
        # works for all ndim
        if indices is None:
            indices = torch.cartesian_prod(*(torch.arange(j, device=top_inds.device) for j in top_inds.shape)).t()
            indices[-1] = top_inds.view(-1)
        return torch.sparse_coo_tensor(indices, top_vals.view(-1), size=dense_size), indices
    
    @staticmethod
    def forward(ctx, Q, K, V, activation, mask=None, args=None):
        '''Computes (args['activation'](Q.K^T)).V
           -----------
           activation: nn.ReLU(), etc
           args is a dict() with following keys:
               - topk: must be > 0
           NOTE: 1) pytorch 1.8 supports sparse x dense only for ndim <= 3
                 2) Dropout not recommended with topk
        '''
        assert isinstance(activation, Deterministic), 'activation must be wrapped in Deterministic'
        
        assert args is not None and 'topk' in args and 0 < args['topk'] <= K.shape[-2], 'this method requires 0 < topk << K.shape[-2]'
        topk = args['topk']
        get_sparse_coo = _AttentionNoCacheSparse.get_sparse_coo
        matmul = torch.matmul if Q.ndim != 3 else torch.bmm  # pytorch 1.8 supports sparse x dense only for ndim <= 3
        
        dots = Q.matmul(K.transpose(-1, -2))  # [Lq, Lk]
        if mask is not None: 
            dots.masked_fill_(mask, max_neg_value(dots))
            mask = None
        
        top_dots, top_inds = dots.topk(topk, dim=-1, sorted=False)  # [Lq, topk]
        del dots
        
        attn, _ = get_sparse_coo(activation(top_dots, record_rng=activation.training), 
                                    top_inds, Q.shape[:-1] + (K.shape[-2],))
        out = matmul(attn, V)  # [Lq, d] == [Lq, Lk] x [Lk, d]

        ctx.activation = activation
        ctx.args = args
        ctx.matmul = matmul
        ctx.save_for_backward(Q, K, V, top_dots, top_inds)
        return out    # [Lq, d]    
    
    @staticmethod
    def backward(ctx, d_out):
        Q, K, V, top_dots, top_inds = ctx.saved_tensors
        args = ctx.args
        activation, topk, matmul = ctx.activation, args['topk'], ctx.matmul
        get_sparse_coo = _AttentionNoCacheSparse.get_sparse_coo
        assert topk > 0
        
        # these can be fused in block-sparse
        d_top_attn = d_out.matmul(V.transpose(-2,-1)).gather(-1, top_inds)  # [Lq, topk] <-- [Lq, Lk] == [Lq, d] x [d, Lk]
        
        # recompute d_top_dots later used for d_dots
        with torch.enable_grad():
            top_dots.requires_grad = True
            top_attn = activation(top_dots, set_rng=True)  # [Lq, topk]
        top_attn.backward(d_top_attn)
        d_top_dots = top_dots.grad
        del top_dots, d_top_attn
        top_attn = top_attn.detach()                       # [Lq, topk]

        # compute attn 
        attn, indices = get_sparse_coo(top_attn, top_inds, Q.shape[:-1] + (K.shape[-2],))  # [Lq, Lk]
        del top_attn, top_inds
        d_V = matmul(attn.transpose(-1,-2), d_out)        # [Lk, d]  == [Lk, Lq] x [Lq, d]
        del d_out, attn
        # compute d_dots
        d_dots, _ = get_sparse_coo(d_top_dots, None, Q.shape[:-1] + (K.shape[-2],), indices=indices)  # [Lq, Lk]
        del d_top_dots, indices
        
        d_Q = matmul(d_dots, K)                            # [Lq, d] == [Lq, Lk] x [Lk, d]
        d_K = matmul(d_dots.transpose_(-1,-2), Q)          # [Lk, d]  == [Lk, Lq] x [Lq, d]
        
        return d_Q, d_K, d_V, None, None, None
    


class AttentionNoCacheSparse(nn.Module):
    def __init__(self, activation):
        '''activation: nn.Softmax(-1), nn.ReLU(), etc.
           NOTE: for vanilla softmax attention remember to supply activation with scaling by d**-0.5 + softmax.
           If topk will be used later , dropout not recommended.
           Uses torch.sparse to leverage sparsity.
        '''
        super().__init__()
        self.activation = Deterministic(activation)
    
    def forward(self, Q, K, V, mask=None, causal_masking=False, args=None):
        '''Computes self.activation(Q.K^T).V
           -----------
           Q, K, V: [...,Lq,d], [...,Lk,d], [...,Lk,d]
           mask: bool - must be broadcastable. True's are masked (default None)
           causal_masking: will apply causal masking (mask should be None)
           args is a dict() with following optional keys:
               - Q_chunk_size: queries are chunked and looped over to limit max mem usage (default Lq)
               - topk: must be > 0
           for FF layers please reshape to 2d beforehand
           for multi-head attention you must reshape to 3d beforehand - similarly mask
        '''
        assert not causal_masking or mask is None, 'mask should not be provided with causal masking'
        Q_chunks, Lq = 1, Q.shape[-2] 
        if args is not None and 'Q_chunk_size' in args:
            Q_chunk_size = args['Q_chunk_size'] if args['Q_chunk_size'] > 0 else Lq
            Q_chunks = max(1, Lq // Q_chunk_size)
        
        assert 0 < args['topk'] <= K.shape[-2]
        assert Q.ndim == K.ndim == V.ndim <= 3, 'forced by torch sparse requirements - please reshape to 3d beforehand'
        
        out = Q.new_zeros(Q.shape[:-1] + (V.shape[-1],))
        for chunk_ids in torch.arange(Lq, device=Q.device).chunk(Q_chunks):
            chunk_mask = None
            if mask is not None:
                # we cant realize a large Lq x Lk mask so we must realize it after chunking 
                assert mask.shape[-2] in [1, Lq]
                chunk_mask = mask if mask.shape[-2] == 1 else mask[...,chunk_ids,:]
            elif causal_masking:
                assert Q.shape[-2] == K.shape[-2]
                chunk_mask = torch.triu(torch.ones(len(chunk_ids), K.shape[-2], device=Q.device, dtype=torch.bool),
                                        diagonal=1+chunk_ids[0])  # [Cq, Lk]
            out[...,chunk_ids,:] = _AttentionNoCacheSparse.apply(Q[...,chunk_ids,:], K, V, self.activation, chunk_mask, args)  # [Cq, d]
        return out  # [Lq,d]
    

    
    
# class Matmul(torch.autograd.function.Function):
#     @staticmethod
#     def forward(ctx, A, B):
#         '''Computes A.matmul(B) with a simple optimization for large outputs.'''
#         ctx.save_for_backward(A, B)
#         C = A.matmul(B)    
#         return C
    
#     @staticmethod
#     def backward(ctx, d_C):
#         A, B = ctx.saved_tensors
#         d_A = Matmul.matmul_x_y_t(d_C, B)  # [a,b] = [a,c]x[c,b]
#         d_B = Matmul.matmul_x_t_y(A, d_C)
#         return d_A, d_B

#     @staticmethod
#     def matmul_x_t_y(x, y):
#         '''compute x^T.y'''
#         a, b, c = x.shape[-1], x.shape[-2], y.shape[-1]
#         if b*a <= b*c + c*a:
#             return x.transpose(-2,-1).matmul(y)                 # [a, c] = [a, b] x [b, c] 
#         return y.transpose(-2,-1).matmul(x).transpose(-2,-1)    # [a, c] = ([c, b] x [b, a])^T 
    
#     @staticmethod
#     def matmul_x_y_t(x, y):
#         '''compute x.y^T'''
#         a, b, c = x.shape[-2], x.shape[-1], y.shape[-2]
#         if c*b <= a*b + c*a:
#             return x.matmul(y.transpose(-2,-1))                 # [a, c] = [a, b] x [b, c] 
#         return y.matmul(x.transpose(-2,-1)).transpose(-2,-1)    # [a, c] = ([c, b] x [b, a])^T 

