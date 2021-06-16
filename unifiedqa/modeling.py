import copy
import math
import os, sys
import warnings
import logging 

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

sys.path.insert(1, os.path.join(sys.path[0], '..'))  # to import from parent dir
from nocache_attention.nocache_attention import AttentionNoCache

logger = logging.getLogger(__name__)

T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]

ACT2FN = {"gelu": F.gelu, "relu": F.relu, "softplus": F.softplus, "softmax": nn.Softmax(-1)}


class T5DenseReluDense(nn.Module):
    def __init__(self, config, ff_proj_topk=-1, Q_chunk_size=-1, no_ff_dropout=False):
        super().__init__()
        self.config = config
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.ff_proj_topk, self.Q_chunk_size = ff_proj_topk, Q_chunk_size
        self.dropout = nn.Dropout(0 if no_ff_dropout else config.dropout_rate)
        self.attention = AttentionNoCache(lambda x: self.dropout(F.relu(x)))
        # Q_chunk_size < config.d_model not helpful
        
    def forward(self, hidden_states):
        in_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, in_shape[-1])  #  [b*t,d]
        K = self.wi.weight                                    #  [ff,d]
        V = self.wo.weight.t()                                #  [ff,d]
        args = {'topk':self.ff_proj_topk, 'Q_chunk_size':self.Q_chunk_size}
        return self.attention(hidden_states, K, V, args=args).view(*in_shape)
        
#     def forward(self, hidden_states):
#         hidden_states = self.wi(hidden_states)  # [b,t,ff]
#         hidden_states = self.topk_mask_(hidden_states, k=self.ff_proj_topk, func=F.relu)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.wo(hidden_states)  # [b,t,d]
#         return hidden_states
    
#     @staticmethod
#     def topk_mask_(x, k=-1, func=None, sample_wise=False):
#         # x: [b,t,ff]
#         func = (lambda x: x) if func is None else func 
#         in_shape = x.shape
#         if sample_wise: 
#             x = x.view(*x.shape[:-2], -1)  # [b, t*ff]
#         if k <= 0 or k >= x.shape[-1]: 
#             out_x = func(x)
#         else:
#             tp_x, tp_inds = x.topk(k, dim=-1, sorted=False)
#             out_x = x.zero_().scatter_(-1, tp_inds, func(tp_x))
#         return out_x.view(*in_shape)

    @staticmethod
    def replace_ff_with_custom(model, **kwargs):
        for child_name, child in model.named_children():
            if child_name == 'DenseReluDense':
                new = T5DenseReluDense(**kwargs)
                new.load_state_dict(child.state_dict())
                setattr(model, child_name, new)
                print(f'{child_name} replaced')
                
            else:
                T5DenseReluDense.replace_ff_with_custom(child, **kwargs)

