import ujson as json
import random, jsonlines, pickle
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Sampler, BatchSampler


def read_file(file):
    file = str(file)
    
    if file.endswith('jsonl'):
        with jsonlines.open(file, 'r') as reader:
            return [d for d in reader.iter()]
    
    elif file.endswith('json'):
        with open(file, encoding='utf8') as f:
            return json.load(f)
    
    elif file.endswith('pt'):
        return torch.load(file, map_location='cpu')
    
    elif any([file.endswith(ext) for ext in ['pkl', 'pickle', 'pck', 'pcl']]):
        with open(file, 'rb') as f:
            return pickle.load(f)
        
    elif file.endswith('txt'):
        with open(file, encoding='utf8') as f:
            return f.read()


def write_file(data, file):
    file = str(file)
    
    if file.endswith('jsonl'):
        with jsonlines.open(file, mode='w') as writer:
            writer.write_all(data)

    elif file.endswith('json'):
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    
    elif file.endswith('pt'):
        torch.save(data, file)
        
    elif any([file.endswith(ext) for ext in ['pkl', 'pickle', 'pck', 'pcl']]):
        with open(file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    elif file.endswith('txt'):
        with open(file, 'w', encoding='utf8') as f:
            f.write(data)


def getGiB(t, ndigits=1):
    return round(t.element_size()*t.numel()*2**-30, ndigits)


class ChunkedRandomSampler(Sampler):
    r"""Samples contiguous chunks of given length randomly.
    Arguments:
        data_source (Dataset): dataset to sample from
        chunk_size (int): number of contiguous samples in a chunk, default=1.
    """

    def __init__(self, data_source, chunk_size=1):
        self.data_source = data_source
        self.chunk_size = chunk_size

    @property
    def num_samples(self):
        # dataset size might change at runtime
        return len(self.data_source)

    def __iter__(self):
        n, chunk_size = len(self.data_source), self.chunk_size
        n_chunks = n // chunk_size
        indices = torch.arange(n_chunks*chunk_size).view(-1, chunk_size)[torch.randperm(n_chunks)].view(-1)
        spill = list(range(n_chunks*chunk_size, n))
        return iter(indices.tolist() + spill)

    def __len__(self):
        return self.num_samples

