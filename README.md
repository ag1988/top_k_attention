## Memory-efficient Transformers via Top-k Attention

This repository contains the accompanying code for the paper:

**"Memory-efficient Transformers via Top-k Attention."** Ankit Gupta, Guy Dar, Shaya Goodman, David Ciprut, Jonathan Berant.
[[PDF]](https://arxiv.org/pdf/2106.06899.pdf)


### Structure
The repository contains:
* our implementation/benchmarking of top-k attention (in `nocache_attention` dir)
* unifiedqa/T5 finetuning/inference using our top-k attention at feed-forward layers (in `unifiedqa` dir)

Coming Soon: 
* BERT QA model with top-k attention
* T5 multi-head layers with top-k attention (current code is only for FF layers)

---
### Citation
```
@article{gupta2021memoryefficient,
  title={Memory-efficient Transformers via Top-k Attention}, 
  author={Ankit Gupta and Guy Dar and Shaya Goodman and David Ciprut and Jonathan Berant},
  year={2021},
  journal = {arXiv preprint arXiv:2106.06899},
}
```
