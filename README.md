# blitz

This repository is a minimal implementation of [Flash Attention](https://arxiv.org/abs/2205.14135):
- `flash_attn_torch.py` contains an implementation of Algorithm 1 of the original Flash Attention paper in pytorch (this is of course horribly inefficient, but it is good for demonstrating how Flash Attention works in the language of numpy/pytorch as opposed to triton/CUDA)
- `flash_attn_triton.py` contains a triton implementation of Flash Attention

## To do
- [ ] implement backward pass
- [ ] implement Flash Attention 2