import torch
from torch.nn import functional as F
import time

from attn_torch import torch_scaled_dot_product_attention
from flash_attn_torch import torch_flash_attention_kernel

IMPLEMENTATIONS = {
    "F.scaled_dot_product_attention": F.scaled_dot_product_attention,
    "torch_scaled_dot_product_attention": torch_scaled_dot_product_attention,
    "torch_scaled_dot_product_attention (compiled)": torch.compile(torch_scaled_dot_product_attention),
    "torch_flash_attention_kernel": torch_flash_attention_kernel,
}


def prepare_inputs(N, d):
    q = torch.randn(N, d)
    k = torch.randn(N, d)
    v = torch.randn(N, d)

    q = q.contiguous().cuda()
    k = k.contiguous().cuda()
    v = v.contiguous().cuda()

    return q, k, v


def benchmark(implementation, *args, **kwargs):
    torch.cuda.synchronize()
    start = time.time()
    outputs = implementation(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()

    return outputs, end - start


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    N = 2048
    d = 512
    inputs = prepare_inputs(N, d)
    print(f"{N=}, {d=}:")
    for name, implementation in IMPLEMENTATIONS.items():
        # Dry run
        fake_inputs = prepare_inputs(N, d)
        benchmark(implementation, *fake_inputs)

        outputs, t = benchmark(implementation, *inputs)
        print(f"  {name:50s}: {t*1e3:.2f} ms")
        del outputs
