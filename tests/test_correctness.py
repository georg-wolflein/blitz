import pytest
import torch
from torch.nn import functional as F
from functools import partial
import itertools

from attn_torch import torch_scaled_dot_product_attention
from flash_attn_torch import torch_flash_attention_kernel


@pytest.mark.parametrize(
    "reference,implementation",
    [
        (F.scaled_dot_product_attention, torch_scaled_dot_product_attention),
        (F.scaled_dot_product_attention, partial(torch_flash_attention_kernel, B_r=2, B_c=2)),
    ],
)
@pytest.mark.parametrize(
    "N,d",
    list(itertools.product(((i,) for i in range(5)), range(2, 17))),
)
def test_compare_scaled_dot_product_attention_implementations(reference, implementation, N, d):
    q = torch.randn(*N, d)
    k = torch.randn(*N, d)
    v = torch.randn(*N, d)

    q = q.contiguous().cuda()
    k = k.contiguous().cuda()
    v = v.contiguous().cuda()

    outputs = implementation(q, k, v)
    outputs_reference = reference(q, k, v)

    # Compare to reference implementation
    assert outputs.shape == outputs_reference.shape
    assert torch.allclose(outputs, outputs_reference, atol=1e-6)
