import pytest
import torch
from torch.nn import functional as F
from functools import partial
import itertools

from attn_torch import torch_attention
from flash_attn_torch import torch_flash_attention
from flash_attn_triton import triton_flash_attention


@pytest.mark.parametrize(
    "reference,implementation",
    [
        (F.scaled_dot_product_attention, torch_attention),
        (F.scaled_dot_product_attention, partial(torch_flash_attention, B_r=16, B_c=16)),
        (F.scaled_dot_product_attention, partial(triton_flash_attention, B_r=16, B_c=16)),
    ],
)
@pytest.mark.parametrize("Z", (1, 3, 4, 16, 17))
@pytest.mark.parametrize("H", (1, 2, 3, 4, 7, 8))
@pytest.mark.parametrize("N", (16, 17, 31, 32, 33))
@pytest.mark.parametrize("D", (16, 32, 64))
def test_compare_scaled_dot_product_attention_implementations(reference, implementation, Z, H, N, D):
    q = torch.randn(Z, H, N, D)
    k = torch.randn(Z, H, N, D)
    v = torch.randn(Z, H, N, D)

    q = q.cuda()
    k = k.cuda()
    v = v.cuda()

    outputs = implementation(q, k, v)
    outputs_reference = reference(q, k, v)

    # Compare to reference implementation
    assert outputs.shape == outputs_reference.shape
    assert torch.allclose(outputs, outputs_reference, atol=1e-5)
