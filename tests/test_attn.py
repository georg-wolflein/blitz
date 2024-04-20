import pytest
import torch
from torch.nn import functional as F
from functools import partial

from attn_torch import torch_scaled_dot_product_attention
from flash_attn_torch import torch_flash_attention_kernel


@pytest.mark.parametrize(
    "reference,implementation",
    [
        (F.scaled_dot_product_attention, torch_scaled_dot_product_attention),
        # (F.scaled_dot_product_attention, partial(torch_flash_attention_kernel, B_r=2, B_c=2)),
    ],
)
def test_compare_scaled_dot_product_attention_implementations(reference, implementation):
    B = 2
    H = 2
    S = 4
    D_k = 8
    D_v = 16

    q = torch.randn(B, H, S, D_k)
    k = torch.randn(B, H, S, D_k)
    v = torch.randn(B, H, S, D_v)

    q = q.contiguous().cuda()
    k = k.contiguous().cuda()
    v = v.contiguous().cuda()

    outputs, *_ = implementation(q, k, v)
    assert outputs.shape == (B, H, S, D_v)

    # Compare to reference implementation
    outputs_reference = reference(q, k, v)
    assert torch.allclose(outputs, outputs_reference, atol=1e-6)
