from torch.nn import functional as F


def torch_scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Reference implementation of scaled dot product attention using torch operations.

    Args:
        q: Queries tensor of shape [B, H, S, D_k]
        k: Keys tensor of shape [B, H, S, D_k]
        v: Values tensor of shape [B, H, S, D_v]

    Returns:
        values: The output of the attention mechanism of shape [B, H, S, D_v]
        attn_logits: The attention logits of shape [B, H, S, S]
        attention: The attention weights of shape [B, H, S, S]

    Shapes:
        B: batch size
        H: number of heads
        S: sequence length
        D_k: key dimension
        D_v: value dimension
    """
    d_k = q.shape[-1]
    attn_logits = q @ k.transpose(-2, -1)  # [batch_size, num_heads, seq_len, seq_len]
    attn_logits = attn_logits / d_k**0.5
    attention = F.softmax(attn_logits, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
    values = attention @ v  # [batch_size, num_heads, seq_len, embed_dim]
    return values, attn_logits, attention


if __name__ == "__main__":
    import torch

    # Test scaled dot product attention
    torch.manual_seed(0)
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
    torch_scaled_dot_product_attention = torch.compile(torch_scaled_dot_product_attention)

    values, *_ = torch_scaled_dot_product_attention(q, k, v)
    assert values.shape == (B, H, S, D_v)

    print(values)
