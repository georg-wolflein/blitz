import torch
import triton
import triton.language as tl


def cdiv(a, b):
    return (a + b - 1) // b


# @triton.autotune(
#     [
#         triton.Config({"B_r": B_r, "B_c": B_c}, num_warps=num_warps, num_stages=num_stages)
#         for B_r in [32, 64, 128]
#         for B_c in [32, 64, 128]
#         for num_warps in [4, 8]
#         for num_stages in [3, 4, 5]
#     ],
#     key=["Z", "H", "N", "D"],
# )
@triton.jit
def triton_flash_attention_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    stride_QZ,
    stride_QH,
    stride_QN,
    stride_QD,
    stride_KZ,
    stride_KH,
    stride_KN,
    stride_KD,
    stride_VZ,
    stride_VH,
    stride_VN,
    stride_VD,
    stride_OZ,
    stride_OH,
    stride_ON,
    stride_OD,
    Z: int,  # batch size
    H: int,  # number of heads
    N: int,  # sequence length
    D: int,  # embedding dimension (per head)
    softmax_scale: float,
    B_r: tl.constexpr,
    B_c: tl.constexpr,
    B_d: tl.constexpr,
    allow_tf32: tl.constexpr = True,
):
    assert D == B_d

    # Index into outer loop (inner loop in Algorithm 1)
    i = tl.program_id(0)

    # Find the correct start position for this block in terms of the Z, H dimensions (batch and head dimensions)
    zh = tl.program_id(1)
    z = zh // H
    h = zh % H
    Q_ptr = Q_ptr + z.to(tl.int64) * stride_QZ + h.to(tl.int64) * stride_QH
    K_ptr = K_ptr + z.to(tl.int64) * stride_KZ + h.to(tl.int64) * stride_KH
    V_ptr = V_ptr + z.to(tl.int64) * stride_VZ + h.to(tl.int64) * stride_VH
    O_ptr = O_ptr + z.to(tl.int64) * stride_OZ + h.to(tl.int64) * stride_OH

    # 8. Load Q_i into SRAM; will sty in SRAM throughout this block
    Q_i_ptrs = tl.make_block_ptr(
        base=Q_ptr,
        shape=(N, D),
        strides=(stride_QN, stride_QD),
        offsets=(i * B_r, 0),
        block_shape=(B_r, B_d),
        order=(0, 1),
    )
    Q_i = tl.load(Q_i_ptrs, boundary_check=(0, 1))  # [B_r, D]

    # Initialize local O_i, l_i, m_i for this block
    O_i = tl.zeros((B_r, B_d), dtype=Q_i.dtype)  # [B_r, D]
    l_i = tl.zeros((B_r,), dtype=Q_i.dtype)  # [B_r]
    m_i = tl.full((B_r,), -float("inf"), dtype=Q_i.dtype)  # [B_r]

    # 3. Divide K, V into T_c blocks of size [B_c, D] each
    T_c = tl.cdiv(N, B_c)

    # We only prepare the block pointer here; loading/adavancing will be done in the inner loop
    K_j_ptrs = tl.make_block_ptr(
        base=K_ptr,
        shape=(D, N),
        strides=(stride_KD, stride_KN),
        offsets=(0, 0),
        block_shape=(B_d, B_c),
        order=(1, 0),
    )  # NOTE: we are loading K_j^T, so the strides and order are swapped
    V_j_ptrs = tl.make_block_ptr(
        base=V_ptr,
        shape=(N, D),
        strides=(stride_VN, stride_VD),
        offsets=(0, 0),
        block_shape=(B_c, B_d),
        order=(0, 1),
    )

    # Inner loop (NOTE: in Algorithm 1, this is the outer loop; Algorithm 1's inner loop is the outer loop here via tl.program_id(0))
    for j in range(T_c):
        # 3. Divide K, V into T_c blocks of size [B_c, D] each
        # 6. Load K_j, V_j into SRAM
        K_j = tl.load(K_j_ptrs, boundary_check=(1, 0))  # [D, B_c] # NOTE: K_j is loaded in its transpose
        V_j = tl.load(V_j_ptrs, boundary_check=(0, 1))  # [B_c, D]

        # 9. On chip, compute S_ij = Q_i @ K_j^T
        S_ij = tl.dot(Q_i, K_j, allow_tf32=allow_tf32)  # [B_r, B_c] # NOTE: K_j is already loaded in its transpose

        # 9a. Scale by sqrt(d) (not in the paper, but part of the attention formula)
        S_ij = S_ij * softmax_scale

        # 9b. Mask out-of-bounds elements
        rows = j * B_c + tl.arange(0, B_c)
        S_ij = tl.where((rows[None, :] < N), S_ij, -float("inf"))

        # 10. On chip, compute mtilde_ij = rowmax(S_ij)
        mtilde_ij = tl.max(S_ij, axis=1)  # [B_r]

        # 10. On chip, compute Ptilde_ij = exp(S_ij - mtilde_ij)
        Ptilde_ij = tl.exp(S_ij - mtilde_ij[:, None])  # [B_r, B_c]

        # 11. On chip, compute ltilde_ij = rowsum(Ptilde_ij)
        ltilde_ij = tl.sum(Ptilde_ij, axis=1)  # [B_r]

        # 11. On chip, compute mnew_i = max(m_i, mtilde_ij)
        mnew_i = tl.maximum(m_i, mtilde_ij)  # [B_r]

        # 11. On chip, compute lnew_i = exp(m_i - mnew_i) * l_i + exp(mtilde_ij - mnew_i) * ltilde_ij
        alpha = tl.exp(m_i - mnew_i)  # [B_r]
        beta = tl.exp(mtilde_ij - mnew_i)  # [B_r]
        lnew_i = alpha * l_i + beta * ltilde_ij  # [B_r]

        # 12. Write O_i = diag(lnew_i)^-1 (diag(l_i) exp(m_i - mnew_i) O_i + exp(mtilde_ij - mnew_i) Ptilde_ij V_j) to HBM
        P_scale = beta / lnew_i  # [B_r]
        O_scale = l_i / lnew_i * alpha  # [B_r]
        O_i = O_i * O_scale[:, None] + tl.dot(Ptilde_ij * P_scale[:, None], V_j, allow_tf32=allow_tf32)

        # 13. Write l_i = lnew_i to HBM
        l_i = lnew_i

        # 13. Write m_i = mnew_i to HBM
        m_i = mnew_i

        # Advance block pointers to the next block
        K_j_ptrs = K_j_ptrs.advance((0, B_c))  # NOTE: K_j is loaded in its transpose
        V_j_ptrs = V_j_ptrs.advance((B_c, 0))

    # 12. Write O_i to HBM
    O_i_ptrs = tl.make_block_ptr(
        base=O_ptr,
        shape=(N, D),
        strides=(stride_ON, stride_OD),
        offsets=(i * B_r, 0),
        block_shape=(B_r, B_d),
        order=(0, 1),
    )
    tl.store(O_i_ptrs, O_i, boundary_check=(0, 1))


def triton_flash_attention(Q, K, V, allow_tf32=True, **kwargs):
    Z, H, N, D = Q.shape
    dtype = Q.dtype

    softmax_scale = 1.0 / D**0.5

    # 2. Initialize O, l, m in HBM
    O = torch.zeros(Z, H, N, D, device=Q.device, dtype=dtype)  # [N, d]

    B_d = triton.next_power_of_2(D)
    B_r = 128 if D <= 128 else 64
    B_c = 64 if D <= 64 else 32
    num_stages = 4 if D <= 64 else (3 if D <= 128 else 2)
    num_warps = 8

    # print(f"Using B_r={B_r}, B_c={B_c}, num_warps={num_warps}, num_stages={num_stages} for {Z=}, {H=}, {N=}, {D=}")

    B_r = kwargs.get("B_r", B_r)
    B_c = kwargs.get("B_c", B_c)
    num_warps = kwargs.get("num_warps", num_warps)
    num_stages = kwargs.get("num_stages", num_stages)

    grid = lambda meta: (cdiv(N, meta["B_r"]), Z * H)

    triton_flash_attention_kernel[grid](
        Q,
        K,
        V,
        O,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        Q.stride(3),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        V.stride(3),
        O.stride(0),
        O.stride(1),
        O.stride(2),
        O.stride(3),
        Z=Z,
        H=H,
        N=N,
        D=D,
        softmax_scale=softmax_scale,
        B_d=B_d,
        B_r=B_r,
        B_c=B_c,
        num_stages=num_stages,
        num_warps=num_warps,
        allow_tf32=allow_tf32,
    )
    return O


if __name__ == "__main__":
    embed_dim = 1024
    Z = 1
    H = 4
    N = 1024
    D = embed_dim // H

    Q = torch.randn(Z, H, N, D, device="cuda")
    K = torch.randn(Z, H, N, D, device="cuda")
    V = torch.randn(Z, H, N, D, device="cuda")

    from torch.utils import benchmark

    t = benchmark.Timer(
        setup="from __main__ import triton_flash_attention",
        stmt="triton_flash_attention(Q, K, V)",
        globals={"Q": Q, "K": K, "V": V},
        num_threads=1,
    )
    print(t.blocked_autorange(min_run_time=1.0))
