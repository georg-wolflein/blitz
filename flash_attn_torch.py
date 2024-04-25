import torch
from typing import Optional


def cdiv(a, b):
    return (a + b - 1) // b


def pad_to(x, dim, size, value=0.0):
    """Append padding to the input tensor x to match the target size along the given dimension."""
    pad_size = size - x.size(dim)
    if pad_size > 0:
        pad_dims = list(x.shape)
        pad_dims[dim] = pad_size
        pad = torch.full(pad_dims, value, dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad], dim=dim)
    return x


@torch.no_grad()
def torch_flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    softmax_scale: Optional[float] = None,
    B_r: int = 128,
    B_c: int = 128,
):
    """Flash attention kernel implementation using torch operations.

    This implementation closely follows Algorithm 1 in the FlashAttention paper: https://arxiv.org/pdf/2205.14135.pdf.
    The only difference is that we perform the attention scaling by sqrt(d) as part of the computation.

    This implementation is not intended to be used; it is only for reference and testing purposes.

    Args:
        Q: Queries tensor of shape [Z, H, N, D]
        K: Keys tensor of shape [Z, H, N, D]
        V: Values tensor of shape [Z, H, N, D]
        B_r: The block size for the rows
        B_c: The block size for the columns
    """

    Z, H, N, D = Q.shape
    dtype = Q.dtype
    device = Q.device

    softmax_scale = softmax_scale or 1.0 / (D**0.5)

    def inner(Q, K, V):

        # 2. Initialize O in HBM
        O = torch.zeros(N, D, device=device, dtype=dtype)  # [N, D]

        # 3. Divide Q into T_r blocks of size [B_r, D] each
        T_r = cdiv(N, B_r)
        Q = list(torch.split(Q, B_r))  # [T_r, B_r, D]

        # 3. Divide K, V into T_c blocks of size [B_c, d] each
        T_c = cdiv(N, B_c)
        K = list(torch.split(K, B_c))  # [T_c, B_c, D]
        V = list(torch.split(V, B_c))  # [T_c, B_c, D]

        # 4. Divide O into T_r blocks of size [B_r, D] each
        O = list()

        # 7. Outer loop (NOTE: in Algorithm 1, this is the inner loop)
        for i in range(T_r):
            # 8. Load Q_i, O_i, l_i, m_i into SRAM
            Q_i = Q[i]  # [B_r, D]
            Q_i = pad_to(Q_i, 0, B_r)  # simulate padding

            # 2. and 4. Divide l, m into T_r blocks of size [B_r] each
            l_i = torch.zeros(B_r, device=device, dtype=dtype)  # [B_r]
            m_i = torch.full((B_r,), float("-inf"), device=device, dtype=dtype)  # [B_r]
            O_i = torch.zeros(B_r, D, device=device, dtype=dtype)  # [B_r, D]

            # 5. Inner loop (NOTE: in Algorithm 1, this is the outer loop)
            for j in range(T_c):
                # 6. Load K_j, V_j into SRAM
                K_j = K[j]  # [B_c, d]
                V_j = V[j]  # [B_c, d]

                K_j = pad_to(K_j, 0, B_c)  # simulate padding
                V_j = pad_to(V_j, 0, B_c)  # simulate padding

                # 9. On chip, compute S_ij = Q_i @ K_j^T
                S_ij = Q_i @ K_j.T  # [B_r, B_c]

                # 9a. Scale by sqrt(d) (not in the paper, but part of the attention formula)
                S_ij = S_ij * softmax_scale

                # 9b. Mask out-of-bounds elements
                S_ij = torch.where(torch.arange(B_c, device=device).unsqueeze(0) + j * B_c < N, S_ij, -float("inf"))

                # 10. On chip, compute mtilde_ij = rowmax(S_ij)
                mtilde_ij = S_ij.max(dim=1).values  # [B_r]

                # 10. On chip, compute Ptilde_ij = exp(S_ij - mtilde_ij)
                Ptilde_ij = torch.exp(S_ij - mtilde_ij.unsqueeze(1))  # [B_r, B_c]

                # 11. On chip, compute ltilde_ij = rowsum(Ptilde_ij)
                ltilde_ij = Ptilde_ij.sum(dim=1)  # [B_r]

                # 11. On chip, compute mnew_i = max(m_i, mtilde_ij)
                mnew_i = torch.maximum(m_i, mtilde_ij)  # [B_r]

                # 11. On chip, compute lnew_i = exp(m_i - mnew_i) * l_i + exp(mtilde_ij - mnew_i) * ltilde_ij
                alpha = torch.exp(m_i - mnew_i)  # [B_r]
                beta = torch.exp(mtilde_ij - mnew_i)  # [B_r]
                lnew_i = alpha * l_i + beta * ltilde_ij  # [B_r]

                # 12. Write O_i = diag(lnew_i)^-1 (diag(l_i) exp(m_i - mnew_i) O_i + exp(mtilde_ij - mnew_i) Ptilde_ij V_j) to HBM
                P_scale = beta / lnew_i  # [B_r]
                O_scale = l_i / lnew_i * alpha  # [B_r]
                O_i = O_i * O_scale.unsqueeze(1) + (Ptilde_ij * P_scale.unsqueeze(1)) @ V_j

                # 13. Write l_i = lnew_i to HBM
                l_i = lnew_i

                # 13. Write m_i = mnew_i to HBM
                m_i = mnew_i

            O.append(O_i)  # write to HBM

        O = torch.cat(O)
        O = O[:N]  # remove padding
        return O

    # Run inner across Z, H dimensions
    O = torch.stack([torch.stack([inner(Q[z, h], K[z, h], V[z, h]) for h in range(H)]) for z in range(Z)])
    return O
