import torch


def cdiv(a, b):
    return (a + b - 1) // b


@torch.no_grad()
def torch_flash_attention_kernel(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, B_r: int = 128, B_c: int = 128):
    """Flash attention kernel implementation using torch operations.

    This implementation closely follows Algorithm 1 in the FlashAttention paper: https://arxiv.org/pdf/2205.14135.pdf.
    The only difference is that we perform the attention scaling by sqrt(d) as part of the computation.

    This implementation is not intended to be used; it is only for reference and testing purposes.

    Args:
        Q: Queries tensor of shape [N, d]
        K: Keys tensor of shape [N, d]
        V: Values tensor of shape [N, d]
        B_r: The block size for the rows
        B_c: The block size for the columns
    """

    N, d = Q.shape
    dtype = Q.dtype

    # 2. Initialize O, l, m in HBM
    O = torch.zeros(N, d, device=Q.device, dtype=dtype)  # [N, d]
    l = torch.zeros(N, device=Q.device, dtype=dtype)  # [N]
    m = torch.full((N,), float("-inf"), device=Q.device, dtype=dtype)  # [N]

    # 3. Divide Q into T_r blocks of size [B_r, d] each
    T_r = cdiv(N, B_r)
    Q = list(torch.split(Q, B_r))  # [T_r, B_r, d]

    # 3. Divide K, V into T_c blocks of size [B_c, d] each
    T_c = cdiv(N, B_c)
    K = list(torch.split(K, B_c))  # [T_c, B_c, d]
    V = list(torch.split(V, B_c))  # [T_c, B_c, d]

    # 4. Divide O into T_r blocks of size [B_r, d] each
    O = list(torch.split(O, B_r))  # [T_r, B_r, d]

    # 4. Divide l into T_r blocks of size [B_r] each
    l = list(torch.split(l, B_r))  # [T_r, B_r]

    # 4. Divide m into T_r blocks of size [B_r] each
    m = list(torch.split(m, B_r))  # [T_r, B_r]

    # 5. Outer loop
    for j in range(T_c):
        # 6. Load K_j, V_j into SRAM
        K_j = K[j]  # [B_c, d]
        V_j = V[j]  # [B_c, d]

        # 7. Inner loop
        for i in range(T_r):
            # 8. Load Q_i, O_i, l_i, m_i into SRAM
            Q_i = Q[i]  # [B_r, d]
            O_i = O[i]  # [B_r, d]
            l_i = l[i]  # [B_r]
            m_i = m[i]  # [B_r]

            # 9. On chip, compute S_ij = Q_i @ K_j^T
            S_ij = Q_i @ K_j.T  # [B_r, B_c]

            # 9a. Scale by sqrt(d) (not in the paper, but part of the attention formula)
            S_ij = S_ij / (d**0.5)

            # 10. On chip, compute mtilde_ij = rowmax(S_ij)
            mtilde_ij = S_ij.max(dim=1).values  # [B_r]

            # 10. On chip, compute Ptilde_ij = exp(S_ij - mtilde_ij)
            Ptilde_ij = torch.exp(S_ij - mtilde_ij.unsqueeze(1))  # [B_r, B_c]

            # 11. On chip, compute ltilde_ij = rowsum(Ptilde_ij)
            ltilde_ij = Ptilde_ij.sum(dim=1)  # [B_r]

            # 11. On chip, compute mnew_i = max(m_i, mtilde_ij)
            mnew_i = torch.maximum(m_i, mtilde_ij)  # [B_r]

            # 11. On chip, compute lnew_i = exp(m_i - mnew_i) * l_i + exp(mtilde_ij - mnew_i) * ltilde_ij
            lnew_i = torch.exp(m_i - mnew_i) * l_i + torch.exp(mtilde_ij - mnew_i) * ltilde_ij  # [B_r]

            # 12. Write O_i = diag(lnew_i)^-1 (diag(l_i) exp(m_i - mnew_i) O_i + exp(mtilde_ij - mnew_i) Ptilde_ij V_j) to HBM
            #           O_i = a @ (b + c)
            #           where:
            #             a = diag(lnew_i) ** -1                             [B_r, B_r]
            #             b = diag(l_i) * exp(m_i - mnew_i) @ O_i            [B_r, d]
            #                 [B_r, B_r]  [B_r]               [B_r, d]
            #             c = exp(mtilde_ij - mnew_i) * Ptilde_ij @ V_j      [B_r, d]
            #                 [B_r]                     [B_r, B_c]  [B_c, d]
            _a = torch.diag(lnew_i**-1)  # [B_r, B_r]
            _b = torch.diag(l_i) * torch.exp(m_i - mnew_i) @ O_i  # [B_r, d]
            _c = torch.exp(mtilde_ij - mnew_i).unsqueeze(1) * Ptilde_ij @ V_j
            O_i = _a @ (_b + _c)
            O[i] = O_i  # write to HBM

            # 13. Write l_i = lnew_i to HBM
            l_i = lnew_i
            l[i] = l_i  # write to HBM

            # 13. Write m_i = mnew_i to HBM
            m_i = mnew_i
            m[i] = m_i  # write to HBM

    O = torch.cat(O)
    return O
