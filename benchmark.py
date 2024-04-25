from torch.utils import benchmark
from tqdm import tqdm
import torch

if __name__ == "__main__":

    shapes = [
        (Z, H, N, embed_dim // H)
        for embed_dim in (512, 1024, 2048)
        for Z in (1, 2, 4)
        for H in (4, 8, 16)
        for N in (1024, 2048, 4096)
    ]

    results = []
    label = "attention"
    min_run_time = 1.0

    for Z, H, N, D in (pbar := tqdm(shapes, desc="Benchmarking")):
        Q = torch.randn(Z, H, N, D, device="cuda")
        K = torch.randn(Z, H, N, D, device="cuda")
        V = torch.randn(Z, H, N, D, device="cuda")
        sub_label = f"[{Z}, {H}, {N}, {D}]"
        pbar.set_postfix_str(sub_label)

        t = benchmark.Timer(
            setup="from flash_attn_triton import triton_flash_attention",
            stmt="triton_flash_attention(Q, K, V)",
            globals={"Q": Q, "K": K, "V": V},
            num_threads=1,
            description="triton (flash)",
            label=label,
            sub_label=sub_label,
        )
        results.append(t.blocked_autorange(min_run_time=min_run_time))

        t = benchmark.Timer(
            setup="from torch.nn.functional import scaled_dot_product_attention",
            stmt="scaled_dot_product_attention(Q, K, V)",
            globals={"Q": Q, "K": K, "V": V},
            num_threads=1,
            description="scaled_dot_product_attention",
            label=label,
            sub_label=sub_label,
        )

        results.append(t.blocked_autorange(min_run_time=min_run_time))
        t = benchmark.Timer(
            setup="from attn_torch import torch_attention",
            stmt="torch_attention(Q, K, V)",
            globals={"Q": Q, "K": K, "V": V},
            num_threads=1,
            description="torch",
            label=label,
            sub_label=sub_label,
        )
        results.append(t.blocked_autorange(min_run_time=min_run_time))

        compare = benchmark.Compare(results)
        compare.print()
    compare = benchmark.Compare(results)
    compare.print()
