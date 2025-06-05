import time
import torch
import torch.nn.functional as F
from infllm_v2 import flash_attn_infllmv2_stage1

def naive_attention(q, k, v, causal=False):
    # 计算 attention
    k = k.repeat_interleave(q.shape[0] // k.shape[0], dim=0)
    v = v.repeat_interleave(q.shape[0] // v.shape[0], dim=0)
    # 计算 attention
    attn = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)
    if causal:
        causal_mask = torch.zeros(q.shape[1], k.shape[1], device=q.device).bool()
        for i in range(q.shape[1]):
            for j in range(k.shape[1]):
                if i >= (j * 16 + 32 - 1):
                    causal_mask[i, j] = True
        attn = attn.masked_fill(~causal_mask, -float('inf'))
    score = F.softmax(attn, dim=-1)
    score = score.reshape(2, 16, q.shape[1], k.shape[1]).sum(dim=1)
    return score

def test_flash_attn_varlen(seqlen_q=256, seqlen_k=16, n_heads=32, n_kv_heads=2, head_dim=128, dtype=torch.float16, bench=False, causal=False):
    q = torch.randn(n_heads, seqlen_q, head_dim, dtype=dtype).cuda()
    k = torch.randn(n_kv_heads, seqlen_k, head_dim, dtype=dtype).cuda()
    v = torch.randn(n_kv_heads, seqlen_k, head_dim, dtype=dtype).cuda()
    cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32).cuda()
    cu_seqlens_k = torch.tensor([0, seqlen_k], dtype=torch.int32).cuda()

    # 朴素实现
    if not bench:
        naive_score = naive_attention(q, k, v, causal=causal)

    q = q.transpose(0, 1).contiguous().clone()
    k = k.transpose(0, 1).contiguous().clone()
    v = v.transpose(0, 1).contiguous().clone()

    flash_score = flash_attn_infllmv2_stage1(
        q,
        k,
        torch.tensor([[[1]]], dtype=q.dtype, device=q.device),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        causal=causal,
    )

    if bench:
        f = lambda : flash_attn_infllmv2_stage1(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k, return_attn_probs=True, causal=causal)
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        st = time.time()
        for _ in range(10):
            f()
        torch.cuda.synchronize()
        et = time.time()
        print(f"seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, causal: {causal}")
        print(f"flash_attn_infllmv2_stage1 time: {(et - st) / 10 * 1000} ms")
        f = lambda : flash_attn_infllmv2_stage1(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k, return_attn_probs=False, causal=causal)
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        st = time.time()
        for _ in range(10):
            f()
        torch.cuda.synchronize()
        et = time.time()
        print(f"flash_attn_infllmv2_stage1 time (no return_attn_probs): {(et - st) / 10 * 1000} ms")
    else:
        flash_score = flash_score[:, :seqlen_q, :seqlen_k]
        if causal:
            assert (flash_score[:, :31] == float('-inf')).all()
            naive_score = naive_score[:, 31:]
            flash_score = flash_score[:, 31:]
        # print(naive_score.shape, flash_score.shape)

        # print("score mean diff:", (naive_score - flash_score).abs().mean())
        print(f"{seqlen_q=} {seqlen_k=} {causal=}")
        print("score max diff :", (naive_score - flash_score).abs().max())
        if (naive_score - flash_score).abs().max() > 1e-2:
            print("error: ", seqlen_q, seqlen_k)

if __name__ == "__main__":
    for seqlen in (list(range(1,100))+list(range(100, 1000, 100))+list(range(1000, 10000, 1000))):
        test_flash_attn_varlen(seqlen_q=1, seqlen_k=seqlen, causal=False)
    for seqlen in (list(range(100, 1000, 100))+list(range(1000, 10000, 1000))):
        test_flash_attn_varlen(seqlen_q=seqlen, seqlen_k=seqlen//16, causal=True)
    # test_flash_attn_varlen(seqlen_q=10000, seqlen_k=10000//16, causal=False)
    # test_flash_attn_varlen(seqlen_q=10000, seqlen_k=10000//16, causal=True)
    # test_flash_attn_varlen(seqlen_q=31235, seqlen_k=31235//16, causal=False)
    # test_flash_attn_varlen(seqlen_q=31235, seqlen_k=31235//16, causal=True)
    # test_flash_attn_varlen(seqlen_q=16384, seqlen_k=16384//16, bench=True)
    # test_flash_attn_varlen(seqlen_q=32768, seqlen_k=32768//16, bench=True)
    # test_flash_attn_varlen(seqlen_q=131072, seqlen_k=131072//16, bench=True)
    # test_flash_attn_varlen(seqlen_q=131072, seqlen_k=131072//16, bench=True, causal=True)
