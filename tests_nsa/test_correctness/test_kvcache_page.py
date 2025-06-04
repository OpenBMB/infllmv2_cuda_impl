import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import _generate_block_kvcache
from flash_attn import flash_attn_with_kvcache

def naive_attention(q, k_full, v_full, blockmask):
    # 计算 attention
    k_full = k_full.repeat_interleave(q.shape[1] // k_full.shape[1], dim=1)
    v_full = v_full.repeat_interleave(q.shape[1] // v_full.shape[1], dim=1)
    # 计算 attention
    attn = q @ k_full.transpose(-2, -1) / (q.size(-1) ** 0.5)
    if blockmask is not None:
        attn = attn.masked_fill(~blockmask, -float('inf'))
    attn_weights = F.softmax(attn, dim=-1)
    output = attn_weights @ v_full
    return output


def test_flash_attn_with_kvcache(
    batch_size=1,
    seq_len=1,
    cache_len=100,
    ratio=1.0,
    n_heads=32,
    n_kv_heads=2,
    head_dim=128,
    dtype=torch.float16,
):
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype).cuda()
    cache_seqlens = torch.full((batch_size,), cache_len, dtype=torch.int32).cuda()

    (
        k_cache,
        v_cache,
        block_table,
        k_cache_paged,
        v_cache_paged,
        num_blocks,
    ) = _generate_block_kvcache(
        seqlen_k=cache_len, paged_kv_block_size=256, batch_size=batch_size, nheads_k=n_kv_heads, d=head_dim, device="cuda", dtype=dtype
    )

    if ratio < 1:
        num_k_blocks = (cache_len + 63) // 64
        num_act = int(num_k_blocks * ratio)
        topk_idx = torch.zeros((batch_size, n_kv_heads, seq_len, num_act), dtype=torch.int32).cuda()
        for b in range(batch_size):
            for h in range(n_kv_heads):
                for s in range(seq_len):
                    topk_idx[b, h, s, :] = torch.randperm(num_k_blocks)[:num_act]
        blockmask = torch.zeros((batch_size, n_kv_heads, seq_len, cache_len), dtype=torch.bool).cuda()
        for b in range(batch_size):
            for h in range(n_kv_heads):
                for s in range(seq_len):
                    for idx in topk_idx[b, h, s]:
                        blockmask[b, h, s, idx * 64: (idx + 1) * 64] = 1
        blockmask = blockmask.repeat_interleave(q.shape[1] // blockmask.shape[1], dim=1)
    else:
        topk_idx = None
        blockmask = None
    
    # 朴素实现
    naive_out = naive_attention(
        q, k_cache.transpose(1, 2).contiguous().clone(), v_cache.transpose(1, 2).contiguous().clone(), blockmask
    )

    naive_out = naive_out.transpose(1, 2).contiguous().clone()
    q = q.transpose(1, 2).contiguous().clone()

    # print(topk_idx)

    flash_out = flash_attn_with_kvcache(
        q,
        k_cache_paged,
        v_cache_paged,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        topk_idx=topk_idx,
    )

    # print(naive_out.shape, flash_out.shape)
    #print(naive_out)
    #print(flash_out)
    
    print(f"{cache_len=}, {ratio=}")
    print("mean diff:", (naive_out - flash_out).abs().mean())
    print("max diff :", (naive_out - flash_out).abs().max())

if __name__ == "__main__":
    test_flash_attn_with_kvcache(cache_len=500, ratio=0.5)
    test_flash_attn_with_kvcache(cache_len=500, ratio=1.0)
    test_flash_attn_with_kvcache(cache_len=1024, ratio=0.5)
    test_flash_attn_with_kvcache(cache_len=1024, ratio=1.0)
    test_flash_attn_with_kvcache(cache_len=10000, ratio=0.5)
    test_flash_attn_with_kvcache(cache_len=10000, ratio=1.0)
    test_flash_attn_with_kvcache(cache_len=100000, ratio=0.5)
    test_flash_attn_with_kvcache(cache_len=100000, ratio=1.0)