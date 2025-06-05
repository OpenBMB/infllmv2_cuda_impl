import torch
import torch.nn.functional as F
from infllm_v2 import infllmv2_attn_with_kvcache

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

def test_infllmv2_attn_with_kvcache(batch_size=1, seq_len=1, cache_len=100, ratio=1., n_heads=32, n_kv_heads=2, head_dim=128, dtype=torch.bfloat16):
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype).cuda()
    k_cache = torch.randn(batch_size, n_kv_heads, cache_len, head_dim, dtype=dtype).cuda()
    v_cache = torch.randn(batch_size, n_kv_heads, cache_len, head_dim, dtype=dtype).cuda()
    k = torch.randn(batch_size, n_kv_heads, seq_len, head_dim, dtype=dtype).cuda()
    v = torch.randn(batch_size, n_kv_heads, seq_len, head_dim, dtype=dtype).cuda()
    k_pad = torch.randn(batch_size, n_kv_heads, cache_len + seq_len, head_dim, dtype=dtype).cuda()
    v_pad = torch.randn(batch_size, n_kv_heads, cache_len + seq_len, head_dim, dtype=dtype).cuda()
    k_pad[:, :, :cache_len, :] = k_cache
    v_pad[:, :, :cache_len, :] = v_cache
    cache_seqlens = torch.full((batch_size,), cache_len, dtype=torch.int32).cuda()

    if ratio < 1:
        num_k_blocks = (cache_len + seq_len + 63) // 64
        num_act = int(num_k_blocks * ratio)
        topk_idx = torch.zeros((batch_size, n_kv_heads, seq_len, num_act), dtype=torch.int32).cuda()
        for b in range(batch_size):
            for h in range(n_kv_heads):
                for s in range(seq_len):
                    topk_idx[b, h, s, :] = torch.randperm(num_k_blocks)[:num_act]
        blockmask = torch.zeros((batch_size, n_kv_heads, seq_len, cache_len + seq_len), dtype=torch.bool).cuda()
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
    naive_out = naive_attention(q, torch.cat([k_cache, k], dim=2), torch.cat([v_cache, v], dim=2), blockmask)

    naive_out = naive_out.transpose(1, 2).contiguous().clone()
    q = q.transpose(1, 2).contiguous().clone()
    k_pad = k_pad.transpose(1, 2).contiguous().clone()
    v_pad = v_pad.transpose(1, 2).contiguous().clone()
    k = k.transpose(1, 2).contiguous().clone()
    v = v.transpose(1, 2).contiguous().clone()

    print(topk_idx)

    flash_out = infllmv2_attn_with_kvcache(
        q,
        k_pad,
        v_pad,
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        topk_idx=topk_idx,
    )

    print(naive_out.shape, flash_out.shape)
    print(naive_out)
    print(flash_out)
    
    print("mean diff:", (naive_out - flash_out).abs().mean())
    print("max diff :", (naive_out - flash_out).abs().max())

if __name__ == "__main__":
    test_infllmv2_attn_with_kvcache(cache_len=10000, ratio=0.5)
