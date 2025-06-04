import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import _generate_block_kvcache, generate_random_padding_mask, generate_qkv

def naive_attention(q, k, v, blockmask):
    # 计算 attention
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    # 计算 attention
    attn = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)
    if blockmask is not None:
        attn = attn.masked_fill(~blockmask, -float('inf'))
    attn_weights = F.softmax(attn, dim=-1)
    output = attn_weights @ v
    return output

def test_flash_attn_varlen(batch_size=8, seqlen_q=100, seqlen_k=100, ratio=1., n_heads=32, n_kv_heads=2, head_dim=128, dtype=torch.float16):
    q = torch.randn(batch_size, seqlen_q, n_heads, head_dim, device="cuda", dtype=dtype)
    k, v, block_table, k_cache_paged, v_cache_paged, num_blocks = _generate_block_kvcache(
        seqlen_k=seqlen_k, paged_kv_block_size=256, batch_size=batch_size, nheads_k=n_kv_heads, d=head_dim, device="cuda", dtype=dtype
    )
    query_padding_mask = None #generate_random_padding_mask(seqlen_q, batch_size, "cuda", mode="random")
    key_padding_mask = None #generate_random_padding_mask(seqlen_k, batch_size, "cuda", mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    if ratio < 1:
        num_k_blocks = (seqlen_k + 63) // 64
        num_act = int(num_k_blocks * ratio)
        topk_idx = torch.zeros((n_kv_heads, batch_size * seqlen_q, num_act), dtype=torch.int32).cuda()
        for h in range(n_kv_heads):
            for s in range(batch_size * seqlen_q):
                topk_idx[h, s, :] = torch.randperm(num_k_blocks)[:num_act]
        blockmask = torch.zeros((n_kv_heads, batch_size * seqlen_q, seqlen_k), dtype=torch.bool).cuda()
        for h in range(n_kv_heads):
            for s in range(batch_size * seqlen_q):
                for idx in topk_idx[h, s]:
                    blockmask[h, s, idx * 64: (idx + 1) * 64] = 1
        blockmask = blockmask.repeat_interleave(q.shape[2] // blockmask.shape[0], dim=0)
        blockmask = blockmask.reshape(n_heads, batch_size, seqlen_q, seqlen_k).transpose(0, 1).contiguous().clone()
    else:
        topk_idx = None
        blockmask = None
    
    # 朴素实现
    naive_out = naive_attention(
        q.transpose(1, 2).contiguous().clone(),
        k.transpose(1, 2).contiguous().clone(),
        v.transpose(1, 2).contiguous().clone(),
        blockmask,
    )
    naive_out = naive_out.transpose(1, 2).contiguous().clone()

    out_unpad = flash_attn_varlen_func(
        q_unpad,
        k_cache_paged,
        v_cache_paged,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        causal=False,
        block_table=block_table,
        topk_idx=topk_idx,
    )
    flash_out = output_pad_fn(out_unpad)

    # print(naive_out.shape, flash_out.shape)
    # print(naive_out)
    # print(flash_out)

    print(f"{batch_size=}, {seqlen_q=}, {seqlen_k=}, {ratio=}")
    print("mean diff:", (naive_out - flash_out).abs().mean())
    print("max diff :", (naive_out - flash_out).abs().max())

if __name__ == "__main__":
    test_flash_attn_varlen(batch_size=1, seqlen_q=1024, seqlen_k=1024, ratio=0.5)
    test_flash_attn_varlen(batch_size=1, seqlen_q=1024, seqlen_k=1024, ratio=1)
    test_flash_attn_varlen(batch_size=1, seqlen_q=1024, seqlen_k=1024, ratio=0.5)
    test_flash_attn_varlen(batch_size=1, seqlen_q=1024, seqlen_k=2048, ratio=0.5)
    test_flash_attn_varlen(batch_size=1, seqlen_q=1500, seqlen_k=1500, ratio=0.2)
