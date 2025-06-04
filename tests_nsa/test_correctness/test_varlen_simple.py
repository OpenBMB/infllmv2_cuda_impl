import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func

def naive_attention(q, k, v, blockmask):
    # 计算 attention
    k = k.repeat_interleave(q.shape[0] // k.shape[0], dim=0)
    v = v.repeat_interleave(q.shape[0] // v.shape[0], dim=0)
    # 计算 attention
    attn = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)
    if blockmask is not None:
        attn = attn.masked_fill(~blockmask, -float('inf'))
    attn_weights = F.softmax(attn, dim=-1)
    output = attn_weights @ v
    return output

def test_flash_attn_varlen(seq_len_q=100, seq_len_k=100, ratio=1., n_heads=32, n_kv_heads=2, head_dim=128, dtype=torch.float16):
    q = torch.randn(n_heads, seq_len_q, head_dim, dtype=dtype).cuda()
    k = torch.randn(n_kv_heads, seq_len_k, head_dim, dtype=dtype).cuda()
    v = torch.randn(n_kv_heads, seq_len_k, head_dim, dtype=dtype).cuda()
    cu_seqlens_q = torch.tensor([0, seq_len_q], dtype=torch.int32).cuda()
    cu_seqlens_k = torch.tensor([0, seq_len_k], dtype=torch.int32).cuda()

    if ratio < 1:
        num_k_blocks = (seq_len_k + 63) // 64
        num_act = int(num_k_blocks * ratio)
        topk_idx = torch.zeros((n_kv_heads, seq_len_q, num_act), dtype=torch.int32).cuda()
        for h in range(n_kv_heads):
            for s in range(seq_len_q):
                topk_idx[h, s, :] = torch.randperm(num_k_blocks)[:num_act]
        blockmask = torch.zeros((n_kv_heads, seq_len_q, seq_len_k), dtype=torch.bool).cuda()
        for h in range(n_kv_heads):
            for s in range(seq_len_q):
                for idx in topk_idx[h, s]:
                    blockmask[h, s, idx * 64: (idx + 1) * 64] = 1
        blockmask = blockmask.repeat_interleave(q.shape[0] // blockmask.shape[0], dim=0)
    else:
        topk_idx = None
        blockmask = None
    
    # 朴素实现
    naive_out = naive_attention(q, k, v, blockmask)

    naive_out = naive_out.transpose(0, 1).contiguous().clone()
    q = q.transpose(0, 1).contiguous().clone()
    k = k.transpose(0, 1).contiguous().clone()
    v = v.transpose(0, 1).contiguous().clone()

    print(topk_idx)

    flash_out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=seq_len_q,
        max_seqlen_k=seq_len_k,
        topk_idx=topk_idx,
        causal=False, # TODO
    )

    print(naive_out.shape, flash_out.shape)
    print(naive_out)
    print(flash_out)
    
    print("mean diff:", (naive_out - flash_out).abs().mean())
    print("max diff :", (naive_out - flash_out).abs().max())

if __name__ == "__main__":
    test_flash_attn_varlen(seq_len_q=1024, seq_len_k=1024, ratio=0.5)
