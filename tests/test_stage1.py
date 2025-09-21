import time
import torch
import torch.nn.functional as F
from infllm_v2 import infllmv2_attn_stage1

def round_multiple(x, m):
    return (x + m - 1) // m * m

def naive_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=False):
    # 将 varlen 输入转换为 padded 形式
    batch_size = len(cu_seqlens_q) - 1
    max_seqlen_q = max(cu_seqlens_q[i+1] - cu_seqlens_q[i] for i in range(batch_size))
    max_seqlen_k = max(cu_seqlens_k[i+1] - cu_seqlens_k[i] for i in range(batch_size))
    
    # 创建 padded 张量
    q_padded = torch.zeros(q.shape[0], batch_size, max_seqlen_q, q.shape[-1], device=q.device, dtype=q.dtype)
    k_padded = torch.zeros(k.shape[0], batch_size, max_seqlen_k, k.shape[-1], device=k.device, dtype=k.dtype)
    v_padded = torch.zeros(v.shape[0], batch_size, max_seqlen_k, v.shape[-1], device=v.device, dtype=v.dtype)
    
    # 填充数据
    for i in range(batch_size):
        q_start = cu_seqlens_q[i]
        q_end = cu_seqlens_q[i + 1]
        k_start = cu_seqlens_k[i]
        k_end = cu_seqlens_k[i + 1]
        q_padded[:, i, :q_end-q_start] = q[:, q_start:q_end]
        k_padded[:, i, :k_end-k_start] = k[:, k_start:k_end]
        v_padded[:, i, :k_end-k_start] = v[:, k_start:k_end]
    
    # 计算 attention
    k_padded = k_padded.repeat_interleave(q_padded.shape[0] // k_padded.shape[0], dim=0)
    v_padded = v_padded.repeat_interleave(q_padded.shape[0] // v_padded.shape[0], dim=0)
    
    attn = q_padded @ k_padded.transpose(-2, -1) / (q_padded.size(-1) ** 0.5)
    
    if causal:
        causal_mask = torch.zeros(batch_size, max_seqlen_q, max_seqlen_k, device=q.device).bool()
        for b in range(batch_size):
            for i in range(max_seqlen_q):
                for j in range(max_seqlen_k):
                    if i >= (j * 16 + 32 - 1):
                        causal_mask[b, i, j] = True
        attn = attn.masked_fill(~causal_mask, -float('inf'))
    
    score = F.softmax(attn, dim=-1)
    score = score.reshape(2, 16, batch_size, max_seqlen_q, max_seqlen_k).sum(dim=1)
    
    # 将结果转回 varlen 形式
    result = []
    for i in range(batch_size):
        q_start = cu_seqlens_q[i]
        q_end = cu_seqlens_q[i + 1]
        k_start = cu_seqlens_k[i]
        k_end = cu_seqlens_k[i + 1]
        
        # 创建填充了 0 的张量
        curr_score = torch.full((2, q_end-q_start, max_seqlen_k), 0, device=q.device, dtype=q.dtype)
        # 填充实际的值
        curr_score[:, :, :k_end-k_start] = score[:, i, :q_end-q_start, :k_end-k_start]
        result.append(curr_score)
    
    final_result = torch.cat(result, dim=1)
    # 将 nan 值替换为 -inf
    final_result = torch.where(torch.isnan(final_result), 0, final_result)
    return final_result

def test_flash_attn_varlen(seqlen_q=256, seqlen_k=16, n_heads=32, n_kv_heads=2, head_dim=128, dtype=torch.bfloat16, bench=False, causal=False, batch_size=2):
    # 生成不同长度的序列
    seqlen_qs = [seqlen_q // 2, seqlen_q]  # 两个序列，长度不同
    seqlen_ks = [seqlen_k // 2, seqlen_k]  # k 也使用不同长度
    total_seqlen_q = sum(seqlen_qs)
    total_seqlen_k = sum(seqlen_ks)
    
    # 准备输入数据
    q = torch.randn(n_heads, total_seqlen_q, head_dim, dtype=dtype).cuda()
    k = torch.randn(n_kv_heads, total_seqlen_k, head_dim, dtype=dtype).cuda()
    # v = torch.randn(n_kv_heads, total_seqlen_k, head_dim, dtype=dtype).cuda()
    v = k.clone()
    # 计算累积序列长度
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device='cuda')
    cu_seqlens_v = torch.zeros(batch_size + 1, dtype=torch.int32, device='cuda')
    for i in range(batch_size):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + seqlen_qs[i]
        cu_seqlens_k[i + 1] = cu_seqlens_k[i] + seqlen_ks[i]
        cu_seqlens_v[i + 1] = cu_seqlens_v[i] + seqlen_ks[i]  # v has same lengths as k

    # 朴素实现
    if not bench:
        naive_score = naive_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal)

    q = q.transpose(0, 1).contiguous().clone()
    k = k.transpose(0, 1).contiguous().clone()
    v = v.transpose(0, 1).contiguous().clone()
    print(f"cu_seqlens_k: {cu_seqlens_k}")
    print(f"cu_seqlens_v: {cu_seqlens_v}")
    flash_score = infllmv2_attn_stage1(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_v=cu_seqlens_v,
        max_seqlen_q=max(seqlen_qs),
        max_seqlen_k=max(seqlen_ks),
        causal=causal,
    )

    if bench:
        f = lambda : infllmv2_attn_stage1(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max(seqlen_qs), max_seqlen_k=max(seqlen_ks), return_attn_probs=True, causal=causal)
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        st = time.time()
        for _ in range(10):
            f()
        torch.cuda.synchronize()
        et = time.time()
        print(f"seqlen_q: {seqlen_qs}, seqlen_k: {seqlen_ks}, causal: {causal}")
        print(f"infllmv2_attn_stage1 time: {(et - st) / 10 * 1000} ms")
        f = lambda : infllmv2_attn_stage1(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max(seqlen_qs), max_seqlen_k=max(seqlen_ks), return_attn_probs=False, causal=causal)
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        st = time.time()
        for _ in range(10):
            f()
        torch.cuda.synchronize()
        et = time.time()
        print(f"infllmv2_attn_stage1 time (no return_attn_probs): {(et - st) / 10 * 1000} ms")
    else:
        flash_score = flash_score[:, :total_seqlen_q, :total_seqlen_k]
        
        if causal:
            # 检查每个序列的前31个位置
            for i in range(batch_size):
                start_idx = cu_seqlens_q[i]
                end_idx = min(cu_seqlens_q[i] + 31, cu_seqlens_q[i + 1])
                # breakpoint()
                # if end_idx > start_idx:  # 只有当序列长度大于31时才检查
                #     assert (flash_score[ :, start_idx: start_idx + 32] == float('-inf')).all(), f"Sequence {i} causal mask check failed"
        print(f"{seqlen_q=} {seqlen_k=} {causal=}")
        print("score max diff :", (naive_score - flash_score).abs().max())
        
        # breakpoint()
        if (naive_score - flash_score).abs().max() > 1e-2:
            print(f"error: seqlen_qs={seqlen_qs}, seqlen_ks={seqlen_ks}")

if __name__ == "__main__":
    # Test 5 cases for causal=False
    test_seqlens = [1000, 5000, 9000]
    # test_seqlens = [2048]
    for seqlen in test_seqlens:
        test_flash_attn_varlen(seqlen_q=1, seqlen_k=seqlen, causal=False)
    
    # Test 5 cases for causal=True
    for seqlen in test_seqlens:
        test_flash_attn_varlen(seqlen_q=seqlen, seqlen_k=seqlen//16, causal=False)

    # test_flash_attn_varlen(seqlen_q=10000, seqlen_k=10000//16, causal=False)
    # test_flash_attn_varlen(seqlen_q=10000, seqlen_k=10000//16, causal=True)
    # test_flash_attn_varlen(seqlen_q=31235, seqlen_k=31235//16, causal=False)
    # test_flash_attn_varlen(seqlen_q=31235, seqlen_k=31235//16, causal=True)
    # test_flash_attn_varlen(seqlen_q=16384, seqlen_k=16384//16, bench=True)
    # test_flash_attn_varlen(seqlen_q=32768, seqlen_k=32768//16, bench=True)
    # test_flash_attn_varlen(seqlen_q=131072, seqlen_k=131072//16, bench=True)
    # test_flash_attn_varlen(seqlen_q=131072, seqlen_k=131072//16, bench=True, causal=True)
