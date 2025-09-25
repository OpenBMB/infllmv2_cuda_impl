import time
import torch
import torch.nn.functional as F
from infllm_v2 import infllmv2_attn_stage1

def round_multiple(x, m):
    return (x + m - 1) // m * m

def calc_chunks_with_stride(cu_seqlen, chunk_size, kernel_stride):
    """
    Compute the chunks that require compression, with stride support.
    """
    # 1. Compute the length of each sequence
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]

    # 2. Compute the start positions of chunks for each sequence (with stride)
    max_seq_len = torch.max(batch_sizes)
    
    # Handle edge case where max_seq_len < chunk_size
    if max_seq_len < chunk_size:
        # No valid chunks
        filtered_indices = torch.tensor([], dtype=torch.long, device=cu_seqlen.device)
        cu_seqlens_compressed = torch.zeros(len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device)
        return filtered_indices, cu_seqlens_compressed
    
    max_num_chunks_per_seq = (max_seq_len - chunk_size) // kernel_stride + 1
    chunk_start_offsets = torch.arange(0, max_num_chunks_per_seq * kernel_stride, kernel_stride, device=cu_seqlen.device)
    seq_starts = cu_seqlen[:-1]
    chunk_start_in_seq = seq_starts[:, None] + chunk_start_offsets[None, :]  # [batch_size, max_num_chunks_per_seq]

    # 3. Filter out chunks that exceed sequence length or are smaller than the full chunk size
    chunk_end_in_seq = chunk_start_in_seq + chunk_size
    valid_chunk_mask = (chunk_end_in_seq <= (seq_starts[:, None] + batch_sizes[:, None]))

    # 4. Filter valid chunk start positions using the valid_chunk_mask
    valid_chunk_starts = chunk_start_in_seq[valid_chunk_mask]  # [num_valid_chunks]
    
    # 5. Generate filtered_indices
    chunk_indices = torch.arange(
        0, chunk_size, device=cu_seqlen.device
    )[None, :]  # [1, chunk_size]
    filtered_indices = valid_chunk_starts[:, None] + chunk_indices  # [num_valid_chunks, chunk_size]
    filtered_indices = filtered_indices.view(-1)  # Flatten to 1D indices

    # 6. Compute compressed cumulative sequence lengths
    num_filtered_chunks_per_batch = valid_chunk_mask.sum(dim=1)  # Number of valid chunks per batch
    cu_seqlens_compressed = torch.zeros(
        len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device
    )
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0)
    
    return filtered_indices, cu_seqlens_compressed

def compress_tensor(tensor, cu_seqlens, kernel_size, kernel_stride, n_heads, head_dim):
    """
    Compress a tensor using kernel_size and kernel_stride parameters.
    
    Args:
        tensor: Input tensor of shape (n_heads, total_seq_len, head_dim)
        cu_seqlens: Cumulative sequence lengths
        kernel_size: Size of each chunk for compression
        kernel_stride: Stride when sliding over the sequence
        n_heads: Number of heads
        head_dim: Head dimension
    
    Returns:
        compressed_tensor: Compressed tensor
        cu_seqlens_compressed: Compressed cumulative sequence lengths
    """
    # Transpose to (total_seq_len, n_heads, head_dim) for easier indexing
    tensor = tensor.transpose(0, 1).contiguous()
    
    # Compute chunk-related metadata
    filtered_indices, cu_seqlens_compressed = calc_chunks_with_stride(
        cu_seqlens, kernel_size, kernel_stride
    )
    
    # Handle edge case where there are no valid chunks
    if filtered_indices.numel() == 0:
        # Return empty tensor with proper shape
        compressed_tensor = torch.empty(n_heads, 0, head_dim, dtype=tensor.dtype, device=tensor.device)
        return compressed_tensor, cu_seqlens_compressed
    
    # Extract filtered vectors
    filtered_tensor = tensor.index_select(0, filtered_indices.view(-1))
    
    # Reshape and compute mean
    filtered_tensor = filtered_tensor.view(
        filtered_tensor.shape[0] // kernel_size, kernel_size, n_heads, head_dim
    )  # [num_chunks, kernel_size, n_heads, head_dim]
    
    compressed_tensor = filtered_tensor.mean(dim=1)  # [num_chunks, n_heads, head_dim]
    
    # Transpose back to (n_heads, total_compressed_len, head_dim)
    compressed_tensor = compressed_tensor.transpose(0, 1).contiguous()
    
    return compressed_tensor, cu_seqlens_compressed

def naive_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=False):
    # 将 varlen 输入转换为 padded 形式
    batch_size = len(cu_seqlens_q) - 1
    max_seqlen_q = max(cu_seqlens_q[i+1] - cu_seqlens_q[i] for i in range(batch_size))
    max_seqlen_k = max(cu_seqlens_k[i+1] - cu_seqlens_k[i] for i in range(batch_size))
    
    # 创建 padded 张量
    q_padded = torch.zeros(q.shape[0], batch_size, max_seqlen_q, q.shape[-1], device=q.device, dtype=q.dtype)
    k_padded = torch.zeros(k.shape[0], batch_size, max_seqlen_k, k.shape[-1], device=k.device, dtype=k.dtype)
    
    # 填充数据
    for i in range(batch_size):
        q_start = cu_seqlens_q[i]
        q_end = cu_seqlens_q[i + 1]
        k_start = cu_seqlens_k[i]
        k_end = cu_seqlens_k[i + 1]
        q_padded[:, i, :q_end-q_start] = q[:, q_start:q_end]
        k_padded[:, i, :k_end-k_start] = k[:, k_start:k_end]
    
    # 计算 attention
    k_padded = k_padded.repeat_interleave(q_padded.shape[0] // k_padded.shape[0], dim=0)
    
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
    seqlen_k0s = [seqlen_k // 2, seqlen_k]  
    total_seqlen_q = sum(seqlen_qs)
    total_seqlen_k0 = sum(seqlen_k0s)
    
    # 准备输入数据
    q = torch.randn(n_heads, total_seqlen_q, head_dim, dtype=dtype).cuda()
    
    # 生成 k0
    k0 = torch.randn(n_kv_heads, total_seqlen_k0, head_dim, dtype=dtype).cuda()
    
    # 计算 k0 的累积序列长度
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device='cuda')
    cu_seqlens_k0 = torch.zeros(batch_size + 1, dtype=torch.int32, device='cuda')
    for i in range(batch_size):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + seqlen_qs[i]
        cu_seqlens_k0[i + 1] = cu_seqlens_k0[i] + seqlen_k0s[i]
    
    # 压缩 k0 得到 k (kernel_size=32, kernel_stride=16)
    k, cu_seqlens_k = compress_tensor(k0, cu_seqlens_k0, kernel_size=32, kernel_stride=16, 
                                      n_heads=n_kv_heads, head_dim=head_dim)
    
    # 压缩 k0 得到 v (kernel_size=128, kernel_stride=64)
    v, cu_seqlens_v = compress_tensor(k0, cu_seqlens_k0, kernel_size=128, kernel_stride=64, 
                                      n_heads=n_kv_heads, head_dim=head_dim)
    
    total_seqlen_k = k.shape[1]
    total_seqlen_v = v.shape[1]
    # breakpoint()
    # 朴素实现
    if not bench:
        naive_score = naive_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal)

    q = q.transpose(0, 1).contiguous().clone()
    k = k.transpose(0, 1).contiguous().clone()
    v = v.transpose(0, 1).contiguous().clone()
    
    # print(f"Original k0 lengths: {seqlen_k0s}")
    # print(f"cu_seqlens_k0: {cu_seqlens_k0}")
    # print(f"Compressed k shape: {k.shape}")
    # print(f"cu_seqlens_k: {cu_seqlens_k}")
    # print(f"Compressed v shape: {v.shape}")
    # print(f"cu_seqlens_v: {cu_seqlens_v}")
    
    # Calculate max sequence lengths from cu_seqlens
    seqlen_ks = [(cu_seqlens_k[i+1] - cu_seqlens_k[i]).item() for i in range(batch_size)]
    seqlen_vs = [(cu_seqlens_v[i+1] - cu_seqlens_v[i]).item() for i in range(batch_size)]
    
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
        
        breakpoint()
        if (naive_score - flash_score).abs().max() > 1e-2:
            print(f"error: seqlen_qs={seqlen_qs}, seqlen_ks={seqlen_ks}")

if __name__ == "__main__":
    # Test cases for causal=False
    test_seqlens = [1000, 5000, 9000]
    # test_seqlens = [2048]
    # for seqlen in test_seqlens:
    #     # seqlen_k is the compressed k length, which will be seqlen//32 after compression
    #     # seqlen_v is the compressed v length, which will be seqlen//128 after compression
    #     test_flash_attn_varlen(seqlen_q=1, seqlen_k=seqlen, causal=False)
    
    # Test cases for causal=True
    for seqlen in test_seqlens:
        # For causal=True, adjust the lengths accordingly
        test_flash_attn_varlen(seqlen_q=seqlen, seqlen_k=seqlen, causal=False)

    # test_flash_attn_varlen(seqlen_q=10000, seqlen_k=10000//16, causal=False)
    # test_flash_attn_varlen(seqlen_q=10000, seqlen_k=10000//16, causal=True)
    # test_flash_attn_varlen(seqlen_q=31235, seqlen_k=31235//16, causal=False)
    # test_flash_attn_varlen(seqlen_q=31235, seqlen_k=31235//16, causal=True)
    # test_flash_attn_varlen(seqlen_q=16384, seqlen_k=16384//16, bench=True)
    # test_flash_attn_varlen(seqlen_q=32768, seqlen_k=32768//16, bench=True)
    # test_flash_attn_varlen(seqlen_q=131072, seqlen_k=131072//16, bench=True)
    # test_flash_attn_varlen(seqlen_q=131072, seqlen_k=131072//16, bench=True, causal=True)
