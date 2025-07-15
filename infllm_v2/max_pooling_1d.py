import torch
from . import C

def max_pooling_1d(
    input: torch.Tensor, # num_heads x q_len x k_len
    cache_len: int,
    local_blocks: int,
    init_blocks: int,
    block_size: int = 64,
    stride: int = 16,
) -> torch.Tensor:
    assert input.dtype == torch.float16 or input.dtype == torch.bfloat16
    input = input.contiguous()
    stride = block_size // stride
    kernel_size = stride + 1
    padding = 1
    num_heads = input.shape[0]
    q_len = input.shape[1]
    k_len = input.shape[2]
    total_len = q_len + cache_len
    out_len = (total_len + block_size - 1) // block_size
    output = torch.zeros(num_heads, q_len, out_len, device=input.device, dtype=input.dtype)
    C.max_pooling_1d(
        torch.cuda.current_stream(input.device).cuda_stream,
        input.data_ptr(),
        output.data_ptr(),
        input.dtype == torch.bfloat16,
        num_heads,
        q_len,
        k_len,
        out_len,
        cache_len,
        kernel_size,
        stride,
        padding,
        block_size,
        local_blocks,
        init_blocks,
    )
    return output


def max_pooling_1d_varlen(
    input: torch.Tensor, # num_heads x total_q x max_k
    cu_seqlens_q: torch.Tensor, # batch_size + 1
    cu_seqlens_k: torch.Tensor, # batch_size + 1
    cache_lens: torch.Tensor, # batch_size
    max_seqlen_q: int,
    max_seqlen_k: int,
    local_blocks: int,
    init_blocks: int,
    block_size: int = 64,
    stride: int = 16,
) -> torch.Tensor:
    """
    Variable-length version of max_pooling_1d that handles packed sequences.
    
    Args:
        input: Tensor of shape (num_heads, total_q, max_k) where:
               - total_q is sum of all query sequence lengths
               - max_k is the maximum key sequence length (padded)
        cu_seqlens_q: Cumulative sequence lengths for queries (batch_size + 1,)
        cu_seqlens_k: Cumulative sequence lengths for keys (batch_size + 1,)
        cache_lens: Cache lengths for each sequence in the batch (batch_size,)
        max_seqlen_q: Maximum query sequence length in the batch
        max_seqlen_k: Maximum key sequence length in the batch
        local_blocks: Number of local blocks for window attention
        init_blocks: Number of initial blocks to mask with inf
        block_size: Block size (default: 64)
        stride: Stride for pooling (default: 16)
    
    Returns:
        output: Tensor of shape (num_heads, total_q, out_len)
    """
    assert input.dtype == torch.float16 or input.dtype == torch.bfloat16
    assert cu_seqlens_q.dtype == torch.int32
    assert cu_seqlens_k.dtype == torch.int32
    assert cache_lens.dtype == torch.int32
    assert input.dim() == 3, f"Expected 3D input, got {input.dim()}D"
    
    input = input.contiguous()
    cu_seqlens_q = cu_seqlens_q.contiguous()
    cu_seqlens_k = cu_seqlens_k.contiguous()
    cache_lens = cache_lens.contiguous()
    
    stride = block_size // stride
    kernel_size = stride + 1
    padding = 1
    
    batch_size = cu_seqlens_q.shape[0] - 1
    num_heads = input.shape[0]
    total_q = input.shape[1]
    
    # Verify dimensions
    assert cu_seqlens_q[-1].item() == total_q, f"total_q mismatch: {cu_seqlens_q[-1].item()} vs {total_q}"
    assert input.shape[2] == max_seqlen_k, f"max_k mismatch: {input.shape[2]} vs {max_seqlen_k}"
    assert cache_lens.shape[0] == batch_size, f"cache_lens batch size mismatch: {cache_lens.shape[0]} vs {batch_size}"
    
    # Calculate output length based on max sequence length and max cache length
    max_cache_len = cache_lens.max().item()
    total_len = max_seqlen_q + max_cache_len
    out_len = (total_len + block_size - 1) // block_size
    
    output = torch.zeros(num_heads, total_q, out_len, device=input.device, dtype=input.dtype)
    
    C.max_pooling_1d_varlen(
        torch.cuda.current_stream(input.device).cuda_stream,
        input.data_ptr(),
        output.data_ptr(),
        cu_seqlens_q.data_ptr(),
        cu_seqlens_k.data_ptr(),
        cache_lens.data_ptr(),
        input.dtype == torch.bfloat16,
        batch_size,
        num_heads,
        max_seqlen_q,
        max_seqlen_k,
        out_len,
        kernel_size,
        stride,
        padding,
        block_size,
        local_blocks,
        init_blocks,
    )
    return output

