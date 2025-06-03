# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_blocksparse_attn_interface.py

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

# Import from infllm_v2's C extension and local modules
from . import C as infllm_cuda
from .blockmask import blockmask_to_uint64
from .topk_to_uint64 import topk_to_uint64
from .uint64_to_bool import uint64_to_bool


def replace_ones_with_count(tensor):
    ones_mask = tensor == 1
    ones_num = ones_mask.sum()
    count = torch.cumsum(ones_mask, dim=-1).to(tensor.dtype)
    count = count * ones_mask
    tensor = tensor.masked_scatter(ones_mask, count[ones_mask])
    return tensor, ones_num


def _block_sparse_attn_forward(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    m_block_dim, n_block_dim,
    head_mask_type,
    streaming_info,
    topk_idx,  # Changed from row_blockmask to topk_idx
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    softmax_scale,
    is_causal,
    exact_streaming,
    return_softmax,
    window_size_left,
    window_size_right,
    block_window_size=0
):
    # Calculate the ratio of q heads to k sequence
    group_size = q.shape[-2] // k.shape[-2]
    
    # Memory-efficient reshaping - avoid intermediate copies
    total_q, nheads_q, dim = q.shape
    nheads_k = k.shape[-2]
    
    # Reshape directly to final format to minimize temporary tensors
    q_final = q.reshape(total_q, nheads_k, group_size, dim)
    q_final = q_final.permute(0, 2, 1, 3)
    q_final = q_final.reshape(total_q * group_size, nheads_k, dim).contiguous()
    
    # Use in-place operations for cu_seqlens_q_expanded to avoid extra allocation
    cu_seqlens_q_expanded = torch.zeros(cu_seqlens_q.shape[0], device=cu_seqlens_q.device, dtype=cu_seqlens_q.dtype)
    for i in range(cu_seqlens_q.shape[0]-1):
        cu_seqlens_q_expanded[i+1] = (cu_seqlens_q[i+1] - cu_seqlens_q[i]) * group_size + cu_seqlens_q_expanded[i]
    
    # Adjust max_seqlen_q_ to account for the expanded sequence
    max_seqlen_q_expanded = max_seqlen_q_ * group_size
    
    # Convert topk_idx to uint64 representation
    blockmask_uint64, last_dim_size = topk_to_uint64(topk_idx, max_seqlen_k_, n_block_dim)
    
    # CUDA operation
    out, _, k_out, v_out, _, softmax_lse, _, rng_state = infllm_cuda.fwd_block(
        q_final, k, v,
        cu_seqlens_q_expanded, cu_seqlens_k,
        m_block_dim, n_block_dim,
        head_mask_type,
        streaming_info,
        blockmask_uint64,
        max_seqlen_q_expanded, max_seqlen_k_,
        p_dropout,
        softmax_scale,
        is_causal,
        exact_streaming,
        return_softmax,
        window_size_left,
        window_size_right, 
        block_window_size,
        None
    )
    
    # Efficient reshaping back to original shape - break into steps to reduce peak memory
    out = out.reshape(total_q, group_size, nheads_k, dim)
    out = out.permute(0, 2, 1, 3)
    out = out.reshape(total_q, nheads_q, dim)
    
    # Clear references to intermediate tensors
    q_final = k_out = v_out = None
    torch.cuda.empty_cache()
    
    return out, q, k, v, softmax_lse, blockmask_uint64, rng_state  # Return blockmask_uint64 to store in ctx


def _block_sparse_attn_backward(
    dout,
    q, k, v,
    out,
    softmax_lse,
    dq, dk, dv,
    cu_seqlens_q, cu_seqlens_k,
    m_block_dim, n_block_dim,
    head_mask_type,
    streaming_info,
    col_blockmask_uint64,  # This is now directly the uint64 representation
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    softmax_scale,
    zero_tensors,
    is_causal,
    window_size_left,
    window_size_right,
    block_window_size=0,
    deterministic=False,
    rng_state=None,
):
    # Calculate the ratio of q heads to k sequence
    group_size = q.shape[-2] // k.shape[-2]
    
    # Get original shapes and dimensions
    total_q, nheads_q, dim = q.shape
    nheads_k = k.shape[-2]
    
    # Memory-efficient reshaping - break into steps with immediate cleanup
    q_final = q.reshape(total_q, nheads_k, group_size, dim)
    q_final = q_final.permute(0, 2, 1, 3)
    q_final = q_final.reshape(total_q * group_size, nheads_k, dim).contiguous()
    
    dout_final = dout.reshape(total_q, nheads_k, group_size, dim)
    dout_final = dout_final.permute(0, 2, 1, 3)
    dout_final = dout_final.reshape(total_q * group_size, nheads_k, dim).contiguous()
    
    out_final = out.reshape(total_q, nheads_k, group_size, dim)
    out_final = out_final.permute(0, 2, 1, 3)
    out_final = out_final.reshape(total_q * group_size, nheads_k, dim).contiguous()
    
    # Reduce memory by computing cu_seqlens_q_expanded in-place
    cu_seqlens_q_expanded = torch.zeros(cu_seqlens_q.shape[0], device=cu_seqlens_q.device, dtype=cu_seqlens_q.dtype)
    for i in range(cu_seqlens_q.shape[0]-1):
        cu_seqlens_q_expanded[i+1] = (cu_seqlens_q[i+1] - cu_seqlens_q[i]) * group_size + cu_seqlens_q_expanded[i]
    
    max_seqlen_q_expanded = max_seqlen_q_ * group_size
    
    # Create dq_temp directly with correct shape
    dq_temp = torch.empty_like(q_final)
    
    # CUDA operation
    _, _, _, softmax_d = infllm_cuda.bwd_block(
        dout_final,
        q_final, k, v,
        out_final,
        softmax_lse,
        dq_temp, dk, dv,
        cu_seqlens_q_expanded, cu_seqlens_k,
        m_block_dim, n_block_dim,
        head_mask_type,
        streaming_info,
        col_blockmask_uint64,
        max_seqlen_q_expanded, max_seqlen_k_,
        p_dropout,
        softmax_scale,
        zero_tensors,
        is_causal,
        window_size_left,
        window_size_right,
        block_window_size,
        deterministic,
        None, rng_state
    )
    
    # Free memory immediately after use
    dout_final = out_final = None
    torch.cuda.empty_cache()
    
    # Reshape dq_temp directly back to original shape
    dq_temp = dq_temp.reshape(total_q, group_size, nheads_k, dim)
    dq_temp = dq_temp.permute(0, 2, 1, 3)
    dq_temp = dq_temp.reshape(total_q, nheads_q, dim)
    
    # Use in-place copy instead of assignment
    dq.copy_(dq_temp)
    
    # Clean up remaining references
    q_final = dq_temp = None
    torch.cuda.empty_cache()
    
    return dq, dk, dv, softmax_d


class BlockSparseAttnFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                m_block_dim, n_block_dim,
                head_mask_type,
                streaming_info,
                topk_idx,  # Changed from base_blockmask to topk_idx
                max_seqlen_q_, max_seqlen_k_,
                p_dropout,
                softmax_scale,
                is_causal,
                exact_streaming,
                return_softmax,
                window_size_left,
                window_size_right,
                block_window_size=0,
                deterministic=False):
        # Save rng_state because the backward pass will regenerate the dropout mask
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
            
        if exact_streaming:
            assert streaming_info is not None
            assert is_causal
        
        out, q_orig, k_orig, v_orig, softmax_lse, fwd_blockmask_uint64, rng_state= _block_sparse_attn_forward(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            m_block_dim, n_block_dim,
            head_mask_type,
            streaming_info,
            topk_idx,
            max_seqlen_q_, max_seqlen_k_,
            p_dropout,
            softmax_scale,
            is_causal,
            exact_streaming,
            return_softmax=False,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            block_window_size=block_window_size
        )

        # Save base_blockmask instead of col_blockmask to avoid computation in forward pass
        ctx.save_for_backward(q_orig, k_orig, v_orig,
                              out, softmax_lse,
                              cu_seqlens_q, cu_seqlens_k,
                              head_mask_type,
                              streaming_info,
                              fwd_blockmask_uint64,  # Store the uint64 representation
                              rng_state)
        ctx.m_block_dim = m_block_dim
        ctx.n_block_dim = n_block_dim
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.max_seqlen_q_ = max_seqlen_q_
        ctx.max_seqlen_k_ = max_seqlen_k_
        ctx.p_dropout = p_dropout
        ctx.softmax_scale = softmax_scale
        ctx.is_causal = is_causal
        ctx.exact_streaming = exact_streaming
        ctx.deterministic = deterministic
        ctx.block_window_size = block_window_size
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, head_mask_type, streaming_info, fwd_blockmask_uint64, rng_state = ctx.saved_tensors
        
        # Pre-allocate gradient tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        fwd_blockmask_bool = uint64_to_bool(fwd_blockmask_uint64, (ctx.max_seqlen_k_+ctx.n_block_dim - 1)//ctx.n_block_dim)
        
        # Ensure the tensor is contiguous in memory after transpose
        transposed_blockmask = fwd_blockmask_bool.transpose(1, 2).contiguous()
        
        # Synchronize CUDA stream before conversion
        torch.cuda.synchronize()
        
        # Convert to uint64
        bwd_blockmask_uint64, _ = blockmask_to_uint64(transposed_blockmask)
        
        assert not ctx.exact_streaming, "Exact streaming not supported in backward pass"
        _block_sparse_attn_backward(
            dout,
            q, k, v,
            out,
            softmax_lse,
            dq, dk, dv,
            cu_seqlens_q, cu_seqlens_k,
            ctx.m_block_dim, ctx.n_block_dim,
            head_mask_type,
            streaming_info,
            bwd_blockmask_uint64,  # Use the uint64 matrix directly
            ctx.max_seqlen_q_, ctx.max_seqlen_k_,
            ctx.p_dropout,
            ctx.softmax_scale,
            True,  # zero_tensors
            ctx.is_causal,
            ctx.window_size_left,
            ctx.window_size_right,
            ctx.block_window_size,
            ctx.deterministic,
            rng_state=rng_state
        )
        
        # Free memory of tensors no longer needed
        del bwd_blockmask_uint64, out, softmax_lse, rng_state
        torch.cuda.empty_cache()
        
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def block_sparse_attn_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,
    topk_idx,  # Changed from base_blockmask to topk_idx
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    deterministic=False,
    softmax_scale=None,
    is_causal=False,
    exact_streaming=False,
    return_attn_probs=False,
    use_checkpoint=False,
    window_size_left=-1,
    window_size_right=-1,
    block_window_size=0,
):
    head_mask_type, blocksparse_head_num = replace_ones_with_count(head_mask_type)
    if topk_idx is not None:
        assert topk_idx.shape[0] == blocksparse_head_num
    
    # Use different chunk sizes depending on tensor size to manage memory
    chunk_size = 1024 if q.shape[0] > 8192 else 2048
    
    func = BlockSparseAttnFun
    
    # Define block dimensions - using a fixed block size of 64 for n_block_dim
    m_block_dim = 16
    n_block_dim = 64
    

    # Define a wrapper function for checkpointing
    def checkpoint_wrapper(q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type, streaming_info, topk_idx, block_window_size):
        return func.apply(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            m_block_dim, n_block_dim,
            head_mask_type,
            streaming_info,
            topk_idx,
            max_seqlen_q_, max_seqlen_k_,
            p_dropout,
            softmax_scale,
            is_causal,
            exact_streaming,
            return_attn_probs,
            window_size_left, window_size_right,
            block_window_size,
            deterministic
        )
    
    # Conditionally use checkpointing based on use_checkpoint parameter
    if use_checkpoint:
        return checkpoint(
            checkpoint_wrapper,
            q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type, streaming_info, topk_idx, block_window_size,
            preserve_rng_state=True
        )
    else:
        return func.apply(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            m_block_dim, n_block_dim,
            head_mask_type,
            streaming_info,
            topk_idx,
            max_seqlen_q_, max_seqlen_k_,
            p_dropout,
            softmax_scale,
            is_causal,
            exact_streaming,
            return_attn_probs,
            window_size_left, window_size_right,
            block_window_size,
            deterministic
        )


def block_sparse_attn_kvcache_func(
    q,                    # batch_size x seqlen_q x num_heads x head_size
    k_cache,               # batch_size_c x seqlen_k x num_heads_k x head_size
    v_cache, 
    m_block_dim = 16, 
    n_block_dim = 64,              # batch_size_c x seqlen_k x num_heads_k x head_size
    head_mask_type = None,       # num_heads
    out = None,
    streaming_info=None,  # num_heads*2
    topk_idx=None,        # (num_blocksparse_heads, num_blocks, topk)
    k=None,               # batch_size x seqlen_knew x num_heads_k x head_size
    v=None,               # batch_size x seqlen_knew x num_heads_k x head_size
    seqlens_k=None,       # batch_size
    rotary_cos=None,      # seqlen_ro x (rotary_dim / 2)
    rotary_sin=None,      # seqlen_ro x (rotary_dim / 2)
    cache_batch_idx=None, # indices to index into the KV cache
    alibi_slopes=None,    # num_heads or batch_size x num_heads
    softmax_scale=None,
    causal=False,
    exact_streaming=False,
    window_size_left=-1,
    window_size_right=-1,
    block_window_size=0,
    rotary_interleaved=False,
    num_splits=16
):
    # head_mask_type, blocksparse_head_num = replace_ones_with_count(head_mask_type)
    

    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
        # Calculate the ratio of q heads to k sequence
    # batch_size, seqlen_q, nheads_q, dim = q.shape
    nheads_k = k_cache.shape[-2]
    seqlen_k = k_cache.shape[1] #+ k.shape[1] if k is not None else k_cache.shape[1]
    # group_size = nheads_q // nheads_k
    
    # Memory-efficient reshaping for 4D input tensors
    # Reshape q from [batch_size, seqlen_q, nheads_q, dim] to [batch_size, seqlen_q*group_size, nheads_k, dim]
    # by rearranging the head dimension into sequence groups
    # q_reshaped = q.reshape(batch_size, seqlen_q, nheads_k, group_size, dim)
    # q_reshaped = q_reshaped.permute(0, 1, 3, 2, 4)  # [batch_size, seqlen_q, group_size, nheads_k, dim]
    # q_final = q_reshaped.reshape(batch_size, seqlen_q * group_size, nheads_k, dim).contiguous()
    
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if seqlens_k is not None and isinstance(seqlens_k, int):
        seqlens_k = torch.full(
            (k_cache.shape[0],), seqlens_k, dtype=torch.int32, device=k_cache.device
        )
        seqlens_k = maybe_contiguous(seqlens_k)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    
        # Convert topk_idx to uint64 representation if provided
    blockmask_uint64 = None
    if topk_idx is not None:
        assert topk_idx.shape[1] == nheads_k, "Number of heads in topk_idx must match blocksparse_head_num"
        
        blockmask_uint64, _ = topk_to_uint64(topk_idx, seqlen_k, n_block_dim)
    
    # Call the CUDA implementation
    # print(f"blockmask_uint64.shape: {blockmask_uint64.shape}, k_cache.shape[1]: {k_cache.shape[1]}, seqlen_k: {seqlen_k}, n_block_dim: {n_block_dim}")
    out, softmax_lse = infllm_cuda.fwd_block_kvcache(
        q,
        k_cache,
        v_cache,
        m_block_dim,
        n_block_dim,
        head_mask_type,
        streaming_info,
        blockmask_uint64,
        k,
        v,
        seqlens_k,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        alibi_slopes,
        out,
        softmax_scale,
        causal,
        exact_streaming,
        window_size_left,
        window_size_right,
        block_window_size,
        rotary_interleaved,
        num_splits
    )
    return out, softmax_lse 