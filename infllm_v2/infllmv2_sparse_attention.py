# Copyright (c) 2023, Tri Dao.

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import os

# Import from infllm_v2's C extension and local modules
from . import C as infllm_cuda
from .topk_to_uint64 import topk_to_uint64 as cuda_topk_to_uint64
from .uint64_to_bool import uint64_to_bool as cuda_uint64_to_bool
from .blockmask_to_uint64 import blockmask_to_uint64 as cuda_blockmask_to_uint64

# isort: on

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def round_multiple(x, m):
    return (x + m - 1) // m * m


# torch.compile() support is only enabled for pytorch >= 2.4
# The reason for this is that we are using the new custom_op and register_fake
# APIs, which support inplace modification of inputs in the function itself
if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:
    def noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper


@_torch_custom_op_wrapper("infllmv2_attn::_infllmv2_attn_varlen_forward", mutates_args=(), device_types="cuda")
def _infllmv2_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
    block_table: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    topk_idx: Optional[torch.Tensor] = None,

) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    if topk_idx is not None:
        # Calculate group size (number of query heads per K/V head)
        nheads_q = q.shape[1]
        nheads_k = k.shape[1]
        group_size = nheads_q // nheads_k
        head_dim = q.shape[-1]
        
        # Optimize for MQA case (nheads_k == 1)
        if nheads_k == 1:
            # Direct reshape for MQA - no transpose needed
            q = q.reshape(-1, 1, head_dim).contiguous()
            cu_seqlens_q = cu_seqlens_q * nheads_q
            max_seqlen_q = max_seqlen_q * nheads_q
        else:
            # General case for GQA/MHA
            q = q.reshape(-1, nheads_k, group_size, head_dim).transpose(1, 2).reshape(-1, nheads_k, head_dim).contiguous()
            cu_seqlens_q = cu_seqlens_q * group_size
            max_seqlen_q = max_seqlen_q * group_size
        
        assert topk_idx.dtype == torch.int32
        fwd_blockmask_uint64, _ = cuda_topk_to_uint64(topk_idx, max_seqlen_k, 64) # N_BLOCK_DIM=64
    else:
        fwd_blockmask_uint64 = None

    out, softmax_lse, S_dmask, rng_state = infllm_cuda.varlen_fwd(
        q,
        k,
        v,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        None,
        fwd_blockmask_uint64,
    )
    # if out.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    if topk_idx is not None:
        # Reshape output back to original dimensions
        if nheads_k == 1:
            # Direct reshape for MQA - no transpose needed
            out = out.reshape(-1, nheads_q, head_dim).contiguous()
        else:
            # General case for GQA/MHA
            out = out.reshape(-1, group_size, nheads_k, head_dim).transpose(1, 2).reshape(-1, nheads_q, head_dim).contiguous()
    
    return out, softmax_lse, S_dmask, fwd_blockmask_uint64, rng_state



_wrapped_infllmv2_attn_varlen_forward = _infllmv2_attn_varlen_forward


@_torch_custom_op_wrapper("infllmv2_attn::_infllmv2_attn_varlen_backward", mutates_args=("dq", "dk", "dv"), device_types="cuda")
def _infllmv2_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    bwd_blockmask_uint64 : Optional[torch.Tensor] = None,  # Use the uint64 matrix directly
    rng_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    # Calculate the ratio of q heads to k sequence
    group_size = q.shape[-2] // k.shape[-2]
    # Get original shapes and dimensions
    total_q, nheads_q, dim = q.shape
    nheads_k = k.shape[-2]
    
    # Optimize for MQA case (nheads_k == 1)
    if nheads_k == 1:
        # Direct reshape for MQA - no transpose needed
        q_final = q.reshape(total_q * nheads_q, 1, dim).contiguous()
        dout_final = dout.reshape(total_q * nheads_q, 1, dim).contiguous()
        out_final = out.reshape(total_q * nheads_q, 1, dim).contiguous()
    else:
        # Memory-efficient reshaping for GQA/MHA - break into steps with immediate cleanup
        q_final = q.reshape(total_q, nheads_k, group_size, dim)
        q_final = q_final.permute(0, 2, 1, 3)
        q_final = q_final.reshape(total_q * group_size, nheads_k, dim).contiguous()
        
        dout_final = dout.reshape(total_q, nheads_k, group_size, dim)
        dout_final = dout_final.permute(0, 2, 1, 3)
        dout_final = dout_final.reshape(total_q * group_size, nheads_k, dim).contiguous()
        
        out_final = out.reshape(total_q, nheads_k, group_size, dim)
        out_final = out_final.permute(0, 2, 1, 3)
        out_final = out_final.reshape(total_q * group_size, nheads_k, dim).contiguous()
    # q = q.reshape(-1, 16, 2, head_dim).reshape(-1, 2, head_dim)
    # breakpoint()
    # with open("/user/luopeiyan/a/Block-Sparse-Attention/tests/after_q_output_flash_attn_varlen_forward.txt", "a+") as f:
    #     f.write(str(q) + "\n") 
    # Reduce memory by computing cu_seqlens_q_expanded in-place
    if nheads_k == 1:
        # For MQA, multiply by nheads_q
        cu_seqlens_q_expanded = torch.zeros(cu_seqlens_q.shape[0], device=cu_seqlens_q.device, dtype=cu_seqlens_q.dtype)
        for i in range(cu_seqlens_q.shape[0]-1):
            cu_seqlens_q_expanded[i+1] = (cu_seqlens_q[i+1] - cu_seqlens_q[i]) * nheads_q + cu_seqlens_q_expanded[i]
        max_seqlen_q_expanded = max_seqlen_q * nheads_q
    else:
        # For GQA/MHA, multiply by group_size
        cu_seqlens_q_expanded = torch.zeros(cu_seqlens_q.shape[0], device=cu_seqlens_q.device, dtype=cu_seqlens_q.dtype)
        for i in range(cu_seqlens_q.shape[0]-1):
            cu_seqlens_q_expanded[i+1] = (cu_seqlens_q[i+1] - cu_seqlens_q[i]) * group_size + cu_seqlens_q_expanded[i]
        max_seqlen_q_expanded = max_seqlen_q * group_size
    # Create dq_temp directly with correct shape
    dq_temp = torch.empty_like(q_final)
    
    (
        _,
        _,
        _,
        softmax_d,
    ) = infllm_cuda.varlen_bwd(
        dout_final,
        q_final,
        k,
        v,
        out_final,
        softmax_lse,
        dq_temp,
        dk,
        dv,
        cu_seqlens_q_expanded,
        cu_seqlens_k,
        alibi_slopes,
        max_seqlen_q_expanded,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        bwd_blockmask_uint64,
        None,
        rng_state,
    )
    # Free memory immediately after use
    dout_final = out_final = None
    torch.cuda.empty_cache()
    
    # Reshape dq_temp directly back to original shape
    if nheads_k == 1:
        # Direct reshape for MQA - no transpose needed
        dq_temp = dq_temp.reshape(total_q, nheads_q, dim)
    else:
        # General case for GQA/MHA
        dq_temp = dq_temp.reshape(total_q, group_size, nheads_k, dim)
        dq_temp = dq_temp.permute(0, 2, 1, 3)
        dq_temp = dq_temp.reshape(total_q, nheads_q, dim)
    
    # Use in-place copy instead of assignment
    dq.copy_(dq_temp)
    
    # Clean up remaining references
    q_final = dq_temp = None
    torch.cuda.empty_cache()
    return softmax_d




if torch.__version__ >= "2.4.0":
    _wrapped_infllmv2_attn_varlen_backward = torch.ops.infllmv2_attn._infllmv2_attn_varlen_backward
else:
    _wrapped_infllmv2_attn_varlen_backward = _infllmv2_attn_varlen_backward


class Infllmv2AttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        block_table,
        topk_idx,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, fwd_blockmask_uint64, rng_state = _wrapped_infllmv2_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=block_table,
            topk_idx=topk_idx,
        )
        ctx.save_for_backward(
            q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, fwd_blockmask_uint64, rng_state
        )
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.topk_idx = topk_idx
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, fwd_blockmask_uint64, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        bwd_blockmask_uint64 = None
        if fwd_blockmask_uint64 is not None:
            fwd_blockmask_bool = cuda_uint64_to_bool(fwd_blockmask_uint64, (ctx.max_seqlen_k + 64- 1) // 64) 
            # Ensure the tensor is contiguous in memory after transpose
            transposed_blockmask = fwd_blockmask_bool.transpose(1, 2).contiguous()
            # Synchronize CUDA stream before conversion
            torch.cuda.synchronize()
            # Convert to uint64
            bwd_blockmask_uint64, _ = cuda_blockmask_to_uint64(transposed_blockmask)
        
        head_size_og = dout.size(2)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
        _wrapped_infllmv2_attn_varlen_backward(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            bwd_blockmask_uint64,  # Use the uint64 matrix directly
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def infllmv2_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    topk_idx=None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return Infllmv2AttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_table,
        topk_idx,
    )


def infllmv2_attn_stage1(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    cu_seqlens_v,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=True,
    block_table=None,
):
    """
    Neighborhood Sparse Attention (NSA) Stage 1 with varlen support.
    
    This function performs the first stage of NSA, computing attention scores
    with a specific sparsity pattern where queries are grouped and only attend
    to a subset of keys.
    
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into k.
        cu_seqlens_v: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into v.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation.
        return_attn_probs: bool. Whether to return the attention probabilities.
        block_table: Optional block table for paged attention.
        nsa_group_size: int. Number of groups for neighborhood sparse attention.
        nsa_heads_per_group: int. Number of heads per group.
        
    Return:
        S_dmask: The attention scores/probabilities matrix with NSA sparsity pattern.
                 Shape: (num_heads_k, total_q, max_seqlen_k)
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    
    # Get dimensions
    total_q, nheads, head_dim = q.shape
    batch_size = cu_seqlens_q.numel() - 1
    nheads_k = k.shape[1]
    nheads_per_group = nheads // nheads_k
    
    # Reshape query for NSA pattern
    # From (total_q, nsa_group_size * nsa_heads_per_group, head_dim)
    # To (total_q * nsa_group_size, nsa_heads_per_group, head_dim)
    q = q.reshape(total_q, nheads_k, nheads_per_group, head_dim)
    q = q.transpose(1, 2).reshape(total_q * nheads_per_group, nheads_k, head_dim).contiguous()
    
    # Adjust cu_seqlens and max_seqlen for the reshaped query
    cu_seqlens_q_adjusted = cu_seqlens_q * nheads_per_group
    max_seqlen_q_adjusted = max_seqlen_q * nheads_per_group

    # Call the underlying CUDA kernel
    result = infllm_cuda.varlen_fwd_stage1(
        q,
        k,
        v,
        None,
        cu_seqlens_q_adjusted,
        cu_seqlens_k,
        cu_seqlens_v,
        None,
        None,
        block_table,
        alibi_slopes,
        max_seqlen_q_adjusted,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        True,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        True,
        None,
    )
    
    S_dmask = result[0] if isinstance(result, list) else result
    S_dmask = S_dmask[:,:, :max_seqlen_k]
    S_dmask = torch.where(torch.isnan(S_dmask), 0, S_dmask)
    # if return_attn_probs and S_dmask is not None:
    #     # The kernel now returns shape (num_heads_k, total_q, max_seqlen_k)
    #     assert S_dmask.shape == (nheads_k, total_q, max_seqlen_k), \
    #         f"Expected shape ({nheads_k}, {total_q}, {max_seqlen_k}), got {S_dmask.shape}"
        # TODO causal masking with first block -inf
        # Apply causal masking if needed
        # if causal:
        #     # Calculate the stride for masking
        #     stride = nsa_heads_per_group * nsa_group_size
        #     mask_size = stride - 1
            
            # Apply masking based on the output shape
            # S_dmask shape is (num_heads_k, total_q, max_seqlen_k)
            # if S_dmask.shape[1] > mask_size:  # total_q dimension
            #     S_dmask[:, :mask_size, :] = float('-inf')
    
    return S_dmask


def infllmv2_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
    topk_idx=None,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)
    if topk_idx is not None:
        assert topk_idx.dtype == torch.int32
        blockmask, _ = cuda_topk_to_uint64(topk_idx, k_cache.shape[1] if block_table is None else block_table.shape[1] * k_cache.shape[1], 64) # N_BLOCK_DIM=64
    else:
        blockmask = None
    out, softmax_lse = infllm_cuda.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        None,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        num_splits,
        blockmask,
    )
    return (out, softmax_lse) if return_softmax_lse else out