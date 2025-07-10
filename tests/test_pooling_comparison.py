import torch
import torch.nn.functional as F
import math
import numpy as np
from infllm_v2.max_pooling_1d import max_pooling_1d
import triton
import triton.language as tl


@triton.jit
def _transform_score_kernel(
    s_ptr,  # score, shape: [num_heads, q_len, k_len]
    bs_ptr,  # block wise score: [num_heads, q_len, num_k_block]
    offs,
    cu_seqlens_q,
    # shape
    num_heads,
    num_offs,
    max_k_len,
    max_blocks,
    pad_len,
    # kernel & block size
    block_size,
    block_stride,  # block_size // kernel_stride
    init_blocks,
    local_blocks,
    # stride
    stride_sh,
    stride_sq,
    stride_sk,
    stride_bsh,
    stride_bsq,
    stride_bsk,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = pid_k * BLOCK_SIZE_K
    if pid_q * BLOCK_SIZE_Q >= q_len:
        return
    # load weight
    off_o = tl.arange(0, BLOCK_SIZE_O)
    w = tl.load(offs + off_o, mask=off_o < num_offs, other=0)
    # load score
    off_q = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    off_k = (k_start + tl.arange(0, BLOCK_SIZE_K)) * block_stride - pad_len
    off_k = off_k[None, :] + off_o[:, None]
    s_ptrs = (
        s_ptr
        + q_start * stride_sq
        + pid_h * stride_sh
        + off_q[:, None, None] * stride_sq
        + off_k[None, :, :] * stride_sk
    )
    # weighted sum, [BQ, BO, BK] * [1, BO, 1] -> [BQ, BO, BK] -> [BQ, BK]
    s = tl.load(
        s_ptrs,
        mask=(off_q < q_len)[:, None, None] & (off_k >= 0) & (off_k < max_k_len),
        other=0,
    )
    s = s * w[None, :, None]
    s = tl.max(s, axis=1)
    # init mask and local mask
    off_bq = off_q // block_size
    off_bk = k_start + tl.arange(0, BLOCK_SIZE_K)
    s = tl.where(
        (off_bq[:, None] >= off_bk[None, :])  # causal mask
            & (off_bq[:, None] < off_bk[None, :] + local_blocks),  # local window
        float("-inf"),
        s,
    )
    s = tl.where(        
        (off_bk[None, :] < init_blocks),  # init window
        float("inf"),
        s,
    )
    # store block wise score
    bs_ptrs = (
        bs_ptr
        + q_start * stride_bsq
        + pid_h * stride_bsh
        + off_q[:, None] * stride_bsq
        + off_bk[None, :] * stride_bsk
    )
    tl.store(
        bs_ptrs,
        s,
        mask=(off_q < q_len)[:, None] & (off_bk < max_blocks)[None, :],
    )


def transform_score(
    score: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
) -> torch.Tensor:
    num_k_heads, total_query_len, max_key_len = score.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    pad_len = kernel_size // kernel_stride - 1
    max_blocks = math.ceil(max_seqlen_q / block_size)
    block_score = torch.zeros(
        num_k_heads,
        total_query_len,
        max_blocks,
        dtype=torch.float32,
        device=score.device,
    )
    offs = (
        torch.arange(kernel_size // kernel_stride, device=score.device)[:, None]
        + torch.arange(block_size // kernel_stride, device=score.device)[None, :]
    ).view(-1)
    offs = torch.histc(offs, bins=offs.max() + 1, min=0, max=offs.max())
    num_offs = int(offs.shape[0])
    BLOCK_SIZE_K = min(128, triton.next_power_of_2(max_blocks))
    BLOCK_SIZE_O = triton.next_power_of_2(num_offs)
    BLOCK_SIZE_Q = 8
    grid = (
        num_k_heads * batch_size,
        triton.cdiv(total_query_len, BLOCK_SIZE_Q),
        triton.cdiv(max_blocks, BLOCK_SIZE_K),
    )
    _transform_score_kernel[grid](
        score,
        block_score,
        torch.ones_like(offs, dtype = offs.dtype, device = offs.device),
        cu_seqlens_q,
        num_k_heads,
        offs.shape[0],
        max_key_len,
        max_blocks,
        pad_len,
        block_size,
        block_size // kernel_stride,
        init_blocks,
        local_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_score.stride(0),
        block_score.stride(1),
        block_score.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_O=BLOCK_SIZE_O,
        num_warps=8,
        num_stages=3,
    )
    return block_score




def test_pooling_functions():
    # Load the saved score tensor
    print("Loading score tensor from  /cache/suzhou/downloads/stage1_score.pt")
    score = torch.load("/cache/suzhou/downloads/stage1_score.pt")
    print(f"Score tensor shape: {score.shape}")
    
    # Test parameters (use the same parameters for both functions)
    kernel_size = 32
    kernel_stride = 16
    block_size = 64
    init_blocks = 1
    local_blocks = 32
    
    # Create dummy cu_seqlens for testing
    batch_size = 1
    total_query_len = score.shape[1]
    total_key_len = score.shape[2]
    
    cu_seqlens_q = torch.tensor([0, total_query_len], device=score.device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, total_key_len], device=score.device, dtype=torch.int32)
    max_seqlen_q = total_query_len
    max_seqlen_k = total_key_len
    total_seq_lens = total_key_len
    print("\nRunning transform_score...")
    # Run the original transform_score function
    original_result = transform_score(
        score,
        kernel_size,
        kernel_stride,
        block_size,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        
    )
    
    print("\nRunning max_pooling_1d...")
    # Run the new max_pooling_1d function
    new_result = max_pooling_1d(
        score,
        cache_len=0,
        local_blocks=local_blocks,
        init_blocks=init_blocks,
        block_size=block_size,
        stride=kernel_stride,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    
    # Compare shapes
    print(f"\nComparing shapes:")
    print(f"Original result shape: {original_result.shape}")
    print(f"New result shape: {new_result.shape}")
    
    # Compare values
    print("\nComparing values:")
    if original_result.shape == new_result.shape:
        # Check if values are close
        abs_diff = torch.abs(original_result - new_result)
        
        # Replace NaN values (resulting from inf - inf) with 0 for computing statistics
        abs_diff_no_nan = torch.where(torch.isnan(abs_diff), torch.zeros_like(abs_diff), abs_diff)
        max_diff = torch.max(abs_diff_no_nan).item()
        mean_diff = torch.mean(abs_diff_no_nan).item()
        
        print(f"Maximum absolute difference (excluding NaNs): {max_diff}")
        print(f"Mean absolute difference (excluding NaNs): {mean_diff}")
        
        # Count number of significant differences (threshold: 1e-5)
        threshold = 1e-5
        num_different = torch.sum(abs_diff_no_nan > threshold).item()
        percentage_different = 100 * num_different / torch.numel(original_result)
        print(f"Number of elements with difference > {threshold}: {num_different} ({percentage_different:.4f}%)")
        
        # Specifically compare the -inf positions
        print("\nComparing -inf positions:")
        orig_neg_inf = torch.isinf(original_result) & (original_result < 0)
        new_neg_inf = torch.isinf(new_result) & (new_result < 0)
        
        # Check if -inf positions match
        neg_inf_match = orig_neg_inf == new_neg_inf
        total_neg_inf_positions = torch.sum(orig_neg_inf | new_neg_inf).item()
        matching_neg_inf_positions = torch.sum(neg_inf_match & (orig_neg_inf | new_neg_inf)).item()
        
        print(f"Total -inf positions in either result: {total_neg_inf_positions}")
        print(f"Number of -inf positions in original result: {torch.sum(orig_neg_inf).item()}")
        print(f"Number of -inf positions in new result: {torch.sum(new_neg_inf).item()}")
        print(f"Number of matching -inf positions: {matching_neg_inf_positions}")
        
        if torch.all(neg_inf_match):
            print("All -inf positions match exactly!")
        else:
            # Find positions where -inf doesn't match
            mismatch_positions = ~neg_inf_match & (orig_neg_inf | new_neg_inf)
            num_mismatches = torch.sum(mismatch_positions).item()
            print(f"Found {num_mismatches} mismatched -inf positions")
            
            # Print some examples of mismatches
            if num_mismatches > 0:
                print("\nExamples of -inf mismatches:")
                
                flat_indices = torch.nonzero(mismatch_positions.flatten()).flatten()
                indices = np.array(np.unravel_index(flat_indices.cpu().numpy(), mismatch_positions.shape)).T
                for i, idx in enumerate(indices):
                    h, q, b = idx
                    orig_val = original_result[h, q, b].item()
                    new_val = new_result[h, q, b].item()
                    # print(f"  {i+1}. Position {idx}: Original={orig_val}, New={new_val}")
        
        # Compare for inf values as well
        print("\nComparing inf positions:")
        orig_inf = torch.isinf(original_result) & (original_result > 0)
        new_inf = torch.isinf(new_result) & (new_result > 0)
        
        # Check if inf positions match
        inf_match = orig_inf == new_inf
        total_inf_positions = torch.sum(orig_inf | new_inf).item()
        matching_inf_positions = torch.sum(inf_match & (orig_inf | new_inf)).item()
        
        print(f"Total inf positions in either result: {total_inf_positions}")
        print(f"Number of inf positions in original result: {torch.sum(orig_inf).item()}")
        print(f"Number of inf positions in new result: {torch.sum(new_inf).item()}")
        print(f"Number of matching inf positions: {matching_inf_positions}")
        
        if torch.all(inf_match):
            print("All inf positions match exactly!")
        else:
            # Find positions where inf doesn't match
            mismatch_positions = ~inf_match & (orig_inf | new_inf)
            num_mismatches = torch.sum(mismatch_positions).item()
            print(f"Found {num_mismatches} mismatched inf positions")
            
            # Print some examples of mismatches
            if num_mismatches > 0:
                print("\nExamples of inf mismatches:")
                flat_indices = torch.nonzero(mismatch_positions.flatten())[:10].flatten()
                indices = np.array(np.unravel_index(flat_indices.cpu().numpy(), mismatch_positions.shape)).T
                
                for i, idx in enumerate(indices):
                    h, q, b = idx
                    orig_val = original_result[h, q, b].item()
                    new_val = new_result[h, q, b].item()
                    print(f"  {i+1}. Position {idx}: Original={orig_val}, New={new_val}")
        
        # Compare non-inf values
        print("\nComparing non-infinite values:")
        non_inf_mask = ~torch.isinf(original_result) & ~torch.isinf(new_result)
        non_inf_count = torch.sum(non_inf_mask).item()
        
        if non_inf_count > 0:
            non_inf_diff = torch.abs(original_result[non_inf_mask] - new_result[non_inf_mask])
            non_inf_max_diff = torch.max(non_inf_diff).item()
            non_inf_mean_diff = torch.mean(non_inf_diff).item()
            print(f"Number of positions with non-infinite values in both results: {non_inf_count}")
            print(f"Maximum difference among non-infinite values: {non_inf_max_diff}")
            print(f"Mean difference among non-infinite values: {non_inf_mean_diff}")
        else:
            print("No positions with non-infinite values in both results")
        
    else:
        print("Cannot compare values because shapes are different")


def test_pooling_functions_multibatch():
    # Load data from multibatch directory
    data_dir = "/cache/suzhou/downloads/multibatch"
    
    print(f"Loading data from {data_dir}")
    attn_score = torch.load(f"{data_dir}/attn_score.pt").to(torch.bfloat16)
    cu_seqlens_q = torch.load(f"{data_dir}/cu_seqlens_q.pt")
    cu_seqlens_k = torch.load(f"{data_dir}/cu_seqlens_k.pt")
    max_seqlen_q = torch.load(f"{data_dir}/max_seqlen_q.pt")
    max_seqlen_k = torch.load(f"{data_dir}/max_seqlen_k.pt")
    
    print(f"Score tensor shape: {attn_score.shape}")
    print(f"cu_seqlens_q: {cu_seqlens_q}")
    print(f"cu_seqlens_k: {cu_seqlens_k}")
    print(f"max_seqlen_q: {max_seqlen_q}")
    print(f"max_seqlen_k: {max_seqlen_k}")
    
    # Test parameters (use the same parameters for both functions)
    kernel_size = 32
    kernel_stride = 16
    block_size = 64
    init_blocks = 1
    local_blocks = 32
    
    # Extract dimensions
    num_heads = attn_score.shape[0]
    total_query_len = attn_score.shape[1]
    total_key_len = attn_score.shape[2]
    batch_size = cu_seqlens_q.shape[0] - 1
    
    print(f"\nTest configuration:")
    print(f"batch_size: {batch_size}")
    print(f"num_heads: {num_heads}")
    print(f"total_query_len: {total_query_len}")
    print(f"total_key_len: {total_key_len}")
    
    print("\nRunning transform_score...")
    # Run the original transform_score function
    original_result = transform_score(
        attn_score,
        kernel_size,
        kernel_stride,
        block_size,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )
    
    print("\nRunning max_pooling_1d...")
    # Run the new max_pooling_1d function
    new_result = max_pooling_1d(
        attn_score,
        cache_len=0,
        local_blocks=local_blocks,
        init_blocks=init_blocks,
        block_size=block_size,
        stride=kernel_stride,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    
    # Compare shapes
    print(f"\nComparing shapes:")
    print(f"Original result shape: {original_result.shape}")
    print(f"New result shape: {new_result.shape}")
    
    # Compare values
    print("\nComparing values:")
    if original_result.shape == new_result.shape:
        # Check if values are close
        abs_diff = torch.abs(original_result - new_result)
        
        # Replace NaN values (resulting from inf - inf) with 0 for computing statistics
        abs_diff_no_nan = torch.where(torch.isnan(abs_diff), torch.zeros_like(abs_diff), abs_diff)
        max_diff = torch.max(abs_diff_no_nan).item()
        mean_diff = torch.mean(abs_diff_no_nan).item()
        
        print(f"Maximum absolute difference (excluding NaNs): {max_diff}")
        print(f"Mean absolute difference (excluding NaNs): {mean_diff}")
        
        # Count number of significant differences (threshold: 1e-5)
        threshold = 1e-5
        num_different = torch.sum(abs_diff_no_nan > threshold).item()
        percentage_different = 100 * num_different / torch.numel(original_result)
        print(f"Number of elements with difference > {threshold}: {num_different} ({percentage_different:.4f}%)")
        
        # Analyze per batch
        print("\nPer-batch analysis:")
        for b in range(batch_size):
            q_start = int(cu_seqlens_q[b].item())
            q_end = int(cu_seqlens_q[b+1].item())
            
            batch_orig = original_result[:, q_start:q_end, :]
            batch_new = new_result[:, q_start:q_end, :]
            
            batch_diff = torch.abs(batch_orig - batch_new)
            batch_diff_no_nan = torch.where(torch.isnan(batch_diff), torch.zeros_like(batch_diff), batch_diff)
            
            print(f"\nBatch {b} (queries {q_start}-{q_end}):")
            print(f"  Max difference: {torch.max(batch_diff_no_nan).item()}")
            print(f"  Mean difference: {torch.mean(batch_diff_no_nan).item()}")
            
            # Check inf positions
            orig_inf = torch.isinf(batch_orig)
            new_inf = torch.isinf(batch_new)
            inf_match = torch.all(orig_inf == new_inf)
            print(f"  Inf positions match: {inf_match}")
            
            if not inf_match:
                mismatch_count = torch.sum(orig_inf != new_inf).item()
                print(f"  Number of inf mismatches: {mismatch_count}")
        
        # Save results
        torch.save(original_result, f"{data_dir}/transform_score_output.pt")
        torch.save(new_result, f"{data_dir}/max_pooling_1d_output.pt")
        print(f"\nResults saved to {data_dir}")
        
    else:
        print("Cannot compare values because shapes are different")


def test_pooling_functions_single_batch_from_multibatch():
    """Test with only the first batch from multi-batch data"""
    # Load data from multibatch directory
    data_dir = "/cache/suzhou/downloads/multibatch"
    
    print(f"Loading data from {data_dir}")
    attn_score_full = torch.load(f"{data_dir}/attn_score.pt").to(torch.bfloat16)
    cu_seqlens_q_full = torch.load(f"{data_dir}/cu_seqlens_q.pt")
    cu_seqlens_k_full = torch.load(f"{data_dir}/cu_seqlens_k.pt")
    max_seqlen_q = torch.load(f"{data_dir}/max_seqlen_q.pt")
    max_seqlen_k = torch.load(f"{data_dir}/max_seqlen_k.pt")
    
    print(f"Full score tensor shape: {attn_score_full.shape}")
    print(f"Full cu_seqlens_q: {cu_seqlens_q_full}")
    print(f"Full cu_seqlens_k: {cu_seqlens_k_full}")
    
    # Extract only the first batch
    batch_idx = 0
    q_start = int(cu_seqlens_q_full[batch_idx].item())
    q_end = int(cu_seqlens_q_full[batch_idx + 1].item())
    k_start = int(cu_seqlens_k_full[batch_idx].item())
    k_end = int(cu_seqlens_k_full[batch_idx + 1].item())
    
    # Slice the attention score for the first batch
    attn_score = attn_score_full[:, q_start:q_end, k_start:k_end]
    
    # Create new cu_seqlens for single batch
    cu_seqlens_q = torch.tensor([0, q_end - q_start], device=attn_score.device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, k_end - k_start], device=attn_score.device, dtype=torch.int32)
    
    # Update max_seqlen for single batch
    single_batch_max_seqlen_q = q_end - q_start
    single_batch_max_seqlen_k = k_end - k_start
    
    print(f"\nSingle batch configuration:")
    print(f"Batch index: {batch_idx}")
    print(f"Query range: {q_start}-{q_end} (length: {q_end - q_start})")
    print(f"Key range: {k_start}-{k_end} (length: {k_end - k_start})")
    print(f"Single batch score tensor shape: {attn_score.shape}")
    print(f"Single batch cu_seqlens_q: {cu_seqlens_q}")
    print(f"Single batch cu_seqlens_k: {cu_seqlens_k}")
    
    # Test parameters (use the same parameters for both functions)
    kernel_size = 32
    kernel_stride = 16
    block_size = 64
    init_blocks = 1
    local_blocks = 32
    
    print("\nRunning transform_score on single batch...")
    # Run the original transform_score function
    original_result = transform_score(
        attn_score,
        kernel_size,
        kernel_stride,
        block_size,
        cu_seqlens_q,
        cu_seqlens_k,
        single_batch_max_seqlen_q,
        single_batch_max_seqlen_k,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )
    
    print("\nRunning max_pooling_1d on single batch...")
    # Run the new max_pooling_1d function
    new_result = max_pooling_1d(
        attn_score,
        cache_len=0,
        local_blocks=local_blocks,
        init_blocks=init_blocks,
        block_size=block_size,
        stride=kernel_stride,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=single_batch_max_seqlen_q,
        max_seqlen_k=single_batch_max_seqlen_k,
    )
    
    # Compare shapes
    print(f"\nComparing shapes:")
    print(f"Original result shape: {original_result.shape}")
    print(f"New result shape: {new_result.shape}")
    
    # Compare values
    print("\nComparing values:")
    if original_result.shape == new_result.shape:
        # Check if values are close
        abs_diff = torch.abs(original_result - new_result)
        
        # Replace NaN values (resulting from inf - inf) with 0 for computing statistics
        abs_diff_no_nan = torch.where(torch.isnan(abs_diff), torch.zeros_like(abs_diff), abs_diff)
        max_diff = torch.max(abs_diff_no_nan).item()
        mean_diff = torch.mean(abs_diff_no_nan).item()
        
        print(f"Maximum absolute difference (excluding NaNs): {max_diff}")
        print(f"Mean absolute difference (excluding NaNs): {mean_diff}")
        
        # Count number of significant differences (threshold: 1e-5)
        threshold = 1e-5
        num_different = torch.sum(abs_diff_no_nan > threshold).item()
        percentage_different = 100 * num_different / torch.numel(original_result)
        print(f"Number of elements with difference > {threshold}: {num_different} ({percentage_different:.4f}%)")
        
        # Specifically compare the -inf positions
        print("\nComparing -inf positions:")
        orig_neg_inf = torch.isinf(original_result) & (original_result < 0)
        new_neg_inf = torch.isinf(new_result) & (new_result < 0)
        
        # Check if -inf positions match
        neg_inf_match = orig_neg_inf == new_neg_inf
        total_neg_inf_positions = torch.sum(orig_neg_inf | new_neg_inf).item()
        matching_neg_inf_positions = torch.sum(neg_inf_match & (orig_neg_inf | new_neg_inf)).item()
        
        print(f"Total -inf positions in either result: {total_neg_inf_positions}")
        print(f"Number of -inf positions in original result: {torch.sum(orig_neg_inf).item()}")
        print(f"Number of -inf positions in new result: {torch.sum(new_neg_inf).item()}")
        print(f"Number of matching -inf positions: {matching_neg_inf_positions}")
        
        if torch.all(neg_inf_match):
            print("All -inf positions match exactly!")
        else:
            print("Found mismatched -inf positions")
        
        # Compare for inf values as well
        print("\nComparing inf positions:")
        orig_inf = torch.isinf(original_result) & (original_result > 0)
        new_inf = torch.isinf(new_result) & (new_result > 0)
        
        # Check if inf positions match
        inf_match = orig_inf == new_inf
        total_inf_positions = torch.sum(orig_inf | new_inf).item()
        matching_inf_positions = torch.sum(inf_match & (orig_inf | new_inf)).item()
        
        print(f"Total inf positions in either result: {total_inf_positions}")
        print(f"Number of inf positions in original result: {torch.sum(orig_inf).item()}")
        print(f"Number of inf positions in new result: {torch.sum(new_inf).item()}")
        print(f"Number of matching inf positions: {matching_inf_positions}")
        
        if torch.all(inf_match):
            print("All inf positions match exactly!")
        else:
            print("Found mismatched inf positions")
        
        # Compare non-inf values
        print("\nComparing non-infinite values:")
        non_inf_mask = ~torch.isinf(original_result) & ~torch.isinf(new_result)
        non_inf_count = torch.sum(non_inf_mask).item()
        
        if non_inf_count > 0:
            non_inf_diff = torch.abs(original_result[non_inf_mask] - new_result[non_inf_mask])
            non_inf_max_diff = torch.max(non_inf_diff).item()
            non_inf_mean_diff = torch.mean(non_inf_diff).item()
            print(f"Number of positions with non-infinite values in both results: {non_inf_count}")
            print(f"Maximum difference among non-infinite values: {non_inf_max_diff}")
            print(f"Mean difference among non-infinite values: {non_inf_mean_diff}")
        else:
            print("No positions with non-infinite values in both results")
        
        # Save results for single batch
        torch.save(original_result, f"{data_dir}/transform_score_output_single_batch.pt")
        torch.save(new_result, f"{data_dir}/max_pooling_1d_output_single_batch.pt")
        print(f"\nSingle batch results saved to {data_dir}")
        
    else:
        print("Cannot compare values because shapes are different")


if __name__ == "__main__":
    import sys
    
    # Test with only the first batch from multi-batch data
    print("Testing with single batch (first batch from multi-batch data)")
    test_pooling_functions_single_batch_from_multibatch() 