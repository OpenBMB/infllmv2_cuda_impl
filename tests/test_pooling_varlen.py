import torch
import torch.nn.functional as F
import math
import numpy as np
from infllm_v2.max_pooling_1d import max_pooling_1d, max_pooling_1d_varlen
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


def test_varlen_vs_triton():
    """Test varlen max pooling against triton transform_score with multi-batch data"""
    # Load data from multibatch directory
    data_dir = "/cache/suzhou/downloads/multibatch"
    
    print(f"Loading data from {data_dir}")
    attn_score_full = torch.load(f"{data_dir}/attn_score.pt").to(torch.bfloat16)
    cu_seqlens_q = torch.load(f"{data_dir}/cu_seqlens_q.pt")
    cu_seqlens_k = torch.load(f"{data_dir}/cu_seqlens_k.pt")
    max_seqlen_q = torch.load(f"{data_dir}/max_seqlen_q.pt")
    max_seqlen_k = torch.load(f"{data_dir}/max_seqlen_k.pt")
    
    if isinstance(max_seqlen_q, torch.Tensor):
        max_seqlen_q = max_seqlen_q.item()
    if isinstance(max_seqlen_k, torch.Tensor):
        max_seqlen_k = max_seqlen_k.item()
    
    batch_size = cu_seqlens_q.shape[0] - 1
    num_heads = attn_score_full.shape[0]
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()
    
    print(f"Full score tensor shape: {attn_score_full.shape}")
    print(f"Batch size: {batch_size}")
    print(f"Number of heads: {num_heads}")
    print(f"Total queries: {total_q}")
    print(f"Total keys: {total_k}")
    print(f"cu_seqlens_q: {cu_seqlens_q}")
    print(f"cu_seqlens_k: {cu_seqlens_k}")
    print(f"max_seqlen_q: {max_seqlen_q}")
    print(f"max_seqlen_k: {max_seqlen_k}")
    
    # Test parameters
    kernel_size = 32
    kernel_stride = 16
    block_size = 64
    init_blocks = 1
    local_blocks = 32
    cache_len = 0
    
    # Create cache_lens tensor (all zeros for this test)
    cache_lens = torch.zeros(batch_size, dtype=torch.int32, device='cuda')
    
    # 1. Run original transform_score on full data
    print("\n" + "="*60)
    print("Running transform_score (Triton implementation)...")
    print("="*60)
    triton_result = transform_score(
        attn_score_full,
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
    print(f"Triton result shape: {triton_result.shape}")
    
    # 2. Run varlen max pooling
    print("\n" + "="*60)
    print("Running max_pooling_1d_varlen...")
    print("="*60)
    
    # The varlen version expects the same input format as transform_score
    # Input shape: [num_heads, total_q, max_k]
    print(f"Varlen input shape: {attn_score_full.shape}")
    
    varlen_result = max_pooling_1d_varlen(
        attn_score_full,
        cu_seqlens_q,
        cu_seqlens_k,
        cache_lens,
        max_seqlen_q,
        max_seqlen_k,
        local_blocks=local_blocks,
        init_blocks=init_blocks,
        block_size=block_size,
        stride=kernel_stride
    )
    print(f"Varlen result shape: {varlen_result.shape}")
    
    # 3. Compare results
    print("\n" + "="*60)
    print("Comparing varlen results with Triton...")
    print("="*60)
    
    # Both should have the same shape
    if triton_result.shape == varlen_result.shape:
        # Compare values
        abs_diff = torch.abs(triton_result - varlen_result)
        abs_diff_no_nan = torch.where(torch.isnan(abs_diff), torch.zeros_like(abs_diff), abs_diff)
        max_diff = torch.max(abs_diff_no_nan).item()
        mean_diff = torch.mean(abs_diff_no_nan).item()
        
        print(f"Maximum absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")
        
        # Check -inf positions
        triton_neg_inf = torch.isinf(triton_result) & (triton_result < 0)
        varlen_neg_inf = torch.isinf(varlen_result) & (varlen_result < 0)
        neg_inf_match = torch.all(triton_neg_inf == varlen_neg_inf).item()
        
        # Check inf positions
        triton_inf = torch.isinf(triton_result) & (triton_result > 0)
        varlen_inf = torch.isinf(varlen_result) & (varlen_result > 0)
        inf_match = torch.all(triton_inf == varlen_inf).item()
        
        print(f"-inf positions match: {neg_inf_match}")
        print(f"inf positions match: {inf_match}")
        
        # Count differences
        threshold = 1e-5
        num_different = torch.sum(abs_diff_no_nan > threshold).item()
        percentage_different = 100 * num_different / torch.numel(triton_result)
        print(f"Number of elements with difference > {threshold}: {num_different} ({percentage_different:.4f}%)")
        
        if max_diff < 1e-5 and neg_inf_match and inf_match:
            print("\n✅ SUCCESS: Varlen implementation matches Triton!")
        else:
            print("\n⚠️  WARNING: Results differ between implementations")
            
            # Show some examples of differences
            if num_different > 0 and num_different < 100:
                print("\nExamples of differences:")
                diff_positions = torch.nonzero(abs_diff_no_nan > threshold)
                for i in range(min(5, diff_positions.shape[0])):
                    h, q, b = diff_positions[i].tolist()
                    triton_val = triton_result[h, q, b].item()
                    varlen_val = varlen_result[h, q, b].item()
                    print(f"  Position [{h}, {q}, {b}]: Triton={triton_val:.6f}, Varlen={varlen_val:.6f}, Diff={abs(triton_val-varlen_val):.6f}")
    else:
        print(f"❌ ERROR: Shape mismatch! Triton: {triton_result.shape}, Varlen: {varlen_result.shape}")
    
    # 4. Detailed per-batch comparison
    print("\n" + "="*60)
    print("Detailed per-batch comparison (inf, -inf, and other values)...")
    print("="*60)
    
    # Track which batches match for each value type
    inf_matching_batches = []
    neg_inf_matching_batches = []
    finite_matching_batches = []
    inf_mismatching_batches = []
    neg_inf_mismatching_batches = []
    finite_mismatching_batches = []
    
    for b in range(batch_size):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        q_len = q_end - q_start
        
        print(f"\n{'='*50}")
        print(f"Batch {b}: queries [{q_start}:{q_end}], length={q_len}")
        print(f"{'='*50}")
        
        # Extract batch data from both results
        triton_batch = triton_result[:, q_start:q_end, :]
        varlen_batch = varlen_result[:, q_start:q_end, :]
        
        # Identify different value types
        triton_neg_inf_mask = torch.isinf(triton_batch) & (triton_batch < 0)
        triton_pos_inf_mask = torch.isinf(triton_batch) & (triton_batch > 0)
        triton_finite_mask = torch.isfinite(triton_batch)
        
        varlen_neg_inf_mask = torch.isinf(varlen_batch) & (varlen_batch < 0)
        varlen_pos_inf_mask = torch.isinf(varlen_batch) & (varlen_batch > 0)
        varlen_finite_mask = torch.isfinite(varlen_batch)
        
        # Count values
        print(f"\nValue counts:")
        print(f"  Triton - neg_inf: {triton_neg_inf_mask.sum().item()}, pos_inf: {triton_pos_inf_mask.sum().item()}, finite: {triton_finite_mask.sum().item()}")
        print(f"  Varlen - neg_inf: {varlen_neg_inf_mask.sum().item()}, pos_inf: {varlen_pos_inf_mask.sum().item()}, finite: {varlen_finite_mask.sum().item()}")
        
        # 1. Compare -inf positions
        print(f"\n-inf comparison:")
        neg_inf_match = torch.all(triton_neg_inf_mask == varlen_neg_inf_mask).item()
        print(f"  Position match: {neg_inf_match}")
        if neg_inf_match:
            neg_inf_matching_batches.append(b)
        else:
            neg_inf_mismatching_batches.append(b)
            neg_inf_only_triton = triton_neg_inf_mask & ~varlen_neg_inf_mask
            neg_inf_only_varlen = varlen_neg_inf_mask & ~triton_neg_inf_mask
            print(f"  -inf only in Triton: {neg_inf_only_triton.sum().item()}")
            print(f"  -inf only in Varlen: {neg_inf_only_varlen.sum().item()}")
            
            # Show some examples
            if neg_inf_only_triton.any():
                examples = torch.nonzero(neg_inf_only_triton)[:5]
                print("  Examples where Triton has -inf but Varlen doesn't:")
                for idx in examples:
                    h, q, k = idx.tolist()
                    print(f"    [{h}, {q}, {k}]: Triton=-inf, Varlen={varlen_batch[h, q, k].item():.6f}")
            
            if neg_inf_only_varlen.any():
                examples = torch.nonzero(neg_inf_only_varlen)[:5]
                print("  Examples where Varlen has -inf but Triton doesn't:")
                for idx in examples:
                    h, q, k = idx.tolist()
                    print(f"    [{h}, {q}, {k}]: Triton={triton_batch[h, q, k].item():.6f}, Varlen=-inf")
        
        # 2. Compare inf positions
        print(f"\ninf comparison:")
        pos_inf_match = torch.all(triton_pos_inf_mask == varlen_pos_inf_mask).item()
        print(f"  Position match: {pos_inf_match}")
        if pos_inf_match:
            inf_matching_batches.append(b)
        else:
            inf_mismatching_batches.append(b)
            pos_inf_only_triton = triton_pos_inf_mask & ~varlen_pos_inf_mask
            pos_inf_only_varlen = varlen_pos_inf_mask & ~triton_pos_inf_mask
            print(f"  inf only in Triton: {pos_inf_only_triton.sum().item()}")
            print(f"  inf only in Varlen: {pos_inf_only_varlen.sum().item()}")
            
            # Show some examples
            if pos_inf_only_triton.any():
                examples = torch.nonzero(pos_inf_only_triton)[:5]
                print("  Examples where Triton has inf but Varlen doesn't:")
                for idx in examples:
                    h, q, k = idx.tolist()
                    print(f"    [{h}, {q}, {k}]: Triton=inf, Varlen={varlen_batch[h, q, k].item():.6f}")
            
            if pos_inf_only_varlen.any():
                examples = torch.nonzero(pos_inf_only_varlen)[:5]
                print("  Examples where Varlen has inf but Triton doesn't:")
                for idx in examples:
                    h, q, k = idx.tolist()
                    print(f"    [{h}, {q}, {k}]: Triton={triton_batch[h, q, k].item():.6f}, Varlen=inf")
        
        # 3. Compare finite values
        print(f"\nFinite values comparison:")
        # Only compare where both are finite
        both_finite_mask = triton_finite_mask & varlen_finite_mask
        finite_match = True  # Default to True
        max_finite_diff = 0.0
        
        if both_finite_mask.any():
            triton_finite = triton_batch[both_finite_mask]
            varlen_finite = varlen_batch[both_finite_mask]
            
            finite_diff = torch.abs(triton_finite - varlen_finite)
            max_finite_diff = finite_diff.max().item()
            mean_finite_diff = finite_diff.mean().item()
            
            print(f"  Number of finite values in both: {both_finite_mask.sum().item()}")
            print(f"  Max difference: {max_finite_diff}")
            print(f"  Mean difference: {mean_finite_diff}")
            
            # Count differences above threshold
            threshold = 1e-5
            num_diff_above_threshold = (finite_diff > threshold).sum().item()
            percentage = 100 * num_diff_above_threshold / finite_diff.numel()
            print(f"  Values with diff > {threshold}: {num_diff_above_threshold} ({percentage:.2f}%)")
            
            finite_match = max_finite_diff < threshold
            
            # Show examples of large differences
            if num_diff_above_threshold > 0:
                # Get positions with large differences
                both_finite_positions = torch.nonzero(both_finite_mask)
                large_diff_mask = finite_diff > threshold
                large_diff_indices = torch.nonzero(large_diff_mask).squeeze()[:5]
                
                print("  Examples of large differences in finite values:")
                for idx in large_diff_indices:
                    pos = both_finite_positions[idx]
                    h, q, k = pos.tolist()
                    triton_val = triton_batch[h, q, k].item()
                    varlen_val = varlen_batch[h, q, k].item()
                    diff = abs(triton_val - varlen_val)
                    print(f"    [{h}, {q}, {k}]: Triton={triton_val:.6f}, Varlen={varlen_val:.6f}, Diff={diff:.6f}")
        else:
            print(f"  No finite values found in both results")
            # If no finite values to compare, consider it a match
            finite_match = True
        
        if finite_match:
            finite_matching_batches.append(b)
        else:
            finite_mismatching_batches.append(b)
        
        # Summary for this batch
        print(f"\nBatch {b} summary:")
        all_match = neg_inf_match and pos_inf_match and finite_match
        if all_match:
            print(f"  ✅ All values match!")
        else:
            print(f"  ⚠️  Differences found")
    
    # Print overall summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY - Batch Matching Status")
    print("="*60)
    
    print(f"\nTotal batches: {batch_size}")
    
    print(f"\n🔵 inf values:")
    print(f"  Matching batches ({len(inf_matching_batches)}): {inf_matching_batches}")
    print(f"  Mismatching batches ({len(inf_mismatching_batches)}): {inf_mismatching_batches}")
    
    print(f"\n🔴 -inf values:")
    print(f"  Matching batches ({len(neg_inf_matching_batches)}): {neg_inf_matching_batches}")
    print(f"  Mismatching batches ({len(neg_inf_mismatching_batches)}): {neg_inf_mismatching_batches}")
    
    print(f"\n🟢 Finite values:")
    print(f"  Matching batches ({len(finite_matching_batches)}): {finite_matching_batches}")
    print(f"  Mismatching batches ({len(finite_mismatching_batches)}): {finite_mismatching_batches}")
    
    # Overall match status
    all_inf_match = len(inf_mismatching_batches) == 0
    all_neg_inf_match = len(neg_inf_mismatching_batches) == 0
    all_finite_match = len(finite_mismatching_batches) == 0
    
    print(f"\n📊 Overall Status:")
    if all_inf_match and all_neg_inf_match and all_finite_match:
        print(f"  ✅ ALL BATCHES MATCH PERFECTLY!")
    else:
        print(f"  ⚠️  Some batches have differences:")
        if not all_inf_match:
            print(f"    - inf values differ in {len(inf_mismatching_batches)} batches")
        if not all_neg_inf_match:
            print(f"    - -inf values differ in {len(neg_inf_mismatching_batches)} batches")
        if not all_finite_match:
            print(f"    - Finite values differ in {len(finite_mismatching_batches)} batches")
    
    # 5. Also test with fixed-length version for comparison
    print("\n" + "="*60)
    print("Comparing with fixed-length implementation on individual batches...")
    print("="*60)
    
    for b in range(min(2, batch_size)):  # Test first 2 batches
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item() 
        k_end = cu_seqlens_k[b + 1].item()
        
        print(f"\nBatch {b}:")
        
        # Extract batch data
        batch_score = attn_score_full[:, q_start:q_end, k_start:k_end]
        
        # Run fixed-length version
        fixed_result = max_pooling_1d(
            batch_score.contiguous(),
            cache_len=cache_len,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=kernel_stride
        )
        
        # Extract corresponding part from varlen result
        varlen_batch = varlen_result[:, q_start:q_end, :fixed_result.shape[2]]
        
        # Compare
        if fixed_result.shape == varlen_batch.shape:
            batch_diff = torch.abs(fixed_result - varlen_batch)
            batch_diff_no_nan = torch.where(torch.isnan(batch_diff), torch.zeros_like(batch_diff), batch_diff)
            max_batch_diff = torch.max(batch_diff_no_nan).item()
            
            print(f"  Max difference vs fixed-length: {max_batch_diff}")
            if max_batch_diff < 1e-5:
                print(f"  ✓ Matches fixed-length implementation!")
            else:
                print(f"  ⚠️  Differs from fixed-length implementation")


def test_varlen_correctness():
    """Test the varlen implementation works correctly with its updated design"""
    print("\n" + "="*60)
    print("Testing varlen implementation correctness...")
    print("="*60)
    
    # Create a simple test case
    batch_size = 2
    num_heads = 4
    seqlen_qs = [8, 12]
    seqlen_ks = [16, 20]
    
    # Create cumulative sequence lengths
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device='cuda')
    
    for i in range(batch_size):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + seqlen_qs[i]
        cu_seqlens_k[i + 1] = cu_seqlens_k[i] + seqlen_ks[i]
    
    total_q = cu_seqlens_q[-1].item()
    max_seqlen_q = max(seqlen_qs)
    max_seqlen_k = max(seqlen_ks)
    
    # Create input in the correct format [num_heads, total_q, max_k]
    input_tensor = torch.randn(num_heads, total_q, max_seqlen_k, device='cuda', dtype=torch.bfloat16)
    
    # Test parameters
    cache_len = 0
    local_blocks = 4
    init_blocks = 1
    block_size = 64
    stride = 16
    
    # Create cache_lens tensor (all zeros for this test)
    cache_lens = torch.zeros(batch_size, dtype=torch.int32, device='cuda')
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"cu_seqlens_q: {cu_seqlens_q}")
    print(f"cu_seqlens_k: {cu_seqlens_k}")
    print(f"max_seqlen_q: {max_seqlen_q}")
    print(f"max_seqlen_k: {max_seqlen_k}")
    
    try:
        # Run varlen version
        output = max_pooling_1d_varlen(
            input_tensor,
            cu_seqlens_q,
            cu_seqlens_k,
            cache_lens,
            max_seqlen_q,
            max_seqlen_k,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=stride
        )
        
        print(f"✓ Varlen execution successful!")
        print(f"Output shape: {output.shape}")
        
        # Check output properties
        total_len = max_seqlen_q + cache_len
        out_len = (total_len + block_size - 1) // block_size
        expected_shape = (num_heads, total_q, out_len)
        
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        print(f"✓ Output shape is correct!")
        
        # Check for NaN values
        assert not torch.isnan(output).any(), "Output contains NaN values"
        print(f"✓ No NaN values in output!")
        
        # Count inf values
        num_inf = torch.isinf(output).sum().item()
        num_neg_inf = (torch.isinf(output) & (output < 0)).sum().item()
        num_pos_inf = (torch.isinf(output) & (output > 0)).sum().item()
        print(f"Number of inf values: {num_inf} (neg_inf: {num_neg_inf}, pos_inf: {num_pos_inf})")
        
        # Compare with fixed-length version batch by batch
        print("\nComparing with fixed-length implementation...")
        for b in range(batch_size):
            q_start = cu_seqlens_q[b].item()
            q_end = cu_seqlens_q[b + 1].item()
            k_start = cu_seqlens_k[b].item()
            k_end = cu_seqlens_k[b + 1].item()
            
            # Extract batch data
            batch_input = input_tensor[:, q_start:q_end, k_start:k_end]
            
            # Run fixed-length version
            fixed_output = max_pooling_1d(
                batch_input.contiguous(),
                cache_len=cache_len,
                local_blocks=local_blocks,
                init_blocks=init_blocks,
                block_size=block_size,
                stride=stride
            )
            
            # Extract corresponding part from varlen output
            varlen_batch = output[:, q_start:q_end, :fixed_output.shape[2]]
            
            # Compare
            diff = torch.abs(fixed_output - varlen_batch)
            diff_no_nan = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)
            max_diff = torch.max(diff_no_nan).item()
            
            print(f"  Batch {b}: max difference = {max_diff}")
            if max_diff < 1e-5:
                print(f"    ✓ Matches fixed-length implementation!")
            else:
                print(f"    ⚠️  Differs from fixed-length implementation")
        
    except Exception as e:
        print(f"❌ Varlen test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Testing Variable-Length Max Pooling Implementation")
    print("="*60)
    
    # First test basic correctness
    test_varlen_correctness()
    
    # Then compare with Triton implementation
    print("\n\n")
    test_varlen_vs_triton() 