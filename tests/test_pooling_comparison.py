import torch
import torch.nn.functional as F
import math
import numpy as np
from infllm_v2.max_pooling_1d import max_pooling_1d
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


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
    print("\nRunning transform_score...")

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
    
    print("\nRunning max_pooling_1d...")
    # Run the new max_pooling_1d function
    new_result = max_pooling_1d(
        attn_score,
        cache_len=0,
        local_blocks=local_blocks,
        init_blocks=init_blocks,
        block_size=block_size,
        stride=kernel_stride
    )

    # Compare shapes
    print(f"\nComparing shapes:")
    print(f"attn_score shape: {attn_score.shape}")
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
            
            # Find the position with maximum difference
            max_diff_idx = torch.argmax(non_inf_diff)
            
            # Get the indices in the original tensor
            non_inf_indices = torch.nonzero(non_inf_mask)
            max_diff_position = non_inf_indices[max_diff_idx]
            h, q, b = max_diff_position.tolist()
            
            # Get the values at this position
            orig_val = original_result[h, q, b].item()
            new_val = new_result[h, q, b].item()
            diff_val = new_val - orig_val
            
            print(f"\nPosition with maximum difference:")
            print(f"  Indices: [head={h}, query={q}, block={b}]")
            print(f"  Original value: {orig_val}")
            print(f"  New value: {new_val}")
            print(f"  Difference (new - original): {diff_val}")
            print(f"  Absolute difference: {abs(diff_val)}")
            
            # Also find top 5 positions with largest differences
            print(f"\nTop 5 positions with largest absolute differences:")
            top_k = min(5, non_inf_diff.shape[0])
            top_diffs, top_indices = torch.topk(non_inf_diff, top_k)
            
            for i in range(top_k):
                idx = top_indices[i]
                position = non_inf_indices[idx]
                h, q, b = position.tolist()
                orig = original_result[h, q, b].item()
                new = new_result[h, q, b].item()
                diff = new - orig
                print(f"  {i+1}. [h={h}, q={q}, b={b}]: orig={orig:.4f}, new={new:.4f}, diff={diff:.4f}, |diff|={abs(diff):.4f}")
            breakpoint()
            # Visualize non-infinite values
            visualize_non_infinite_values(original_result, new_result, non_inf_mask)
        else:
            print("No positions with non-infinite values in both results")
        
    else:
        print("Cannot compare values because shapes are different")


def visualize_non_infinite_values(original_result, new_result, non_inf_mask):
    """Visualize the comparison of non-infinite values between two results"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # Extract non-infinite values
    orig_non_inf = original_result[non_inf_mask]
    new_non_inf = new_result[non_inf_mask]
    
    # Calculate differences
    differences = (new_result - original_result)[non_inf_mask]
    abs_differences = torch.abs(differences)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Non-Infinite Values Comparison', fontsize=16)
    
    # 1. Scatter plot of original vs new values
    ax = axes[0, 0]
    ax.scatter(orig_non_inf.float().cpu().numpy(), new_non_inf.float().cpu().numpy(), 
               alpha=0.5, s=1, color='blue')
    min_val = min(orig_non_inf.min().item(), new_non_inf.min().item())
    max_val = max(orig_non_inf.max().item(), new_non_inf.max().item())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    ax.set_xlabel('Original Result')
    ax.set_ylabel('New Result')
    ax.set_title('Original vs New Values Scatter Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Histogram of absolute differences
    ax = axes[0, 1]
    abs_diff_numpy = abs_differences.cpu().numpy()
    n_bins = 50
    ax.hist(abs_diff_numpy, bins=n_bins, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Absolute Difference')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Absolute Differences')
    ax.set_yscale('log')  # Use log scale for better visibility
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Max: {abs_differences.max().item():.4f}\n'
    stats_text += f'Mean: {abs_differences.mean().item():.4f}\n'
    stats_text += f'Std: {abs_differences.std().item():.4f}\n'
    stats_text += f'Median: {abs_differences.median().item():.4f}'
    ax.text(0.7, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Histogram of signed differences
    ax = axes[0, 2]
    diff_numpy = differences.cpu().numpy()
    ax.hist(diff_numpy, bins=n_bins, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Signed Difference (New - Original)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Signed Differences')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 4. Spatial distribution of non-infinite values (heatmap)
    ax = axes[1, 0]
    # Get the first head for visualization
    head_idx = 0
    non_inf_2d = non_inf_mask[head_idx].cpu().numpy()
    im1 = ax.imshow(non_inf_2d, cmap='Blues', aspect='auto')
    ax.set_xlabel('Block Index')
    ax.set_ylabel('Query Index')
    ax.set_title(f'Non-Infinite Value Locations (Head {head_idx})')
    plt.colorbar(im1, ax=ax, label='Has Non-Inf Value')
    
    # 5. Spatial distribution of differences (heatmap)
    ax = axes[1, 1]
    # Create a 2D array for differences
    diff_2d = torch.zeros_like(original_result[head_idx])
    diff_2d[non_inf_mask[head_idx]] = (new_result[head_idx] - original_result[head_idx])[non_inf_mask[head_idx]]
    diff_2d_numpy = diff_2d.cpu().numpy()
    
    # Use a diverging colormap centered at 0
    vmax = max(abs(diff_2d_numpy.min()), abs(diff_2d_numpy.max()))
    im2 = ax.imshow(diff_2d_numpy, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Block Index')
    ax.set_ylabel('Query Index')
    ax.set_title(f'Spatial Distribution of Differences (Head {head_idx})')
    plt.colorbar(im2, ax=ax, label='Difference (New - Original)')
    
    # 6. Box plot of differences by magnitude ranges
    ax = axes[1, 2]
    # Categorize differences by magnitude
    diff_ranges = [
        (0, 0.001, '0-0.001'),
        (0.001, 0.01, '0.001-0.01'),
        (0.01, 0.1, '0.01-0.1'),
        (0.1, 1.0, '0.1-1.0'),
        (1.0, float('inf'), '>1.0')
    ]
    
    range_data = []
    range_labels = []
    for min_val, max_val, label in diff_ranges:
        mask = (abs_differences >= min_val) & (abs_differences < max_val)
        if mask.any():
            range_data.append(differences[mask].cpu().numpy())
            range_labels.append(f'{label}\n(n={mask.sum().item()})')
    
    if range_data:
        ax.boxplot(range_data, labels=range_labels)
        ax.set_ylabel('Signed Difference')
        ax.set_xlabel('Absolute Difference Range')
        ax.set_title('Differences Grouped by Magnitude')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('non_infinite_values_comparison.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'non_infinite_values_comparison.png'")
    
    # Create an additional detailed view for large differences
    large_diff_threshold = 1.0
    large_diff_mask = abs_differences > large_diff_threshold
    
    if large_diff_mask.any():
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        fig2.suptitle(f'Analysis of Large Differences (|diff| > {large_diff_threshold})', fontsize=14)
        
        # Find positions with large differences
        large_diff_positions = torch.nonzero(non_inf_mask)
        large_diff_filter = abs_differences > large_diff_threshold
        large_diff_positions = large_diff_positions[large_diff_filter]
        
        # Plot distribution of large differences
        ax = axes2[0]
        large_diffs = differences[large_diff_mask].cpu().numpy()
        ax.hist(large_diffs, bins=30, edgecolor='black', alpha=0.7, color='red')
        ax.set_xlabel('Difference Value')
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of Large Differences (n={len(large_diffs)})')
        ax.grid(True, alpha=0.3)
        
        # Show spatial locations of large differences
        ax = axes2[1]
        # Create scatter plot of positions
        if len(large_diff_positions) > 0:
            heads = large_diff_positions[:, 0].cpu().numpy()
            queries = large_diff_positions[:, 1].cpu().numpy()
            blocks = large_diff_positions[:, 2].cpu().numpy()
            
            # Use color to indicate the magnitude of difference
            diff_values = differences[large_diff_mask].cpu().numpy()
            scatter = ax.scatter(blocks, queries, c=diff_values, cmap='RdBu_r', 
                               s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax.set_xlabel('Block Index')
            ax.set_ylabel('Query Index')
            ax.set_title(f'Positions of Large Differences (All Heads)')
            plt.colorbar(scatter, ax=ax, label='Difference Value')
            
            # Add text with position statistics
            pos_stats = f'Heads: {np.unique(heads).tolist()[:5]}{"..." if len(np.unique(heads)) > 5 else ""}\n'
            pos_stats += f'Query range: [{queries.min()}, {queries.max()}]\n'
            pos_stats += f'Block range: [{blocks.min()}, {blocks.max()}]'
            ax.text(0.02, 0.98, pos_stats, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('large_differences_analysis.png', dpi=300, bbox_inches='tight')
        print("Large differences analysis saved as 'large_differences_analysis.png'")
    
    plt.show()


if __name__ == "__main__":
    test_pooling_functions() 