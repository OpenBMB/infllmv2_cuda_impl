# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py

import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import numpy as np
torch.set_printoptions(profile="full")

# Create output directory for visualizations
VISUALIZATION_DIR = Path("/user/qiqi/FA_NSA/gradient_visualizations")
VISUALIZATION_DIR.mkdir(exist_ok=True)


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def _get_block_size(device, head_dim, is_dropout, causal):
    """
    替代函数：返回固定的块大小，与infllm_v2库中使用的相同
    """
    # 使用与infllm_v2相同的固定块大小
    blocksize_m = 16  # m_block_dim
    blocksize_n = 64  # n_block_dim
    return blocksize_m, blocksize_n


def pad_input(tensor_unpad, indices, batch_size, seqlen):
    """
    简化的pad_input替代函数
    将unpadded tensor重新padding为指定的shape
    """
    # 这是一个简化版本，适用于当前的使用场景
    total_tokens, *other_dims = tensor_unpad.shape
    output_shape = [batch_size, seqlen] + other_dims
    output = torch.zeros(output_shape, dtype=tensor_unpad.dtype, device=tensor_unpad.device)
    
    # 使用indices重建tensor
    # 注意：这是一个简化实现，可能需要根据具体使用场景调整
    if indices is not None and len(indices) == total_tokens:
        # 将tokens放回到正确的位置
        for i, (batch_idx, seq_idx) in enumerate(indices):
            if batch_idx < batch_size and seq_idx < seqlen:
                output[batch_idx, seq_idx] = tensor_unpad[i]
    else:
        # 如果没有indices，按顺序填充（假设是连续的）
        tokens_per_batch = total_tokens // batch_size
        for b in range(batch_size):
            start_idx = b * tokens_per_batch
            end_idx = min((b + 1) * tokens_per_batch, total_tokens)
            actual_len = min(end_idx - start_idx, seqlen)
            if actual_len > 0:
                output[b, :actual_len] = tensor_unpad[start_idx:start_idx + actual_len]
    
    return output


def unpad_input(input_tensor, attention_mask):
    """
    简化的unpad_input替代函数
    根据attention mask移除padding
    """
    batch_size, seqlen = attention_mask.shape
    # 获取有效token的位置
    valid_mask = attention_mask.bool()
    
    # 计算cumulative lengths
    seqlens = valid_mask.sum(dim=1)
    cu_seqlens = torch.cat([torch.tensor([0], device=input_tensor.device), seqlens.cumsum(dim=0)])
    
    # 获取有效tokens的indices
    indices = []
    for b in range(batch_size):
        for s in range(seqlen):
            if valid_mask[b, s]:
                indices.append((b, s))
    
    # 提取有效tokens
    output_list = []
    for b, s in indices:
        output_list.append(input_tensor[b, s])
    
    output_unpad = torch.stack(output_list) if output_list else torch.empty((0,) + input_tensor.shape[2:], device=input_tensor.device, dtype=input_tensor.dtype)
    max_seqlen = seqlens.max().item()
    
    return output_unpad, indices, cu_seqlens.to(torch.int32), max_seqlen


def generate_topk_indices(nheads_k, total_seqlen, seqlen_k, sparsity, block_size, device):
    """
    Generate random topk indices for block sparse attention.
    
    Args:
        nheads_k: Number of key heads
        total_seqlen: Total sequence length (batch * seqlen_q)
        seqlen_k: Key sequence length
        sparsity: Sparsity level (0.0 to 1.0)
        block_size: Size of each block
        device: Device to create the indices on
        
    Returns:
        torch.Tensor: Topk indices with shape [nheads_k, total_seqlen, k]
    """
    # Calculate number of blocks in key dimension
    k_blocks = (seqlen_k + block_size - 1) // block_size
    
    # Calculate how many blocks to keep (top-k)
    k = max(0, int(k_blocks * (1 - sparsity)))
    
    # Option 1: Generate random valid indices for each query position
    # For each head and query position, sample k random indices from 0 to k_blocks-1
    indices = torch.stack([
        torch.stack([
            torch.randperm(k_blocks, device=device)[:k]
            for _ in range(total_seqlen)
        ])
        for _ in range(nheads_k)
    ])
    
    # Make sure indices are sorted for better performance
    indices = indices.sort(dim=-1)[0].to(torch.int32)
    
    return indices

def generate_batch_topk_indices(nheads_k, batch_size, seqlen_q, seqlen_k, num_topk, block_size, device, mode="random"):
    """
    Generate random topk indices for block sparse attention with explicit batch handling.
    
    Args:
        nheads_k: Number of key heads
        batch_size: Batch size
        seqlen_q: Query sequence length per batch item
        seqlen_k: Key sequence length
        num_topk: Number of top-k blocks to keep per query position, or list of num_topk values per batch
        block_size: Size of each block
        device: Device to create the indices on
        mode: Sparsity pattern mode:
            - "random": Random pattern based on number of blocks to keep (default)
            - "half_heads_dense": First half of heads are fully dense, second half follow pattern
            - "half_seq_dense": First half of sequence positions are fully dense, second half follow pattern
        
    Returns:
        torch.Tensor: Topk indices with shape [batch_size, nheads_k, seqlen_q, k]
    """
    # Calculate number of blocks in key dimension
    k_blocks = (seqlen_k + block_size - 1) // block_size
    
    # Handle both scalar and per-batch num_topk
    if isinstance(num_topk, (int)):
        num_topk_per_batch = [num_topk] * batch_size
    else:
        assert len(num_topk) == batch_size, "If providing per-batch num_topk, must match batch_size"
        num_topk_per_batch = num_topk
    
    # Create indices for each batch separately
    batch_indices = []
    
    for b in range(batch_size):
        # Get number of blocks to keep (top-k) for this batch
        k = min(max(1, num_topk_per_batch[b]), k_blocks)
        
        # Initialize batch_idx
        if mode == "random":
            # Generate indices for this batch with random pattern
            batch_idx = torch.stack([
                torch.stack([
                    torch.randperm(k_blocks, device=device)[:k]
                    for _ in range(seqlen_q)
                ])
                for _ in range(nheads_k)
            ])
        
        elif mode == "half_heads_dense":
            # Calculate the halfway point for heads
            half_heads = nheads_k // 2
            
            # Initialize indices for all heads and sequence positions
            batch_idx = torch.zeros((nheads_k, seqlen_q, k), dtype=torch.int32, device=device)
            
            # First half of heads are fully dense (select blocks 0 to k-1)
            for h in range(half_heads, nheads_k):
                for i in range(seqlen_q):
                    batch_idx[h, i] = torch.arange(k, device=device)
            
            # Second half of heads use random pattern
            for h in range(half_heads):
                for i in range(seqlen_q):
                    batch_idx[h, i] = torch.randperm(k_blocks, device=device)[:k]
                    
        elif mode == "half_seq_dense":
            # Calculate the halfway point for sequence positions
            half_seq = seqlen_q // 2
            
            # Initialize indices for all heads and sequence positions
            batch_idx = torch.zeros((nheads_k, seqlen_q, k), dtype=torch.int32, device=device)
            
            # For all heads
            for h in range(nheads_k):
                # First half of sequence positions are fully dense (select blocks 0 to k-1)
                for i in range(half_seq):
                    batch_idx[h, i] = torch.arange(k, device=device)
                
                # Second half of sequence positions use random pattern
                for i in range(half_seq, seqlen_q):
                    batch_idx[h, i] = torch.randperm(k_blocks, device=device)[:k]
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Valid options are 'random', 'half_heads_dense', 'half_seq_dense'")
        
        # Sort indices for better performance
        batch_idx = batch_idx.sort(dim=-1)[0].to(torch.int32)
        batch_indices.append(batch_idx)
    
    # Stack along batch dimension
    indices = torch.stack(batch_indices)
    
    return indices

def convert_topk_to_base_blockmask(
    topk_idx: torch.Tensor,
    max_seqlen_k: int,
    block_size: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Convert topk indices to block sparse attention mask
    
    Args:
        topk_idx: Tensor of shape [num_heads, total_seqlen, k] containing block indices
        max_seqlen_k: Maximum sequence length for keys
        block_size: Size of each block
        device: Output device
    
    Returns:
        mask: Boolean mask of shape [num_heads, total_seqlen, k_blocks]
    """
    # Calculate number of key blocks
    k_blocks = (max_seqlen_k + block_size - 1) // block_size
    num_heads, total_seqlen, k = topk_idx.shape

    # Initialize all-False mask
    mask = torch.zeros(num_heads, total_seqlen, k_blocks, 
                       dtype=torch.bool, device=device)

    # Filter out any -1 values (if present)
    valid_mask = topk_idx != -1
    
    # Generate index mask - get head, seq positions and corresponding indices
    batch_idx, seq_idx, k_idx = torch.where(valid_mask)
    block_idx = topk_idx[valid_mask]
    
    # Set corresponding positions to True
    mask[batch_idx, seq_idx, block_idx] = True

    return mask

def convert_batch_topk_to_base_blockmask(
    topk_idx: torch.Tensor,
    max_seqlen_k: int,
    block_size: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Convert batch topk indices to block sparse attention mask with batch dimension
    
    Args:
        topk_idx: Tensor of shape [batch_size, num_heads, seqlen_q, k] containing block indices
        max_seqlen_k: Maximum sequence length for keys
        block_size: Size of each block
        device: Output device
    
    Returns:
        mask: Boolean mask of shape [batch_size, num_heads, seqlen_q, k_blocks]
    """
    # Calculate number of key blocks
    k_blocks = (max_seqlen_k + block_size - 1) // block_size
    batch_size, num_heads, seqlen_q, k = topk_idx.shape

    # Initialize all-False mask
    mask = torch.zeros(batch_size, num_heads, seqlen_q, k_blocks, 
                       dtype=torch.bool, device=device)

    # Filter out any -1 values (if present)
    valid_mask = topk_idx != -1
    
    # Generate index mask - get batch, head, seq positions and corresponding indices
    b_idx, h_idx, q_idx, k_idx = torch.where(valid_mask)
    block_idx = topk_idx[valid_mask]
    
    # Set corresponding positions to True
    mask[b_idx, h_idx, q_idx, block_idx] = True

    return mask

def visualize_gradient_differences(dq, dq_ref, dk, dk_ref, dv, dv_ref, config_name):
    """Create heatmaps of gradient differences and save to files."""
    
    # Create directory for this test case
    test_dir = VISUALIZATION_DIR / config_name
    test_dir.mkdir(exist_ok=True)
    
    # Function to plot and save heatmap
    def plot_gradient_heatmap(grad_diff, name, max_val=None):
        # Convert to numpy and take absolute values
        if grad_diff.dim() > 2:
            grad_diff_reshaped = grad_diff.abs().reshape(-1, grad_diff.shape[-1])
        else:
            grad_diff_reshaped = grad_diff.abs()
            
        grad_np = grad_diff_reshaped.detach().float().cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        if max_val:
            im = plt.imshow(grad_np, cmap='hot', aspect='auto', vmax=max_val)
        else:
            im = plt.imshow(grad_np, cmap='hot', aspect='auto')
        plt.colorbar(im, label='Absolute Difference')
        plt.title(f'Gradient Difference Heatmap - {name}')
        plt.xlabel('Hidden Dimension')
        plt.ylabel('Sequence × Batch × Heads')
        
        # Save figure
        plt.savefig(test_dir / f"{name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save max values info
        with open(test_dir / f"{name}_stats.txt", "w") as f:
            f.write(f"Max diff: {grad_diff.abs().max().item()}\n")
            f.write(f"Mean diff: {grad_diff.abs().mean().item()}\n")
            f.write(f"Shape: {tuple(grad_diff.shape)}\n")
            
            # Report locations of largest differences
            flat_indices = grad_diff.abs().flatten().argsort(descending=True)[:10]
            multidim_indices = [np.unravel_index(idx.item(), grad_diff.shape) for idx in flat_indices]
            f.write("\nTop 10 largest differences locations:\n")
            for i, idx in enumerate(multidim_indices):
                f.write(f"{i+1}. Position {idx}: {grad_diff[idx].item()}\n")
    
    # Function to plot detailed analysis for gradients
    def plot_detailed_analysis(grad, grad_ref, name):
        diff = grad - grad_ref
        abs_diff = diff.abs()
        
        # Create a directory for detailed analysis
        detail_dir = test_dir / f"{name}_detailed"
        detail_dir.mkdir(exist_ok=True)
        
        # 1. Create histograms of differences
        plt.figure(figsize=(10, 6))
        diff_np = diff.flatten().detach().float().cpu().numpy()
        plt.hist(diff_np, bins=100, alpha=0.7)
        plt.title(f'{name} Difference Distribution')
        plt.xlabel('Difference Value')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.savefig(detail_dir / "difference_histogram.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Analysis by head
        if grad.dim() >= 3:  # Check if we have head dimension
            num_heads = grad.shape[2] if grad.dim() >= 4 else grad.shape[1]
            head_errors = []
            
            for h in range(num_heads):
                if grad.dim() >= 4:
                    head_diff = abs_diff[:, :, h, :].mean().item()
                else:
                    head_diff = abs_diff[:, h, :].mean().item()
                head_errors.append(head_diff)
            
            # Plot head analysis
            plt.figure(figsize=(12, 6))
            plt.bar(range(num_heads), head_errors)
            plt.title(f'{name} Error by Attention Head')
            plt.xlabel('Head Index')
            plt.ylabel('Mean Absolute Error')
            plt.savefig(detail_dir / "head_error.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save head error data
            with open(detail_dir / "head_error.txt", "w") as f:
                for h, err in enumerate(head_errors):
                    f.write(f"Head {h}: {err}\n")
                f.write(f"\nWorst head: {np.argmax(head_errors)} (error: {max(head_errors)})\n")
        
        # 3. Analysis by dimension - Convert to float32 before numpy()
        dim_errors = abs_diff.mean(dim=tuple(range(abs_diff.dim() - 1))).detach().float().cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(dim_errors)), dim_errors)
        plt.title(f'{name} Error by Hidden Dimension')
        plt.xlabel('Dimension Index')
        plt.ylabel('Mean Absolute Error')
        plt.savefig(detail_dir / "dimension_error.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save top dimension errors
        with open(detail_dir / "dimension_error.txt", "w") as f:
            top_dims = np.argsort(dim_errors)[::-1][:10]
            f.write("Top 10 dimensions with highest error:\n")
            for i, dim in enumerate(top_dims):
                f.write(f"{i+1}. Dimension {dim}: {dim_errors[dim]}\n")
        
        # 4. Create error heatmap across sequence positions (useful for causal attention)
        if grad.dim() >= 3:
            # Average across batch and heads to get seq_len x dim error map
            if grad.dim() == 4:  # [batch, seq, head, dim]
                seq_errors = abs_diff.mean(dim=(0, 2)).detach().float().cpu().numpy()
            else:  # [batch, head, seq, dim] or similar
                seq_errors = abs_diff.mean(dim=(0, 1)).detach().float().cpu().numpy()
                
            plt.figure(figsize=(10, 8))
            plt.imshow(seq_errors, aspect='auto', cmap='hot')
            plt.colorbar(label='Mean Absolute Error')
            plt.title(f'{name} Error by Sequence Position')
            plt.xlabel('Hidden Dimension')
            plt.ylabel('Sequence Position')
            plt.savefig(detail_dir / "sequence_error.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    # Add this function to analyze token-specific errors
    def analyze_token_errors(grad, grad_ref, name):
        diff = grad - grad_ref
        abs_diff = diff.abs()
        
        # Calculate error by token position (averaging across batch, heads, and hidden dims)
        # Shape: [seq_len]
        if grad.dim() == 4:  # [batch, seq, head, dim]
            token_errors = abs_diff.mean(dim=(0, 2, 3)).detach().float().cpu().numpy()
        else:
            token_errors = abs_diff.mean(dim=(0, 2)).detach().float().cpu().numpy()
            
        # Find top error positions
        top_error_tokens = np.argsort(token_errors)[::-1][:20]  # Top 20 error positions
        
        # Calculate moving average to identify how many tokens from start have errors
        window_size = 5
        moving_avg = np.convolve(token_errors, np.ones(window_size)/window_size, mode='valid')
        
        # Find where error drops significantly (threshold at 10% of max error)
        error_threshold = token_errors.max() * 0.1
        below_threshold = np.where(moving_avg < error_threshold)[0]
        error_span = below_threshold[0] if len(below_threshold) > 0 else len(token_errors)
        
        # Log findings
        print(f"\n===== {name} TOKEN ERROR ANALYSIS =====")
        print(f"First {error_span} tokens have significant errors")
        print(f"Top error positions:")
        for i, pos in enumerate(top_error_tokens[:10]):
            print(f"  {i+1}. Token position {pos}: error = {token_errors[pos]:.6f}")
            
        # Save detailed token error plot
        plt.figure(figsize=(12, 6))
        plt.plot(token_errors, 'b-', label='Error by token position')
        plt.axhline(y=error_threshold, color='r', linestyle='--', label='Error threshold')
        for pos in top_error_tokens[:5]:
            plt.plot(pos, token_errors[pos], 'ro')
            plt.text(pos, token_errors[pos], f"{pos}", verticalalignment='bottom')
        plt.title(f'{name} Error by Token Position')
        plt.xlabel('Token Position')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.savefig(test_dir / f"{name}_token_errors.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save token error data
        with open(test_dir / f"{name}_token_errors.txt", "w") as f:
            f.write(f"First {error_span} tokens have significant errors\n\n")
            f.write("Token position, Error magnitude\n")
            for i, err in enumerate(token_errors):
                f.write(f"{i}, {err}\n")
                
        return error_span, token_errors
    
    # Get maximum difference value across all gradients for consistent scaling
    max_diff = max(
        (dq - dq_ref).abs().max().item(),
        (dk - dk_ref).abs().max().item(),
        (dv - dv_ref).abs().max().item()
    )
    
    # Plot each gradient difference
    plot_gradient_heatmap(dq - dq_ref, "dQ_diff", max_diff)
    plot_gradient_heatmap(dk - dk_ref, "dK_diff", max_diff)
    plot_gradient_heatmap(dv - dv_ref, "dV_diff", max_diff)
    
    # Also plot raw gradients to compare patterns
    plot_gradient_heatmap(dq, "dQ_impl")
    plot_gradient_heatmap(dq_ref, "dQ_ref")
    plot_gradient_heatmap(dk, "dK_impl")
    plot_gradient_heatmap(dk_ref, "dK_ref")
    plot_gradient_heatmap(dv, "dV_impl")
    plot_gradient_heatmap(dv_ref, "dV_ref")
    
    # Add detailed analysis for dk and dv
    logger.info("Generating detailed analysis for dQ gradients")
    plot_detailed_analysis(dq, dq_ref, "dQ")
    
    logger.info("Generating detailed analysis for dK gradients")
    plot_detailed_analysis(dk, dk_ref, "dK")
    
    logger.info("Generating detailed analysis for dV gradients")
    plot_detailed_analysis(dv, dv_ref, "dV")
    
    # Analyze correlation between dK and dV differences
    dk_diff = (dk - dk_ref).abs().flatten().detach().float().cpu().numpy()
    dv_diff = (dv - dv_ref).abs().flatten().detach().float().cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.hexbin(dk_diff, dv_diff, gridsize=50, cmap='inferno', bins='log')
    plt.colorbar(label='log10(N)')
    plt.xlabel('dK Absolute Difference')
    plt.ylabel('dV Absolute Difference')
    plt.title('Correlation between dK and dV Differences')
    plt.savefig(test_dir / "dk_dv_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Call the new analysis function for each gradient
    q_error_span, q_token_errors = analyze_token_errors(dq, dq_ref, "dQ")
    k_error_span, k_token_errors = analyze_token_errors(dk, dk_ref, "dK")
    v_error_span, v_token_errors = analyze_token_errors(dv, dv_ref, "dV")
    
    # Compare the gradients
    print("\n===== GRADIENT ERROR SUMMARY =====")
    print(f"dQ: First {q_error_span} tokens have significant errors")
    print(f"dK: First {k_error_span} tokens have significant errors")
    print(f"dV: First {v_error_span} tokens have significant errors")
    
    # Add this to compare token error patterns
    plt.figure(figsize=(12, 6))
    plt.plot(q_token_errors, 'r-', label='dQ errors')
    plt.plot(k_token_errors, 'g-', label='dK errors')
    plt.plot(v_token_errors, 'b-', label='dV errors')
    plt.title('Gradient Errors by Token Position')
    plt.xlabel('Token Position')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.savefig(test_dir / "gradient_token_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

def visualize_output_differences(out, out_ref, out_pt, config_name):
    """Create heatmaps of output differences and save to files."""
    
    # Create directory for this test case
    test_dir = VISUALIZATION_DIR / config_name
    test_dir.mkdir(exist_ok=True)
    
    # Function to plot and save heatmap
    def plot_output_heatmap(output_diff, name, max_val=None):
        # Convert to numpy and take absolute values
        if output_diff.dim() > 2:
            output_diff_reshaped = output_diff.abs().reshape(-1, output_diff.shape[-1])
        else:
            output_diff_reshaped = output_diff.abs()
            
        output_np = output_diff_reshaped.detach().float().cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        if max_val:
            im = plt.imshow(output_np, cmap='hot', aspect='auto', vmax=max_val)
        else:
            im = plt.imshow(output_np, cmap='hot', aspect='auto')
        plt.colorbar(im, label='Absolute Difference')
        plt.title(f'Output Difference Heatmap - {name}')
        plt.xlabel('Hidden Dimension')
        plt.ylabel('Sequence × Batch × Heads')
        
        # Save figure
        plt.savefig(test_dir / f"{name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save max values info
        with open(test_dir / f"{name}_stats.txt", "w") as f:
            f.write(f"Max diff: {output_diff.abs().max().item()}\n")
            f.write(f"Mean diff: {output_diff.abs().mean().item()}\n")
            f.write(f"Shape: {tuple(output_diff.shape)}\n")
            
            # Report locations of largest differences
            flat_indices = output_diff.abs().flatten().argsort(descending=True)[:10]
            multidim_indices = [np.unravel_index(idx.item(), output_diff.shape) for idx in flat_indices]
            f.write("\nTop 10 largest differences locations:\n")
            for i, idx in enumerate(multidim_indices):
                f.write(f"{i+1}. Position {idx}: {output_diff[idx].item()}\n")
    
    # Function to analyze token-specific output differences
    def analyze_token_errors(out, out_ref, name):
        diff = out - out_ref
        abs_diff = diff.abs()
        
        # Calculate error by token position (averaging across batch, heads, and hidden dims)
        # Shape: [seq_len]
        if out.dim() == 4:  # [batch, seq, head, dim]
            token_errors = abs_diff.mean(dim=(0, 2, 3)).detach().float().cpu().numpy()
        else:
            token_errors = abs_diff.mean(dim=(0, 2)).detach().float().cpu().numpy()
            
        # Find top error positions
        top_error_tokens = np.argsort(token_errors)[::-1][:20]  # Top 20 error positions
        
        # Calculate moving average to identify how many tokens from start have errors
        window_size = 5
        moving_avg = np.convolve(token_errors, np.ones(window_size)/window_size, mode='valid')
        
        # Find where error drops significantly (threshold at 10% of max error)
        error_threshold = token_errors.max() * 0.1
        below_threshold = np.where(moving_avg < error_threshold)[0]
        error_span = below_threshold[0] if len(below_threshold) > 0 else len(token_errors)
        
        # Log findings
        print(f"\n===== {name} TOKEN ERROR ANALYSIS =====")
        print(f"First {error_span} tokens have significant errors")
        print(f"Top error positions:")
        for i, pos in enumerate(top_error_tokens[:10]):
            print(f"  {i+1}. Token position {pos}: error = {token_errors[pos]:.6f}")
            
        # Save detailed token error plot
        plt.figure(figsize=(12, 6))
        plt.plot(token_errors, 'b-', label='Error by token position')
        plt.axhline(y=error_threshold, color='r', linestyle='--', label='Error threshold')
        for pos in top_error_tokens[:5]:
            plt.plot(pos, token_errors[pos], 'ro')
            plt.text(pos, token_errors[pos], f"{pos}", verticalalignment='bottom')
        plt.title(f'{name} Error by Token Position')
        plt.xlabel('Token Position')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.savefig(test_dir / f"{name}_token_errors.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save token error data
        with open(test_dir / f"{name}_token_errors.txt", "w") as f:
            f.write(f"First {error_span} tokens have significant errors\n\n")
            f.write("Token position, Error magnitude\n")
            for i, err in enumerate(token_errors):
                f.write(f"{i}, {err}\n")
                
        return error_span, token_errors
    
    # Function to analyze the output values directly
    def analyze_output_distribution(output, output_ref, name):
        detail_dir = test_dir / f"{name}_detailed"
        detail_dir.mkdir(exist_ok=True)
        
        # Get raw values for comparison
        output_flat = output.reshape(-1).detach().float().cpu().numpy()
        output_ref_flat = output_ref.reshape(-1).detach().float().cpu().numpy()
        
        # Plot histograms of values
        plt.figure(figsize=(12, 6))
        plt.hist(output_flat, bins=100, alpha=0.5, label='Implementation')
        plt.hist(output_ref_flat, bins=100, alpha=0.5, label='Reference')
        plt.title(f'{name} Value Distribution Comparison')
        plt.xlabel('Output Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(detail_dir / "value_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Check for NaN/Inf/Zero values
        impl_nan = np.isnan(output_flat).sum()
        ref_nan = np.isnan(output_ref_flat).sum()
        impl_inf = np.isinf(output_flat).sum()
        ref_inf = np.isinf(output_ref_flat).sum()
        impl_zero = (output_flat == 0).sum()
        ref_zero = (output_ref_flat == 0).sum()
        
        with open(detail_dir / "special_values.txt", "w") as f:
            f.write(f"Implementation NaN count: {impl_nan}\n")
            f.write(f"Reference NaN count: {ref_nan}\n")
            f.write(f"Implementation Inf count: {impl_inf}\n")
            f.write(f"Reference Inf count: {ref_inf}\n")
            f.write(f"Implementation Zero count: {impl_zero}\n")
            f.write(f"Reference Zero count: {ref_zero}\n")
        
        # Check correlations between outputs
        plt.figure(figsize=(8, 8))
        plt.hexbin(output_ref_flat, output_flat, gridsize=50, cmap='inferno', bins='log')
        plt.colorbar(label='log10(N)')
        plt.xlabel('Reference Output Values')
        plt.ylabel('Implementation Output Values')
        plt.title('Correlation between Reference and Implementation')
        plt.plot([-3, 3], [-3, 3], 'r--')  # Ideal correlation line
        plt.savefig(detail_dir / "output_correlation.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        return impl_nan, ref_nan, impl_inf, ref_inf, impl_zero, ref_zero
    
    # Get maximum difference value for consistent scaling
    max_diff = max(
        (out - out_ref).abs().max().item(),
        (out_pt - out_ref).abs().max().item()
    )
    
    # Plot output differences
    plot_output_heatmap(out - out_ref, "Output_impl_diff", max_diff)
    plot_output_heatmap(out_pt - out_ref, "Output_pt_diff", max_diff)
    
    # Also plot raw outputs to compare patterns
    plot_output_heatmap(out, "Output_impl")
    plot_output_heatmap(out_ref, "Output_ref")
    plot_output_heatmap(out_pt, "Output_pt")
    
    # Analyze token-specific errors
    logger.info("Analyzing token-specific output errors")
    o_error_span, o_token_errors = analyze_token_errors(out, out_ref, "Output")
    pt_error_span, pt_token_errors = analyze_token_errors(out_pt, out_ref, "PT_Output")
    
    # Compare the outputs in detail
    logger.info("Analyzing output distributions")
    impl_stats = analyze_output_distribution(out, out_ref, "Output")
    pt_stats = analyze_output_distribution(out_pt, out_ref, "PT_Output")
    
    # Compare token error patterns
    plt.figure(figsize=(12, 6))
    plt.plot(o_token_errors, 'r-', label='Implementation errors')
    plt.plot(pt_token_errors, 'g-', label='PyTorch errors')
    plt.title('Output Errors by Token Position')
    plt.xlabel('Token Position')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.savefig(test_dir / "output_token_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary report
    with open(test_dir / "output_analysis_summary.txt", "w") as f:
        f.write(f"===== OUTPUT ANALYSIS SUMMARY =====\n\n")
        f.write(f"Implementation max diff: {(out - out_ref).abs().max().item()}\n")
        f.write(f"Implementation mean diff: {(out - out_ref).abs().mean().item()}\n")
        f.write(f"PyTorch max diff: {(out_pt - out_ref).abs().max().item()}\n")
        f.write(f"PyTorch mean diff: {(out_pt - out_ref).abs().mean().item()}\n\n")
        
        f.write(f"Implementation first {o_error_span} tokens have significant errors\n")
        f.write(f"PyTorch first {pt_error_span} tokens have significant errors\n\n")
        
        f.write("Special values in Implementation:\n")
        f.write(f"  NaN count: {impl_stats[0]}\n")
        f.write(f"  Inf count: {impl_stats[2]}\n")
        f.write(f"  Zero count: {impl_stats[4]}\n\n")
        
        f.write("Special values in Reference:\n")
        f.write(f"  NaN count: {impl_stats[1]}\n")
        f.write(f"  Inf count: {impl_stats[3]}\n")
        f.write(f"  Zero count: {impl_stats[5]}\n")


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
        # breakpoint()
        # lengths = torch.randint(
        #     max_seqlen - 2, max_seqlen + 1, (batch_size, 1), device=device
        # )
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask

def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )

def construct_exact_streaming_mask(
    seqlen_q,
    seqlen_k,
    sink_size,  # -1 means infinite window size
    local_size,
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    assert sink_size >= 0
    assert local_size >= 1
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )

    sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
    mask = torch.logical_or(
        col_idx > torch.minimum(row_idx + sk - sq, sk),
        torch.logical_and(
        col_idx < row_idx + sk - sq - (local_size-1), col_idx >= sink_size,
        )
    )

    return mask

def generate_qkv(
    q, k, v, query_padding_mask=None, key_padding_mask=None, kvpacked=False, qkvpacked=False
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )

def generate_base_sparsity_mask(max_seqlen_q, max_seqlen_k, round_base, m_block_dim, n_block_dim, 
                                     batch_size, num_blocksparse_heads, pattern_type="random", 
                                     sparsity_list=None, causal=False, device="cuda"):
    """
    Generate structured sparsity masks for block sparse attention.
    
    Parameters:
        pattern_type: String indicating the pattern type:
            - "random": Random sparsity based on sparsity_list (original behavior)
            - "seq_half": First half of sequence positions are True, second half False
            - "head_half": First half of heads are fully dense, second half fully sparse
            - "checkerboard": Alternating True/False in a checkerboard pattern
    """
    def round_to_multiple(x, base):
        return ((x + base - 1) // base) * base
    
    nrow, ncol = round_to_multiple(max_seqlen_q, round_base), round_to_multiple(max_seqlen_k, round_base) // n_block_dim
    base_mask = torch.zeros(batch_size, num_blocksparse_heads, nrow, ncol, device=device, dtype=torch.bool)
    
    if pattern_type == "random" and sparsity_list is not None:
        assert len(sparsity_list) == num_blocksparse_heads
        # Original random pattern logic
        for batch in range(batch_size):
            for head_rank in range(num_blocksparse_heads):
                sparsity = sparsity_list[head_rank]
                if not sparsity == 0.0 and not sparsity == 1.0:
                    for i in range(nrow):
                        idx = nrow - i - 1
                        if causal:
                            available_col_num = max(0, ncol - i)
                            num_one = max(1, int(sparsity * available_col_num))
                            base_mask[batch][head_rank][idx, torch.randperm(available_col_num)[:num_one]] = True
                        else:
                            available_col_num = ncol
                            num_one = max(1, int(sparsity * available_col_num))
                            base_mask[batch][head_rank][idx, torch.randperm(available_col_num)[:num_one]] = True
                elif sparsity == 1.0:
                    base_mask[batch][head_rank] = torch.ones_like(base_mask[batch][head_rank])
    
    elif pattern_type == "seq_half":
        # First half sequence positions are True, second half False
        for batch in range(batch_size):
            for head_rank in range(num_blocksparse_heads):
                half_point = nrow // 2
                if causal:
                    # For causal attention, ensure upper triangular constraint
                    for i in range(half_point):
                        row_idx = nrow - i - 1
                        # Only set columns up to the current row index as True (causal constraint)
                        max_col = min(ncol, nrow - i - 1)
                        base_mask[batch, head_rank, row_idx, :max_col] = True
                else:
                    # Non-causal case - original behavior
                    base_mask[batch, head_rank, :half_point, :] = True
    
    elif pattern_type == "head_half":
        # First half heads are fully dense, second half fully sparse
        half_point = num_blocksparse_heads // 2
        for batch in range(batch_size):
            base_mask[batch, :half_point] = torch.ones_like(base_mask[batch, :half_point])
            # Second half remains all zeros (sparse)
    
    elif pattern_type == "checkerboard":
        # Checkerboard pattern
        for batch in range(batch_size):
            for head_rank in range(num_blocksparse_heads):
                for i in range(nrow):
                    for j in range(ncol):
                        if (i + j) % 2 == 0:  # Alternating pattern
                            base_mask[batch, head_rank, i, j] = True
    
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
                
    return base_mask


def replace_ones_with_count(tensor):
    ones_mask = tensor == 1
    count = torch.cumsum(ones_mask, dim=-1).to(tensor.dtype)
    count = count * ones_mask
    tensor = tensor.masked_scatter(ones_mask, count[ones_mask])
    return tensor


def prepare_mixed_mask(base_blockmask, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k, batch_size, nheads = 32, nheads_k= 2, m_block_dim=128, n_block_dim=128, block_window_size=0):
    """
    Expand a block-level sparsity mask to a token-level mask for attention_blocksparse_ref,
    handling variable sequence lengths per batch item.
    
    Parameters:
        base_blockmask: Bool tensor of shape [nheads_k, total_unpadded_tokens, ncol] 
            where True values indicate blocks that should be attended to
        cu_seqlens_q: Cumulative sequence lengths for queries [batch_size + 1]
        cu_seqlens_k: Cumulative sequence lengths for keys [batch_size + 1]
        seqlen_q: Maximum query sequence length (padded)
        seqlen_k: Maximum key sequence length (padded)
        batch_size: Number of batches
        m_block_dim: Block size for query dimension (default: 128)
        n_block_dim: Block size for key dimension (default: 128)
        block_window_size: Number of blocks to the left of each query block to attend to (default: 0)
    
    Returns:
        mixed_mask: Bool tensor of shape [batch_size, num_heads, seqlen_q, seqlen_k]
            where True values indicate positions that should be masked out
    """
    # Make a copy of base_blockmask to avoid modifying the original
    modified_blockmask = base_blockmask.clone()
    
    # Helper function to round up to multiple
    def round_to_multiple(x, base):
        return ((x + base - 1) // base) * base
    
    # Apply block window logic if block_window_size > 0
    if block_window_size > 0:
        num_blocks_k = modified_blockmask.shape[2]  # number of blocks in key dimension
        
        # Generate block indices for each query position
        total_unpadded_tokens = modified_blockmask.shape[1]
        
        # Calculate batch-relative positions for each token in the unpadded sequence
        q_positions = torch.arange(total_unpadded_tokens, device=modified_blockmask.device)
        
        # Properly determine which batch each position belongs to
        batch_indices = torch.zeros_like(q_positions)
        for b in range(1, len(cu_seqlens_q)):
            batch_indices = torch.where(
                q_positions >= cu_seqlens_q[b-1],
                torch.where(
                    q_positions < cu_seqlens_q[b],
                    torch.tensor(b-1, device=q_positions.device),
                    batch_indices
                ),
                batch_indices
            )
        
        # Calculate relative positions within each batch
        relative_positions = q_positions - cu_seqlens_q[batch_indices]
        
        # For each query position, check if key blocks are within the window range
        # according to kernel logic: k_idx >= q_block_idx - (block_window_size * n_block_dim) && k_idx <= q_block_idx
        for i in range(total_unpadded_tokens):
            q_block_idx = relative_positions[i]
            for k_block in range(num_blocks_k):
                k_idx = k_block * n_block_dim
                # Matching the kernel logic exactly
                if k_idx >= q_block_idx - (block_window_size * n_block_dim) and k_idx <= round_to_multiple(q_block_idx, n_block_dim):
                    # Set this position to True for all heads
                    modified_blockmask[:, i, k_block] = True
    
    # Expand blocks to token level for all heads
    expanded_mask = repeat(modified_blockmask, "h r c -> (h g) r (c n)", 
                         g = int(nheads / nheads_k), n=n_block_dim)
    # breakpoint()
    # Create batch of masks with padding
    batch_masks = []
    for b in range(batch_size):
        # Get actual sequence lengths and start/end indices for this batch item
        q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b+1].item()
        k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b+1].item()
        q_len = q_end - q_start
        k_len = k_end - k_start
        
        # Create padded mask for this batch item
        # Initialize with all masked out (True)
        batch_mask = torch.ones(nheads, seqlen_q, seqlen_k, dtype=torch.bool, 
                               device=base_blockmask.device)
        
        # Copy the relevant portion from expanded mask corresponding to THIS batch item
        # We need to use the correct slice from the unpadded tokens
        # And invert since True in expanded_mask means "attend to"
        # but True in final mask means "mask out" (set to -inf)
        batch_mask[:, :q_len, :k_len] = ~expanded_mask[:, q_start:q_end, :k_len]

        batch_masks.append(batch_mask)
    
    # Stack all batch masks
    mixed_mask = torch.stack(batch_masks, dim=0)
    return mixed_mask

def prepare_batch_mixed_mask(base_blockmask, seqlen_q, seqlen_k, nheads=32, nheads_k=2, n_block_dim=128, block_window_size=0):
    """
    Expand a batched block-level sparsity mask to a token-level mask for attention_blocksparse_ref.
    
    Parameters:
        base_blockmask: Bool tensor of shape [batch_size, nheads_k, seqlen_q, num_blocks] 
            where True values indicate blocks that should be attended to
        seqlen_q: Query sequence length
        seqlen_k: Key sequence length
        nheads: Total number of attention heads
        nheads_k: Number of key heads (usually fewer than query heads)
        n_block_dim: Block size for key dimension (default: 128)
        block_window_size: Number of blocks to attend to (default: 0)
                          In decoding scenario where queries come after keys,
                          this allows controlling how many past key blocks each
                          query position can attend to.
    
    Returns:
        mixed_mask: Bool tensor of shape [batch_size, num_heads, seqlen_q, seqlen_k]
            where True values indicate positions that should be masked out
    """
    batch_size = base_blockmask.shape[0]
    
    # Make a copy of base_blockmask to avoid modifying the original
    modified_blockmask = base_blockmask.clone()
    
    # Apply block window logic if block_window_size > 0
    if block_window_size > 0:
        # Helper function to round up to multiple
        def round_to_multiple(x, base):
            return ((x + base - 1) // base) * base
        
        # Get number of blocks in key dimension
        num_blocks_k = modified_blockmask.shape[3]  # [batch, heads_k, seqlen_q, num_blocks_k]
        
        # For each batch, head, and query position, apply window logic
        for b in range(batch_size):
            for h in range(modified_blockmask.shape[1]):  # Loop through heads_k
                for q_pos in range(modified_blockmask.shape[2]):  # Loop through seqlen_q
                    # Calculate the block index for this query position
                    q_block_idx = q_pos + seqlen_k
                    
                    # Apply window logic to each key block
                    for k_block in range(num_blocks_k):
                        k_idx = k_block * n_block_dim
                        
                        # Match the kernel logic exactly: 
                        # k_idx >= q_block_idx - (block_window_size * n_block_dim) && k_idx <= q_block_idx
                        if k_idx >= q_block_idx - (block_window_size * n_block_dim) and k_idx <= round_to_multiple(q_block_idx, n_block_dim):
                            # Set this position to True (block should be attended to)
                            modified_blockmask[b, h, q_pos, k_block] = True
    
    # First, we need to expand the blocks to token level
    # Repeat each block index into the full block_size in key dimension
    # And repeat for all corresponding query heads per key head
    expanded_mask = repeat(modified_blockmask, "b h_k q c -> b (h_k g) q (c n)", 
                          g=int(nheads/nheads_k), n=n_block_dim)
    
    # Pad to full sequence length if needed
    if expanded_mask.shape[2] < seqlen_q or expanded_mask.shape[3] < seqlen_k:
        padded_mask = torch.zeros(batch_size, nheads, seqlen_q, seqlen_k, 
                                 dtype=torch.bool, device=expanded_mask.device)
        padded_mask[:, :, :expanded_mask.shape[2], :expanded_mask.shape[3]] = expanded_mask
        expanded_mask = padded_mask
    
    # Truncate if needed
    if expanded_mask.shape[2] > seqlen_q or expanded_mask.shape[3] > seqlen_k:
        expanded_mask = expanded_mask[:, :, :seqlen_q, :seqlen_k]
    
    # Invert the mask - in expanded_mask, True means "attend to"
    # but in attention_blocksparse_ref, True means "mask out" (set to -inf)
    return ~expanded_mask

def attention_blocksparse_ref(
    q, k, v, 
    mixed_mask,
    query_padding_mask=None,
    key_padding_mask=None, 
    p_dropout=0.0, 
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),
    upcast=True,
    reorder_ops=False,
    ):
    # q, k, v = qkv.float().unbind(dim=2)
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    # local mask
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
        
    
    scores.masked_fill_(rearrange(mixed_mask, "b h t s -> b h t s"), float("-inf"))
    
    # print("processed blockmask: ", rearrange(~base_blockmask, "h t s -> 1 h t s"))
    
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
     
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(torch.bitwise_or(local_mask, rearrange(mixed_mask, "b h t s -> b h t s")), dim=-1, keepdim=True), 0.0)
    
    attention = attention.masked_fill(rearrange(mixed_mask, "b h t s -> b h t s"), 0.0)  
    
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - p_dropout)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)
   
   
def convert_flash_attn_S_to_softmax(
    S,
    seqlen_q,
    seqlen_k,
    query_padding_mask,
    key_padding_mask,
    head_dim,
    is_dropout,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """FlashAttention stores the S matrix in a different way.
    Arguments:
        S: (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded)
        query_padding_mask: (batch_size, seqlen_q_rounded)
        key_padding_mask: (batch_size, seqlen_k_rounded)
    """
    if causal:
        window_size = (window_size[0], 0)
    seqlen_q_rounded, seqlen_k_rounded = S.shape[-2:]
    warps_n = 4
    blocksize_m, blocksize_n = _get_block_size(S.device, head_dim, is_dropout, causal)
    nblocks_n = (seqlen_k_rounded + blocksize_n - 1) // blocksize_n
    nblocks_m = (seqlen_q_rounded + blocksize_m - 1) // blocksize_m
    mmas_n = (blocksize_n + 16 - 1) // 16
    S_flat = rearrange(
        S,
        "b h (nblocks_m blocksize_m) (nblocks_n blocksize_n) -> b h nblocks_m nblocks_n (blocksize_m blocksize_n)",
        blocksize_m=blocksize_m,
        blocksize_n=blocksize_n,
    )
    S_converted = rearrange(
        S_flat,
        "b h nblocks_m nblocks_n (mmas_n mmas_m warps_n eight four c2 c1 c0) -> b h (nblocks_m mmas_m warps_n c1 eight) (nblocks_n mmas_n c2 four c0)",
        mmas_n=mmas_n,
        warps_n=warps_n,
        eight=8,
        c0=2,
        c1=2,
        c2=2,
        four=4,
    )

    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            S.device,
        )
        local_mask = F.pad(
            local_mask,
            (0, seqlen_k_rounded - seqlen_k, 0, seqlen_q_rounded - seqlen_q),
            value=True,
        )
        S_converted.masked_fill_(local_mask, 0.0)

    # Need to zero out things not in attention_mask in case S was initialized with random values
    # and some of those values aren't overwritten.
    seqlen_q_og = (
        query_padding_mask.shape[-1] if query_padding_mask is not None else seqlen_q_rounded
    )
    if query_padding_mask is not None:
        query_padding_mask = F.pad(query_padding_mask, (0, seqlen_q_rounded - seqlen_q_og))
        S_converted = S_converted.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    seqlen_k_og = key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
    if key_padding_mask is not None:
        key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k_rounded - seqlen_k_og))
        S_converted = S_converted.masked_fill(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
    S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q_rounded))
    S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k_rounded))
    return S_converted[:, :, :seqlen_q, :seqlen_k]

def normalize_flash_attn_S(
    attn_unnorm,
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    is_dropout=False,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k, v: (batch_size, seqlen_k, nheads, head_dim)
        key_padding_mask: (batch_size, seqlen_q)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
    Output:
        softmax_lse: (batch_size, nheads, seqlen_q)
        softmax_max: (batch_size, nheads, seqlen_q)
    """
    if causal:
        window_size = (window_size[0], 0)
    q, k, v = q.float(), k.float(), v.float()
    _, seqlen_q, _, head_dim = q.shape
    seqlen_k = k.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(head_dim), k)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias.to(dtype=scores.dtype)
    _, block_size_n = _get_block_size(scores.device, head_dim, is_dropout, causal)
    scores_block = scores.split(block_size_n, dim=-1)
    lse_block = torch.stack([torch.logsumexp(s, dim=-1) for s in scores_block], dim=-1)
    lse = torch.logsumexp(lse_block, dim=-1)
    # lse could be -inf (i.e. all values in scores are -inf), and we want to set those to inf
    # so that when we do torch.exp(m - lse), we get 0.0 instead of NaN.
    lse[lse == float("-inf")] = float("inf")
    scores_max_block = torch.stack([torch.amax(s, dim=-1) for s in scores_block], dim=-1)
    cummax_block = torch.cummax(scores_max_block.flip(-1), dim=-1).values.flip(-1).unbind(dim=-1)
    attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
    attn_norm = torch.cat(
        [
            a * rearrange(torch.exp(m - lse), "b h s -> b h s 1")
            for a, m in zip(attn_unnorm_block, cummax_block)
        ],
        dim=-1,
    )
    if query_padding_mask is not None:
        attn_norm.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    return attn_norm.to(dtype=attn_unnorm.dtype)

def get_dropout_fraction(
    dropout_mask,
    mixed_mask,
    m_block_dim, n_block_dim,
    query_padding_mask=None,
    key_padding_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k), bool. True means keep, False means drop.
    query_padding_mask: (batch_size, seqlen_q)
    key_padding_mask: (batch_size, seqlen_k)
    """
    if causal:
        window_size = (window_size[0], 0)
    batch_size, nheads, seqlen_q, seqlen_k = dropout_mask.shape
    dropped = ~dropout_mask
    valid = torch.ones_like(dropout_mask)
    if mixed_mask is not None:
        # mixed_mask = repeat(mixed_mask, "b h s_m s_n -> b h (s_m d_m) (s_n d_n)", d_m=m_block_dim, d_n=n_block_dim)
        # mixed_mask = tailor_mixedmask_for_test(mixed_mask, seqlen_q, seqlen_k)
        dropped.masked_fill_(rearrange(mixed_mask, "b h t s -> b h t s"), False)
        valid.masked_fill_(rearrange(mixed_mask, "b h t s -> b h t s"), False)
    
    if query_padding_mask is not None:
        dropped.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
        valid.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
    if key_padding_mask is not None:
        dropped.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
        valid.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            dropout_mask.device,
        )
        dropped.masked_fill_(local_mask, False)
        valid.masked_fill_(local_mask, False)
    dropped_total = dropped.sum()
    return dropped.sum() / valid.sum()

def modified_check(a, a_ref, a_pt, rel_paremeter):
    assert a.shape == a_ref.shape
    assert a.shape == a_pt.shape
    left = (a - a_ref).abs().max().item()
    right = (a_pt - a_ref).abs().max().item()
    rtol = 1e-3
    if not right == 0:
        assert left < rel_paremeter * right or left < rtol * a_ref.abs().max().item()
    else:
        assert round(left, 4) == 0 or left < rtol * a_ref.abs().max().item()

def tailor_mixedmask_for_test(spanded_base_mixedmask, seqlen_q, seqlen_k):
    batch_size = spanded_base_mixedmask.shape[0]
    nheads = spanded_base_mixedmask.shape[1]
    spanded_base_mixedmask = spanded_base_mixedmask[:, :, :seqlen_q, :seqlen_k]
    pad_blockmask = torch.zeros(batch_size, nheads, seqlen_q, seqlen_k, dtype=torch.bool, device = spanded_base_mixedmask.device)
    pad_blockmask[:, :, :spanded_base_mixedmask.shape[2], :spanded_base_mixedmask.shape[3]] = spanded_base_mixedmask
    spanded_base_mixedmask = pad_blockmask
    spanded_base_mixedmask = spanded_base_mixedmask.contiguous()
    return spanded_base_mixedmask

def visualize_blockmask_accuracy(out, out_ref, mixed_mask, config_name):
    """
    Visualize accuracy differences between regions where blockmask is True vs False.
    
    Args:
        out: Output tensor from implementation being tested
        out_ref: Reference output tensor
        mixed_mask: Block sparse attention mask where True means masked out (don't attend)
        config_name: Name for the configuration, used for saving files
    """
    # Create directory for this test case
    test_dir = VISUALIZATION_DIR / config_name
    test_dir.mkdir(exist_ok=True)
    
    # Calculate absolute errors
    abs_diff = (out - out_ref).abs()
    
    # Map attention mask to query positions in output
    # For each query position, we need to know which keys it attended to vs didn't attend to
    batch_size, nheads, seqlen_q, seqlen_k = mixed_mask.shape
    
    # Compute statistics about the blockmask
    mask_density = mixed_mask.float().mean().item()
    logger.info(f"Block mask density: {mask_density:.4f} (higher means more positions are masked out)")
    
    # Create visualization of the blockmask
    plt.figure(figsize=(12, 10))
    for b in range(min(batch_size, 2)):  # Show first 2 batches max
        for h in range(min(4, nheads)):  # Show first 4 heads max
            plt.subplot(min(batch_size, 2), min(4, nheads), b * min(4, nheads) + h + 1)
            plt.imshow(mixed_mask[b, h].float().cpu().numpy(), cmap='gray_r', aspect='auto')
            plt.title(f'Batch {b}, Head {h}')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
    plt.suptitle('Block Mask Visualization (black=masked out, white=attended to)')
    plt.tight_layout()
    plt.savefig(test_dir / "blockmask_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Analyze output errors per query position
    # Re-use the token error analysis approach but separate by mask patterns
    token_errors = abs_diff.mean(dim=(0, 2, 3)).detach().float().cpu().numpy()
    
    # Create a version of the mask that's averaged across all heads for simplified analysis
    avg_mask = mixed_mask.float().mean(dim=1).cpu().numpy()  # [batch, seq_q, seq_k]
    
    # For each query position, calculate how much of the key sequence was masked out
    mask_ratio_per_query = avg_mask.mean(axis=2)  # [batch, seq_q]
    
    # Average across batches
    avg_mask_ratio = mask_ratio_per_query.mean(axis=0)  # [seq_q]
    
    # Now we can plot token errors compared to mask ratio
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(token_errors, 'b-', label='Error')
    plt.title('Error by Token Position')
    plt.xlabel('Query Position')
    plt.ylabel('Mean Absolute Error')
    
    plt.subplot(1, 2, 2)
    plt.plot(avg_mask_ratio, 'r-', label='Masked Ratio')
    plt.title('Mask Ratio by Token Position')
    plt.xlabel('Query Position')
    plt.ylabel('Proportion of Keys Masked Out')
    
    plt.tight_layout()
    plt.savefig(test_dir / "error_vs_mask_ratio.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create a scatter plot to see correlation between mask ratio and error
    plt.figure(figsize=(10, 8))
    plt.scatter(avg_mask_ratio, token_errors, alpha=0.7)
    plt.xlabel('Proportion of Keys Masked Out')
    plt.ylabel('Mean Absolute Error')
    plt.title('Correlation: Mask Ratio vs Error')
    
    # Add trend line
    if len(avg_mask_ratio) > 1:
        z = np.polyfit(avg_mask_ratio, token_errors, 1)
        p = np.poly1d(z)
        plt.plot(avg_mask_ratio, p(avg_mask_ratio), "r--", alpha=0.8)
        
        # Calculate correlation
        correlation = np.corrcoef(avg_mask_ratio, token_errors)[0, 1]
        plt.annotate(f"Correlation: {correlation:.4f}", xy=(0.05, 0.95), xycoords='axes fraction')
    
    plt.savefig(test_dir / "mask_ratio_error_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Group query positions by mask ratio and analyze errors
    # Create 5 buckets from low mask ratio to high mask ratio
    num_buckets = 5
    bucket_edges = np.linspace(0, 1, num_buckets + 1)
    bucket_labels = [f"{bucket_edges[i]:.2f}-{bucket_edges[i+1]:.2f}" for i in range(num_buckets)]
    
    bucketed_errors = []
    positions_in_bucket = []
    
    for i in range(num_buckets):
        lower, upper = bucket_edges[i], bucket_edges[i+1]
        mask = (avg_mask_ratio >= lower) & (avg_mask_ratio < upper)
        
        if mask.sum() > 0:
            bucket_errors = token_errors[mask]
            bucketed_errors.append(bucket_errors.mean())
            positions_in_bucket.append(mask.sum())
        else:
            bucketed_errors.append(0)
            positions_in_bucket.append(0)
    
    # Create a bar chart showing average error by mask ratio bucket
    plt.figure(figsize=(12, 6))
    plt.bar(bucket_labels, bucketed_errors, alpha=0.7)
    plt.title('Average Error by Mask Ratio Bucket')
    plt.xlabel('Mask Ratio Range')
    plt.ylabel('Mean Absolute Error')
    
    # Add count annotations
    for i, count in enumerate(positions_in_bucket):
        plt.annotate(f"n={count}", 
                    xy=(i, bucketed_errors[i]), 
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center')
    
    plt.savefig(test_dir / "error_by_mask_bucket.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # ADDED: Directly analyze errors where blockmask is True vs False
    # We need to analyze this at the most detailed level (batch, head, query, key)
    # First reshape the output tensors to match the dimensions of the mask
    # For simplicity, we'll just use the first few batches and heads for this analysis
    
    # Take a subset for the detailed analysis
    max_batches = min(batch_size, 2)
    max_heads = min(nheads, 4)
    
    # This analysis will be a bit more involved as we need to correlate error with mask status
    masked_errors = []
    unmasked_errors = []
    
    for b in range(max_batches):
        for h in range(max_heads):
            # Get the mask for this batch and head
            mask_bh = mixed_mask[b, h]
            
            # Calculate position-wise errors
            # Since out is typically [batch, seq, head, dim], we need to reshape
            if out.dim() == 4:  # [batch, seq, head, dim]
                error_bh = abs_diff[b, :, h].mean(dim=-1)  # Average over hidden dim
            else:
                # Handle other shapes if needed
                error_bh = abs_diff[b].mean(dim=-1)  # Simplification - modify as needed
            
            # Now split errors by mask status
            # For each query position, separate errors from masked vs unmasked key positions
            for q in range(seqlen_q):
                # Get the mask and error for this query position
                mask_q = mask_bh[q]  # True = masked out
                
                # We need to map the keys to their contribution in output errors
                # This is an approximation since the output error is already aggregated
                # For a more accurate analysis, we would need the pre-aggregation errors
                
                # For simplicity, use the error at this query position
                error_q = error_bh[q].item()
                
                # Count how many positions are masked vs unmasked for this query
                masked_count = mask_q.sum().item()
                unmasked_count = (mask_q.numel() - masked_count)
                
                # If we have both masked and unmasked positions
                if masked_count > 0 and unmasked_count > 0:
                    # Add to our collections - repeat the error for statistics
                    masked_errors.extend([error_q] * masked_count)
                    unmasked_errors.extend([error_q] * unmasked_count)
    
    # Convert to numpy arrays for analysis
    masked_errors = np.array(masked_errors)
    unmasked_errors = np.array(unmasked_errors)
    
    # Calculate statistics
    masked_mean = masked_errors.mean() if len(masked_errors) > 0 else 0
    unmasked_mean = unmasked_errors.mean() if len(unmasked_errors) > 0 else 0
    masked_std = masked_errors.std() if len(masked_errors) > 0 else 0
    unmasked_std = unmasked_errors.std() if len(unmasked_errors) > 0 else 0
    
    # Create a direct comparison visualization
    plt.figure(figsize=(10, 6))
    
    # Bar chart comparing errors
    plt.bar(['Masked (True)', 'Unmasked (False)'], 
            [masked_mean, unmasked_mean],
            yerr=[masked_std, unmasked_std],
            alpha=0.7)
    
    plt.title('Error Comparison: Masked vs Unmasked Positions')
    plt.ylabel('Mean Absolute Error')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add count annotations
    plt.annotate(f"n={len(masked_errors)}", 
                xy=(0, masked_mean), 
                xytext=(0, 10),
                textcoords="offset points",
                ha='center')
    plt.annotate(f"n={len(unmasked_errors)}", 
                xy=(1, unmasked_mean), 
                xytext=(0, 10),
                textcoords="offset points",
                ha='center')
    
    plt.savefig(test_dir / "masked_vs_unmasked_errors.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # More detailed visualization: histogram of errors
    plt.figure(figsize=(12, 6))
    
    # Only create histograms if we have data
    if len(masked_errors) > 0 and len(unmasked_errors) > 0:
        # Find common range for both histograms
        all_errors = np.concatenate([masked_errors, unmasked_errors])
        min_error, max_error = all_errors.min(), all_errors.max()
        bins = np.linspace(min_error, max_error, 30)
        
        plt.hist(masked_errors, bins=bins, alpha=0.6, label='Masked (True)', density=True)
        plt.hist(unmasked_errors, bins=bins, alpha=0.6, label='Unmasked (False)', density=True)
        plt.legend()
        plt.title('Distribution of Errors: Masked vs Unmasked Positions')
        plt.xlabel('Error Magnitude')
        plt.ylabel('Density')
    else:
        plt.text(0.5, 0.5, 'Insufficient data for histogram', 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.savefig(test_dir / "masked_vs_unmasked_error_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Compute more targeted error analysis specifically for masked vs unmasked positions
    # Here we'll compute token-wise errors directly based on mask status
    true_mask_errors = []
    false_mask_errors = []
    
    # Reshape tensors to get element-wise errors
    # This is a more direct way to compare errors at masked vs. unmasked positions
    for b in range(batch_size):
        # Extract per-element errors and masks
        if out.dim() == 4:  # [batch, seq, head, dim]
            # Average over heads and hidden dimensions
            error_b = abs_diff[b].mean(dim=(1, 2))  # [seq_q, dim]
        else:
            # Handle other shapes as needed
            error_b = abs_diff[b].mean(dim=1)  # Simplified
        
        # Average mask across heads
        mask_b = mixed_mask[b].float().mean(dim=0) > 0.5  # [seq_q, seq_k]
        
        # For each query position
        for q in range(seqlen_q):
            # Get errors and mask for this query position
            # Here we're approximating since we don't have direct access to pre-aggregation errors
            error_q = error_b[q].item()
            mask_q = mask_b[q]  # [seq_k]
            
            # Count fully masked and unmasked positions
            if mask_q.all():
                true_mask_errors.append(error_q)
            elif ~mask_q.any():
                false_mask_errors.append(error_q)
            # For mixed positions, we could split by ratio but we'll focus on pure cases
    
    # Create visualization for pure masked vs. unmasked query positions
    if len(true_mask_errors) > 0 or len(false_mask_errors) > 0:
        plt.figure(figsize=(10, 6))
        
        # Compute stats
        true_mean = np.mean(true_mask_errors) if len(true_mask_errors) > 0 else 0
        false_mean = np.mean(false_mask_errors) if len(false_mask_errors) > 0 else 0
        true_std = np.std(true_mask_errors) if len(true_mask_errors) > 0 else 0
        false_std = np.std(false_mask_errors) if len(false_mask_errors) > 0 else 0
        
        # Bar chart comparing errors for pure cases
        plt.bar(['Fully Masked Queries', 'Fully Unmasked Queries'], 
                [true_mean, false_mean],
                yerr=[true_std, false_std],
                alpha=0.7)
        
        plt.title('Error Comparison: Fully Masked vs Fully Unmasked Queries')
        plt.ylabel('Mean Absolute Error')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add count annotations
        plt.annotate(f"n={len(true_mask_errors)}", 
                    xy=(0, true_mean), 
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center')
        plt.annotate(f"n={len(false_mask_errors)}", 
                    xy=(1, false_mean), 
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center')
        
        plt.savefig(test_dir / "fully_masked_vs_unmasked_queries.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Save detailed stats in a text file
    with open(test_dir / "blockmask_accuracy_stats.txt", "w") as f:
        f.write(f"Block Mask Analysis for {config_name}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Overall mask density: {mask_density:.4f}\n")
        f.write(f"Overall mean absolute error: {abs_diff.abs().mean().item():.6f}\n\n")
        
        f.write("Error by mask ratio bucket:\n")
        for i, label in enumerate(bucket_labels):
            f.write(f"  {label}: {bucketed_errors[i]:.6f} (n={positions_in_bucket[i]})\n")
        
        if len(avg_mask_ratio) > 1:
            f.write(f"\nCorrelation between mask ratio and error: {correlation:.4f}\n")
            
        # Additional analysis for highest error positions
        top_error_indices = np.argsort(token_errors)[::-1][:10]
        f.write("\nTop 10 highest error positions:\n")
        for i, idx in enumerate(top_error_indices):
            f.write(f"  {i+1}. Position {idx}: error={token_errors[idx]:.6f}, mask_ratio={avg_mask_ratio[idx]:.4f}\n")
        
        # Add stats for masked vs unmasked positions
        f.write("\nMasked vs Unmasked Error Analysis:\n")
        f.write(f"  Masked positions (True in mask): mean={masked_mean:.6f}, std={masked_std:.6f}, count={len(masked_errors)}\n")
        f.write(f"  Unmasked positions (False in mask): mean={unmasked_mean:.6f}, std={unmasked_std:.6f}, count={len(unmasked_errors)}\n")
        
        # Add stats for fully masked vs unmasked queries
        if len(true_mask_errors) > 0 or len(false_mask_errors) > 0:
            f.write("\nFully Masked vs Unmasked Query Positions:\n")
            if len(true_mask_errors) > 0:
                f.write(f"  Fully masked queries: mean={np.mean(true_mask_errors):.6f}, std={np.std(true_mask_errors):.6f}, count={len(true_mask_errors)}\n")
            else:
                f.write("  No fully masked queries found\n")
                
            if len(false_mask_errors) > 0:
                f.write(f"  Fully unmasked queries: mean={np.mean(false_mask_errors):.6f}, std={np.std(false_mask_errors):.6f}, count={len(false_mask_errors)}\n")
            else:
                f.write("  No fully unmasked queries found\n")

    logger.info(f"Blockmask accuracy visualization saved to {test_dir}")