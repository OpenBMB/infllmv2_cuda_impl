# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py

import torch
from einops import repeat
from infllm_v2 import (
    block_sparse_attn_func,
)
from utils import (
    generate_random_padding_mask,
    generate_base_sparsity_mask,
    generate_qkv,
    prepare_mixed_mask,
    convert_flash_attn_S_to_softmax,
    normalize_flash_attn_S,
    get_dropout_fraction,
    attention_blocksparse_ref,
    convert_topk_to_base_blockmask,
    generate_topk_indices,
)
import logging
import time
import gc

import numpy as np
import os
from pathlib import Path
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


MAX_HEADDIM_SM8x = 192
block_size = 64
is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)

def test_flash_attn_varlen_block_output(
    seqlen_q, seqlen_k, d, p_dropout, causal, exact_streaming, sink_num, local_num, mha_type, dtype, sparsity, batch_size, nheads, nheads_k, block_window_size=0
):
    logger.info(f"Starting test with parameters: seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, d={d}, p_dropout={p_dropout}, "
                f"causal={causal}, exact_streaming={exact_streaming}, sink_num={sink_num}, local_num={local_num}, "
                f"mha_type={mha_type}, dtype={dtype}, sparsity={sparsity}, batch_size={batch_size}, nheads={nheads}, "
                f"block_window_size={block_window_size}")
    
    # Create a unique config name for this test
    config_name = f"s{seqlen_q}x{seqlen_k}_d{d}_h{nheads}_kv{nheads_k}_drop{p_dropout}_sparsity{sparsity}"
    if causal:
        config_name += "_causal"
    if exact_streaming:
        config_name += "_exact_streaming"
    if block_window_size > 0:
        config_name += f"_window{block_window_size}"
    
    start_time = time.time()
    
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        logger.info("Skipping test: not enough GPU memory")
        return  # Skip if not enough memory
    
    device = "cuda:0"
    block_size = 64
    torch.random.manual_seed(42)
    assert nheads % nheads_k == 0
    
    # ----- Simplified input generation (from test_minimal.py) -----
    logger.info("Generating random data and masks using simplified approach")
    
    # Generate inputs
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True)
    
    # Generate masks - simple full masks
    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")

    alibi_slopes, attn_bias = None, None
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    # Generate topk indices for block sparse attention - use total number of queries after unpadding
    logger.info("Generating topk indices for block sparse attention")
    total_seqlen_q = q_unpad.shape[0]
    topk_idx = generate_topk_indices(nheads_k, total_seqlen_q, max_seqlen_k, sparsity, block_size, device)
    # Also generate block mask for reference implementation
    base_blockmask = convert_topk_to_base_blockmask(topk_idx, max_seqlen_k, block_size, device)
    
    head_mask_type = torch.tensor([1] * nheads_k, device=device, dtype=torch.int32)
    streaming_info = torch.tensor([0, 0] * nheads_k, device=device, dtype=torch.int32)
    
    logger.info("Running block_sparse_attn_func")
    attn_start = time.time()
    out_unpad = block_sparse_attn_func(
        q_unpad, k_unpad, v_unpad,
        cu_seqlens_q, cu_seqlens_k,
        head_mask_type,
        streaming_info,
        topk_idx,  # Use topk_idx directly instead of base_blockmask
        max_seqlen_q, max_seqlen_k,
        p_dropout,
        deterministic=False,
        softmax_scale=None,
        is_causal=causal,
        exact_streaming=exact_streaming,
        return_attn_probs=False,
        block_window_size=block_window_size,
    )
    logger.info(f"block_sparse_attn_func completed in {time.time() - attn_start:.2f}s")
    
    out = output_pad_fn(out_unpad)
    
    # Create expanded mask for reference implementation
    logger.info("Creating expanded mask for reference implementation")
    mixed_mask = prepare_mixed_mask(base_blockmask, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k, batch_size, m_block_dim=1, n_block_dim=block_size, block_window_size=block_window_size)

    logger.info("Computing reference implementation")
    torch.cuda.empty_cache()
    ref_start = time.time()
    out_ref, attn_ref = attention_blocksparse_ref(
            q,
            k,
            v,
            mixed_mask,
            query_padding_mask,
            key_padding_mask,
            p_dropout,
            None,  # dropout_mask
            causal=causal,
        )
    logger.info(f"Reference implementation completed in {time.time() - ref_start:.2f}s")
    
    # Free memory after reference computation to avoid OOM
    torch.cuda.empty_cache()
    gc.collect()
    
    logger.info("Computing PyTorch implementation")
    pt_start = time.time()
    out_pt, attn_pt = attention_blocksparse_ref(
            q,
            k,
            v,
            mixed_mask,
            query_padding_mask,
            key_padding_mask,
            p_dropout,
            None,  # dropout_mask
            causal=causal,
            upcast=False,
            reorder_ops=True,
        )
    logger.info(f"PyTorch implementation completed in {time.time() - pt_start:.2f}s")

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    
    # Find and print forward pass difference locations
    print("\n=== DETAILED FORWARD PASS DIFFERENCE ANALYSIS ===")
    out_diff = (out - out_ref).abs()
    flat_indices = out_diff.flatten().argsort(descending=True)[:5]
    orig_indices = [np.unravel_index(idx.item(), out_diff.shape) for idx in flat_indices]
    
    print("Top 5 forward pass differences:")
    for i, idx in enumerate(orig_indices):
        batch_idx, seq_idx, head_idx, dim_idx = idx
        val_diff = out_diff[batch_idx, seq_idx, head_idx, dim_idx].item()
        val_ref = out_ref[batch_idx, seq_idx, head_idx, dim_idx].item()
        val_ours = out[batch_idx, seq_idx, head_idx, dim_idx].item()
        val_pt = out_pt[batch_idx, seq_idx, head_idx, dim_idx].item()
        
        # Calculate block indices for context
        block_idx_q = seq_idx // block_size
        
        print(f"  {i+1}. Diff={val_diff:.6f} at (batch={batch_idx}, seq={seq_idx}, head={head_idx}, dim={dim_idx})")
        print(f"     Block index: q_block={block_idx_q}")
        print(f"     Reference value: {val_ref:.6f}, Our value: {val_ours:.6f}, PyTorch value: {val_pt:.6f}")
        
        # Check if this is at a block window boundary if block_window_size is being used
        if block_window_size > 0:
            # Check if this query block might be at a boundary of the attention window
            boundary_distance = seq_idx % block_size
            is_near_boundary = boundary_distance < 2 or boundary_distance > block_size - 3
            print(f"     Near block boundary: {is_near_boundary} (position within block: {boundary_distance})")
    
    # Backward pass
    g = torch.randn_like(out)

    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        logger.info("Computing gradients")
        grad_start = time.time()
        
        # MODIFIED: Use backward() with retain_graph instead of autograd.grad
        # Create copies of tensors that require gradients for our implementation
        q_unpad_cp = q_unpad.detach().clone().requires_grad_(True)
        k_unpad_cp = k_unpad.detach().clone().requires_grad_(True)
        v_unpad_cp = v_unpad.detach().clone().requires_grad_(True)
        
        # Run forward pass with the copies
        out_unpad_cp = block_sparse_attn_func(
            q_unpad_cp, k_unpad_cp, v_unpad_cp,
            cu_seqlens_q, cu_seqlens_k,
            head_mask_type,
            streaming_info,
            topk_idx,
            max_seqlen_q, max_seqlen_k,
            p_dropout,
            deterministic=False,
            softmax_scale=None,
            is_causal=causal,
            exact_streaming=exact_streaming,
            return_attn_probs=False,
            block_window_size=block_window_size,
        )
        
        # Compute gradient using backward()
        out_cp = output_pad_fn(out_unpad_cp)
        # Get same gradient shape
        g_unpad = torch.autograd.grad(out_cp, out_unpad_cp, g)[0]
        out_unpad_cp.backward(g_unpad, retain_graph=True)
        
        # Extract gradients
        dq_unpad = q_unpad_cp.grad
        dk_unpad = k_unpad_cp.grad
        dv_unpad = v_unpad_cp.grad
        
        # Replace NaN values with infinity in gradients
        dq_unpad = torch.nan_to_num(dq_unpad, nan=float('inf'))
        dk_unpad = torch.nan_to_num(dk_unpad, nan=float('inf'))
        dv_unpad = torch.nan_to_num(dv_unpad, nan=float('inf'))
        
        dk = dk_pad_fn(dk_unpad)
        dv = dk_pad_fn(dv_unpad)
        dq = dq_pad_fn(dq_unpad)
        
        # Use autograd.grad for reference implementations (they don't use checkpointing)
        (
            dq_ref,
            dk_ref,
            dv_ref,
        ) = torch.autograd.grad(out_ref, (q, k, v), g)
        
        (
            dq_pt,
            dk_pt,
            dv_pt,
        ) = torch.autograd.grad(out_pt, (q, k, v), g)
        
        logger.info(f"Gradient computation completed in {time.time() - grad_start:.2f}s")

        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")
    
    # Ensure memory is freed before next test
    torch.cuda.empty_cache()
    gc.collect()
    
    logger.info(f"Test completed in {time.time() - start_time:.2f}s")
    
    max_diff = (out - out_ref).abs().max().item()
    pt_max_diff = (out_pt - out_ref).abs().max().item()
    
    # Forward check
    if max_diff <= 2 * pt_max_diff:
        print("✅ Test PASSED: Block sparse attention matches reference within tolerance")
        fwd_pass = True
    else:
        print(f"❌ Test FAILED: Block sparse attention difference ({max_diff}) exceeds tolerance (2 * {pt_max_diff})")
        fwd_pass = False
    
    # Backward check
    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        dq_max_diff = (dq - dq_ref).abs().max().item()
        dk_max_diff = (dk - dk_ref).abs().max().item()
        dv_max_diff = (dv - dv_ref).abs().max().item()
        
        dq_pt_max_diff = (dq_pt - dq_ref).abs().max().item()
        dk_pt_max_diff = (dk_pt - dk_ref).abs().max().item()
        dv_pt_max_diff = (dv_pt - dv_ref).abs().max().item()
        
        if (dq_max_diff <= 3 * dq_pt_max_diff and 
            dk_max_diff <= 3 * dk_pt_max_diff and 
            dv_max_diff <= 3 * dv_pt_max_diff):
            print("✅ Backward Test PASSED: Gradients match reference within tolerance")
            bwd_pass = True
        else:
            print(f"❌ Backward Test FAILED: Gradient differences exceed tolerance")
            
            # Add detailed debugging for gradient differences
            print(f"\n=== DETAILED GRADIENT ANALYSIS ===")
            print(f"Block window size: {block_window_size}")
            
            # Find the indices with largest differences
            dq_diff = (dq - dq_ref).abs()
            dk_diff = (dk - dk_ref).abs()
            dv_diff = (dv - dv_ref).abs()
            
            # Get the top 5 worst indices for each gradient
            def get_top_indices(diff_tensor, name):
                flat_indices = diff_tensor.flatten().argsort(descending=True)[:5]
                orig_indices = [np.unravel_index(idx.item(), diff_tensor.shape) for idx in flat_indices]
                print(f"\nTop 5 {name} differences:")
                for i, idx in enumerate(orig_indices):
                    batch_idx, seq_idx, head_idx, dim_idx = idx
                    val_impl = diff_tensor[batch_idx, seq_idx, head_idx, dim_idx].item()
                    val_ref = dq_ref[batch_idx, seq_idx, head_idx, dim_idx].item() if name == "dQ" else \
                              dk_ref[batch_idx, seq_idx, head_idx, dim_idx].item() if name == "dK" else \
                              dv_ref[batch_idx, seq_idx, head_idx, dim_idx].item()
                    val_ours = dq[batch_idx, seq_idx, head_idx, dim_idx].item() if name == "dQ" else \
                               dk[batch_idx, seq_idx, head_idx, dim_idx].item() if name == "dK" else \
                               dv[batch_idx, seq_idx, head_idx, dim_idx].item()
                    
                    # Calculate block indices for context
                    block_idx_q = seq_idx // block_size
                    block_idx_k = seq_idx // block_size if name == "dQ" else seq_idx // block_size
                    
                    print(f"  {i+1}. Diff={val_impl:.6f} at (batch={batch_idx}, seq={seq_idx}, head={head_idx}, dim={dim_idx})")
                    print(f"     Block indices: q_block={block_idx_q}, k_block={block_idx_k}")
                    print(f"     Reference value: {val_ref:.6f}, Our value: {val_ours:.6f}")
                    
                    # Check if this is at a block window boundary
                    if block_window_size > 0:
                        is_boundary = abs(block_idx_q - block_idx_k) == block_window_size
                        print(f"     At window boundary: {is_boundary}")
                
                return orig_indices
            
            dq_indices = get_top_indices(dq_diff, "dQ")
            dk_indices = get_top_indices(dk_diff, "dK")
            dv_indices = get_top_indices(dv_diff, "dV")
            
            # Print mixed mask values at error locations
            print("\n=== MIXED MASK VALUES AT ERROR LOCATIONS ===")
            
            def print_mask_values(indices, name):
                print(f"\nMixed mask values for top {name} differences:")
                for i, idx in enumerate(indices):
                    batch_idx, seq_idx, head_idx, dim_idx = idx
                    
                    # Calculate block indices
                    block_idx_q = seq_idx // block_size
                    block_idx_k_list = []
                    
                    # For each possible k_seq (looking at surrounding blocks)
                    for k_block in range(max(0, block_idx_q - block_window_size - 1), 
                                        min(seqlen_k // block_size, block_idx_q + block_window_size + 2)):
                        # Look at center position in each k_block
                        k_seq = k_block * block_size + block_size // 2
                        if k_seq < seqlen_k:
                            # Check mask value (per head if mask has head dimension)
                            if len(mixed_mask.shape) == 4:  # [batch, head, q, k]
                                mask_val = mixed_mask[batch_idx, head_idx % nheads_k, seq_idx, k_seq].item()
                            else:  # [batch, q, k]
                                mask_val = mixed_mask[batch_idx, seq_idx, k_seq].item()
                            
                            # Only show non-zero/true mask values to reduce output
                            if mask_val:
                                block_idx_k_list.append((k_block, mask_val))
                    
                    print(f"  {i+1}. Location: (batch={batch_idx}, seq={seq_idx}, head={head_idx}, dim={dim_idx})")
                    print(f"     Q block: {block_idx_q}")
                    print(f"     K blocks with non-zero mask: {block_idx_k_list}")
            
            print_mask_values(dq_indices, "dQ")
            print_mask_values(dk_indices, "dK")
            print_mask_values(dv_indices, "dV")
            
            # Print the base block mask structure
            print("\n=== BASE BLOCK MASK STRUCTURE ===")
            # Count number of active blocks per head
            active_blocks_per_head = []
            for h in range(nheads_k):
                if len(base_blockmask.shape) == 3:  # [head, q_blocks, k_blocks]
                    active = base_blockmask[h].sum().item()
                else:  # [q_blocks, k_blocks]
                    active = base_blockmask.sum().item()
                active_blocks_per_head.append(active)
            
            print(f"Active blocks per head: {active_blocks_per_head}")
            print(f"Total blocks: {(max_seqlen_q // block_size) * (max_seqlen_k // block_size)}")
            
            # Check topk indices structure
            print("\n=== TOPK INDICES STRUCTURE ===")
            print(f"topk_idx shape: {topk_idx.shape}")
            print(f"Sample of topk_idx values (first 10 per head):")
            for h in range(min(3, nheads_k)):  # Show only first 3 heads to avoid excessive output
                print(f"  Head {h}: {topk_idx[h, :10].cpu().tolist()}")
            
            bwd_pass = False
        
        return fwd_pass and bwd_pass
    
    return fwd_pass


if __name__ == "__main__":
    # Define test configurations - focus on problem cases
    test_configs = [
        # seqlen_q, seqlen_k, d, p_dropout, causal, exact_streaming, sink_num, local_num, mha_type, dtype, sparsity, batch_size, nheads, nheads_k, block_window_size
        (2048, 2048, 128, 0.0, True, False, 0, 0, "gqa", torch.bfloat16, 0.7, 2, 32, 2, 0),
        (2048, 2048, 128, 0.0, True, False, 0, 0, "gqa", torch.bfloat16, 0.7, 2, 32, 2, 2048//64),
        # Only run the failing test case for detailed debugging
        (2048, 2048, 128, 0.0, True, False, 0, 0, "gqa", torch.bfloat16, 1.0, 2, 32, 2, 2),
        (1024, 1024, 64, 0.0, True, False, 0, 0, "gqa", torch.bfloat16, 0.8, 2, 32, 2, 3),
    ]
    
    # Run tests
    results = []
    for config in test_configs:
        print("\n" + "="*80)
        print(f"Running test with config: {config}")
        print("="*80)
        # Clear memory before each test
        torch.cuda.empty_cache()
        gc.collect()
        
        result = test_flash_attn_varlen_block_output(*config)
        results.append(result)
        
        # Ensure memory is fully cleared after each test
        torch.cuda.empty_cache()
        gc.collect()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for i, (config, result) in enumerate(zip(test_configs, results)):
        status = "PASSED" if result else "FAILED"
        print(f"Test {i+1}: {status} - sparsity={config[10]}, dropout={config[3]}, block_window={config[14]}")
    
    # Overall result
    if all(results):
        print("\nAll tests PASSED! 🎉")
    else:
        print("\nSome tests FAILED! ❌")