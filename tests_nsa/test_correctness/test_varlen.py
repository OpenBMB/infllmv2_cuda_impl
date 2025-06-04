# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py

import torch
from einops import repeat
from flash_attn import flash_attn_varlen_func
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
    print_topk_idx,
    move_sliding_to_topk_dix,
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


skip_ref = False
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
    # query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    # key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
    query_padding_mask = None
    key_padding_mask = None

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
    # print(f"total_seqlen_q: {total_seqlen_q}, max_seqlen_k: {max_seqlen_k}")
    topk_idx = generate_topk_indices(nheads_k, total_seqlen_q, max_seqlen_k, sparsity, block_size, block_window_size * block_size, device)
    # print_topk_idx(topk_idx, block_size, min(16, max_seqlen_k // block_size))
    # Also generate block mask for reference implementation
    base_blockmask = convert_topk_to_base_blockmask(topk_idx, max_seqlen_k, block_size, device)
    # print(f"base_blockmask: {base_blockmask.shape}")
    # print(f"base_blockmask: {list(enumerate(base_blockmask[0, :6*block_size+1:block_size]))}")
    
    head_mask_type = torch.tensor([1] * nheads_k, device=device, dtype=torch.int32)
    streaming_info = torch.tensor([0, 0] * nheads_k, device=device, dtype=torch.int32)
    
    logger.info("Running block_sparse_attn_func")
    attn_start = time.time()
    out_unpad = flash_attn_varlen_func(
        q_unpad, k_unpad, v_unpad,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        p_dropout,
        deterministic=False,
        softmax_scale=None,
        causal=causal,
        return_attn_probs=False,
        block_window_size=block_window_size,
        topk_idx=topk_idx,  # Use topk_idx directly instead of base_blockmask
    )
    logger.info(f"block_sparse_attn_func completed in {time.time() - attn_start:.2f}s")
    out = output_pad_fn(out_unpad)
    
    # Create expanded mask for reference implementation
    logger.info("Creating expanded mask for reference implementation")
    mixed_mask = prepare_mixed_mask(base_blockmask, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k, batch_size, nheads, nheads_k, m_block_dim=1, n_block_dim=block_size, block_window_size=block_window_size)
    # print(f"mixed_mask: {mixed_mask.shape}")
    # print(f"mixed_mask: {list(enumerate(mixed_mask[0, 0, :6*block_size+1:block_size//2, ::block_size]))}\n")
    
    # Free memory after reference computation to avoid OOM
    torch.cuda.empty_cache()
    gc.collect()
    
    logger.info("Computing PyTorch implementation")
    pt_start = time.time()
    out_pt, _ = attention_blocksparse_ref(
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

    if skip_ref:
        logger.info("Skipping reference implementation")
        out_ref = out_pt
    else:
        logger.info("Computing reference implementation")
        ref_start = time.time()
        out_ref, _ = attention_blocksparse_ref(
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

    torch.cuda.empty_cache()

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
    
    return fwd_pass


if __name__ == "__main__":
    # Define test configurations - focus on problem cases
    test_configs = [
        # seqlen_q, seqlen_k, d, p_dropout, causal, exact_streaming, sink_num, local_num, mha_type, dtype, sparsity, batch_size, nheads, nheads_k, block_window_size
        (256 , 256 , 128, 0.0, True,  False, 0, 0, "gqa", torch.float16, 0, 1, 32, 2, 64//64),
        (2048, 2048, 128, 0.0, False, False, 0, 0, "gqa", torch.float16, 0.7, 1, 32, 2, 0),
        (2048, 2048, 128, 0.0, False, False, 0, 0, "gqa", torch.float16, 0.7, 1, 32, 2, 2048//64),
        (2048, 2048, 128, 0.0, False, False, 0, 0, "gqa", torch.float16, 0.25, 1, 32, 2, 512//64),
        (2048, 2048, 128, 0.0, False, False, 0, 0, "gqa", torch.float16, 0, 1, 32, 2, 512//64),
        # Only run the failing test case for detailed debugging
        (2048, 2048, 128, 0.0, False, False, 0, 0, "gqa", torch.float16, 1.0, 1, 32, 2, 0),
        (1024, 1024, 128, 0.0, False, False, 0, 0, "gqa", torch.float16, 0.8, 1, 32, 2, 0),
        (2048, 2048, 128, 0.0, True, False, 0, 0, "gqa", torch.float16, 0.7, 1, 32, 2, 0),
        (2048, 2048, 128, 0.0, True, False, 0, 0, "gqa", torch.float16, 0.7, 1, 32, 2, 2048//64),
        (2048, 2048, 128, 0.0, True, False, 0, 0, "gqa", torch.float16, 0.25, 1, 32, 2, 512//64),
        (2048, 2048, 128, 0.0, True, False, 0, 0, "gqa", torch.float16, 0, 1, 32, 2, 512//64),
        # Only run the failing test case for detailed debugging
        (2048, 2048, 128, 0.0, True, False, 0, 0, "gqa", torch.float16, 1.0, 1, 32, 2, 1),
        (1024, 1024, 128, 0.0, True, False, 0, 0, "gqa", torch.float16, 0.8, 1, 32, 2, 1),
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
        print(f"Test {i+1}: {status} - sparsity={config[10]}, "
              f"block_topk={ max(1, int((config[1]//block_size)*(1-config[10])))}, "
              f"block_window={config[14]}")
    
    # Overall result
    if all(results):
        print("\nAll tests PASSED! 🎉")
    else:
        print("\nSome tests FAILED! ❌")