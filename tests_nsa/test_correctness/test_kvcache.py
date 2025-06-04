
import torch
from einops import repeat
from flash_attn import flash_attn_with_kvcache
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import (
    generate_random_padding_mask,
    generate_base_sparsity_mask,
    generate_qkv,
    prepare_batch_mixed_mask,
    convert_flash_attn_S_to_softmax,
    normalize_flash_attn_S,
    get_dropout_fraction,
    attention_blocksparse_ref,
    convert_topk_to_base_blockmask,
    convert_batch_topk_to_base_blockmask,
    generate_batch_topk_indices,
    visualize_output_differences,
    visualize_gradient_differences,
    visualize_blockmask_accuracy,
)

from nsa import topk_to_uint64 as cuda_topk_to_uint64  
import logging
import time
import gc

import numpy as np
import os
from pathlib import Path
VISUALIZATION_DIR = Path("/home/test/test01/zwl/zwl_nsa/gradient_visualizations")
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


MAX_HEADDIM_SM8x = 192
block_size = 64
is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)

def test_kvcache_correctness(
    seqlen_q, seqlen_k, d, causal, exact_streaming, sink_num, local_num, dtype, num_topk, batch_size, nheads, nheads_k,
    block_window_size=0
):
    logger.info(f"Starting test with parameters: seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, d={d}, "
                f"causal={causal}, exact_streaming={exact_streaming}, sink_num={sink_num}, local_num={local_num}, "
                f"dtype={dtype}, num_topk={num_topk}, batch_size={batch_size}, nheads={nheads}, "
                f"block_window_size={block_window_size}")
    
    # Create a unique config name for this test
    config_name = f"kvcache_s{seqlen_q}x{seqlen_k}_d{d}_h{nheads}_kv{nheads_k}_topk{num_topk}"
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
    torch.random.manual_seed(44)
    assert nheads % nheads_k == 0
    
    # ----- Create KV Cache inputs -----
    logger.info("Generating random data and masks for KV cache test")
    
    # First, create full sequence for reference implementation
    # Forward-pass only: set requires_grad=False
    q_full = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=False)
    k_full = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
    v_full = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
    
    # Generate masks - simple full masks (no padding for simplicity)
    query_padding_mask = torch.ones(batch_size, seqlen_q, device=device, dtype=torch.bool)
    key_padding_mask = torch.ones(batch_size, seqlen_k, device=device, dtype=torch.bool)

    # Simulate new tokens for KV cache 
    # For simplicity, let's say we have 1 new token to add to the cache
    seqlen_knew = 1
    k_new = torch.randn(batch_size, seqlen_knew, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
    v_new = torch.randn(batch_size, seqlen_knew, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
    
    # Create KV cache (this would contain the cached keys and values)
    # For the first test, make a simple scenario: 
    # - kcache and vcache contain the first seqlen_k - seqlen_knew tokens
    # - k_new and v_new contain the remaining token(s)
    seqlens_k = torch.full((batch_size,), seqlen_k - seqlen_knew, dtype=torch.int32, device=device)
    kcache = k_full[:, :-seqlen_knew].contiguous() 
    vcache = v_full[:, :-seqlen_knew].contiguous()
    
    # For simplicity, use same batch indices
    cache_batch_idx = torch.arange(batch_size, device=device, dtype=torch.int32)
    
    # Generate unpadded tensors for the implementation under test
    q_unpad, _, _, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, _, _, _, output_pad_fn, _, _ = \
        generate_qkv(q_full, k_full, v_full, query_padding_mask, key_padding_mask, kvpacked=False)
    
    # Generate topk indices for block sparse attention
    logger.info("Generating topk indices for block sparse attention")
    topk_idx = generate_batch_topk_indices(nheads_k, batch_size, seqlen_q, max_seqlen_k, num_topk, block_size, device, mode="random")
    # Also generate block mask for reference implementation
    base_blockmask = convert_batch_topk_to_base_blockmask(topk_idx, max_seqlen_k, block_size, device)

    topk_uint64, _ = cuda_topk_to_uint64(topk_idx, max_seqlen_k, block_size)
    
    # Prepare rotary embeddings (optional - set to None for initial testing)
    
    rotary_cos = None
    rotary_sin = None

    logger.info(f"Running nsa_attn_with_kvcache {topk_idx}")
    kvcache_start = time.time()
    out = flash_attn_with_kvcache(
        q=q_full,                # [batch_size, seqlen_q, nheads, d]
        k_cache=kcache,          # [batch_size, seqlen_k-1, nheads_k, d] 
        v_cache=vcache,          # [batch_size, seqlen_k-1, nheads_k, d]
        topk_idx=topk_idx,
        block_window_size=block_window_size,
        k=k_new,                 # [batch_size, 1, nheads_k, d]
        v=v_new,                 # [batch_size, 1, nheads_k, d]
        cache_seqlens=seqlens_k,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_batch_idx=cache_batch_idx,
        causal=causal,
        num_splits=16
    )
    logger.info(f"nsa_attn_with_kvcache completed in {time.time() - kvcache_start:.2f}s")

    # ----- Reference Implementation ------
    # For the reference implementation, we'll use the normal block_sparse_attn function 
    # with the full k and v tensors (concatenating cache and new tokens)
    
    # Create expanded mask for reference implementation
    logger.info("Creating expanded mask for reference implementation")
    mixed_mask = prepare_batch_mixed_mask(base_blockmask, seqlen_q, seqlen_k, nheads=nheads, nheads_k=nheads_k, n_block_dim=block_size, block_window_size=block_window_size)

    logger.info("Computing reference implementation")
    torch.cuda.empty_cache()
    ref_start = time.time()
    
    # For reference, combine the cache with the new tokens
    k_full_ref = torch.cat([kcache, k_new], dim=1)
    v_full_ref = torch.cat([vcache, v_new], dim=1)
    
    out_ref, attn_ref = attention_blocksparse_ref(
            q_full,
            k_full_ref,
            v_full_ref,
            mixed_mask,
            query_padding_mask,
            key_padding_mask,
            0.0,    # p_dropout
            None,   # dropout_mask
            causal=causal,
        )
    logger.info(f"Reference implementation completed in {time.time() - ref_start:.2f}s")
    
    # Free memory after reference computation
    torch.cuda.empty_cache()
    gc.collect()
    
    # Compare outputs
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    
    # Add output difference visualization
    logger.info("Generating output difference visualizations")
    visualize_output_differences(out, out_ref, out_ref, config_name)  # Using out_ref twice since we don't have a PyTorch implementation
    
    # Add our new visualization for blockmask accuracy
    logger.info("Generating blockmask accuracy visualizations")
    visualize_blockmask_accuracy(out, out_ref, mixed_mask, config_name)
    
    logger.info(f"Output visualizations saved to {VISUALIZATION_DIR / config_name}")
    
    # Note: Backward pass is skipped in this version
    
    # Ensure memory is freed before next test
    torch.cuda.empty_cache()
    gc.collect()
    
    logger.info(f"Test completed in {time.time() - start_time:.2f}s")
    
    # Evaluate results
    max_diff = (out - out_ref).abs().max().item()
    
    # Forward check tolerance
    tolerance = 1e-2  # Adjust based on expected precision
    
    if max_diff <= tolerance:
        print("✅ Test PASSED: KV cache attention matches reference within tolerance")
        fwd_pass = True
    else:
        print(f"❌ Test FAILED: KV cache attention difference ({max_diff}) exceeds tolerance ({tolerance})")
        fwd_pass = False
    
    return fwd_pass

def test_kvcache_multiple_batches(
    seqlen_q, seqlen_k, d, causal, exact_streaming, dtype, num_topk, batch_size, nheads, nheads_k,
    block_window_size=0
):
    """Test KV cache with multiple batches, varying cache locations."""
    
    logger.info(f"Starting multi-batch test with parameters: seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, "
                f"num_topk={num_topk}, batch_size={batch_size}, block_window_size={block_window_size}")
    
    config_name = f"kvcache_multibatch_s{seqlen_q}x{seqlen_k}_d{d}_h{nheads}_kv{nheads_k}_topk{num_topk}"
    if block_window_size > 0:
        config_name += f"_window{block_window_size}"
    
    device = "cuda:0"
    block_size = 64
    torch.random.manual_seed(44)
    
    # For multi-batch test, create a larger cache for all batches combined
    cache_batch_size = batch_size * 2  # Make cache larger than actual batch
    
    # Generate random sequence lengths for each batch element (between seqlen_k-20 and seqlen_k)
    min_seqlen_k = max(seqlen_k - 20, 1)  # Ensure min length is at least 1
    random_seqlens_k = torch.randint(
        min_seqlen_k, seqlen_k + 1, (batch_size,), device=device, dtype=torch.int32
    )
    logger.info(f"Generated variable sequence lengths between {min_seqlen_k} and {seqlen_k}")
    
    # Create full query, key, and value tensors for the reference implementation
    # For forward-only testing, we don't need gradient tracking
    q_full = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    
    # Create k and v with variable lengths
    k_full = torch.zeros(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    v_full = torch.zeros(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    
    # Fill k_full and v_full with random data up to the actual sequence length for each batch
    for i in range(batch_size):
        actual_seqlen_k = random_seqlens_k[i].item()
        k_full[i, :actual_seqlen_k] = torch.randn(actual_seqlen_k, nheads_k, d, device=device, dtype=dtype)
        v_full[i, :actual_seqlen_k] = torch.randn(actual_seqlen_k, nheads_k, d, device=device, dtype=dtype)
    
    # Create KV cache that's larger than needed for current batch
    # Important: set requires_grad=False to avoid in-place operation errors
    kcache = torch.randn(cache_batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
    vcache = torch.randn(cache_batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
    
    # Simulate different batch sizes stored at different locations in the cache
    # Create non-sequential cache_batch_idx to test the indexing
    cache_batch_idx = torch.randperm(cache_batch_size)[:batch_size].to(device=device, dtype=torch.int32)
    
    # Copy our test batch data into the selected cache locations
    for i, cache_idx in enumerate(cache_batch_idx):
        actual_seqlen_k = random_seqlens_k[i].item()
        # Only fill up to the actual sequence length for this batch
        kcache[cache_idx, :actual_seqlen_k] = k_full[i, :actual_seqlen_k]
        vcache[cache_idx, :actual_seqlen_k] = v_full[i, :actual_seqlen_k]
    
    # No new tokens in this test (seqlen_knew = 0), but use the actual sequence lengths
    seqlens_k = random_seqlens_k
    
    # Prepare padding masks and unpadded tensors
    query_padding_mask = torch.ones(batch_size, seqlen_q, device=device, dtype=torch.bool)
    key_padding_mask = torch.zeros(batch_size, seqlen_k, device=device, dtype=torch.bool)
    for i in range(batch_size):
        key_padding_mask[i, :random_seqlens_k[i]] = True
    
    q_unpad, _, _, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, _, _, _, output_pad_fn, _, _ = \
        generate_qkv(q_full, k_full, v_full, query_padding_mask, key_padding_mask, kvpacked=False)
    
    # Generate topk indices for block sparse attention
    topk_idx = generate_batch_topk_indices(nheads_k, batch_size, seqlen_q, max_seqlen_k, num_topk, block_size, device)
    base_blockmask = convert_batch_topk_to_base_blockmask(topk_idx, max_seqlen_k, block_size, device)
    
    head_mask_type = torch.tensor([1] * nheads_k, device=device, dtype=torch.int32)
    streaming_info = None #torch.tensor([0, 0] * nheads_k, device=device, dtype=torch.int32)
    
    # Run the KV cache implementation
    out = flash_attn_with_kvcache(
        q=q_full,                # [batch_size, seqlen_q, nheads, d]
        k_cache=kcache,          # [cache_batch_size, seqlen_k, nheads_k, d]
        v_cache=vcache,          # [cache_batch_size, seqlen_k, nheads_k, d]
        topk_idx=topk_idx,
        block_window_size=block_window_size,
        k=None,                  # No new keys
        v=None,                  # No new values
        cache_seqlens=seqlens_k,
        rotary_cos=None,         # No rotary embeddings
        rotary_sin=None,         # No rotary embeddings
        cache_batch_idx=cache_batch_idx,
        causal=causal,
        num_splits=16
    )
    
    # Run reference implementation
    mixed_mask = prepare_batch_mixed_mask(base_blockmask, seqlen_q, seqlen_k, nheads=nheads, nheads_k=nheads_k, n_block_dim=block_size, block_window_size=block_window_size)
    
    # Modify the reference implementation to match the cache indexing
    # We need to use the same sequence of batches as used by the KV cache
    # For the reference implementation, create k_full_ref and v_full_ref
    # that match the layout in kcache and vcache based on cache_batch_idx
    
    # Create a version of k_full and v_full that matches the batch indexing in cache_batch_idx
    # In the real implementation, kcache[cache_batch_idx[i]] is used instead of k_full[i]
    k_full_ref = torch.zeros_like(k_full)
    v_full_ref = torch.zeros_like(v_full)
    
    # For each batch index, copy the data from the cache location
    for i in range(batch_size):
        cache_idx = cache_batch_idx[i]
        actual_seqlen_k = random_seqlens_k[i].item()
        # Only copy up to the actual sequence length for this batch
        k_full_ref[i, :actual_seqlen_k] = kcache[cache_idx, :actual_seqlen_k]
        v_full_ref[i, :actual_seqlen_k] = vcache[cache_idx, :actual_seqlen_k]
    
    # Now run the reference implementation with aligned batch indices
    out_ref, _ = attention_blocksparse_ref(
        q_full,
        k_full_ref,
        v_full_ref,
        mixed_mask,
        query_padding_mask,
        key_padding_mask,
        0.0,  # p_dropout
        None, # dropout_mask
        causal=causal,
    )
    
    # Compare outputs
    max_diff = (out - out_ref).abs().max().item()
    mean_diff = (out - out_ref).abs().mean().item()
    
    print(f"Multi-batch KV cache test - max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")
    
    # Add visualization for multi-batch tests
    logger.info("Generating output and blockmask visualizations for multi-batch test")
    visualize_output_differences(out, out_ref, out_ref, config_name)
    visualize_blockmask_accuracy(out, out_ref, mixed_mask, config_name)
    logger.info(f"Visualizations saved to {VISUALIZATION_DIR / config_name}")
    
    # Tolerance check
    tolerance = 1e-2
    if max_diff <= tolerance:
        print("✅ Multi-batch KV cache test PASSED")
        return True
    else:
        print(f"❌ Multi-batch KV cache test FAILED: difference ({max_diff}) exceeds tolerance ({tolerance})")
        return False

def test_kvcache_incremental_decoding(
    seqlen_q, max_seqlen_k, d, causal, exact_streaming, dtype, num_topk, batch_size, nheads, nheads_k, 
    num_steps=5, block_window_size=0
):
    """Test incremental decoding scenario with KV cache."""
    
    logger.info(f"Starting incremental decoding test with parameters: "
                f"seqlen_q={seqlen_q}, max_seqlen_k={max_seqlen_k}, "
                f"num_topk={num_topk}, batch_size={batch_size}, num_steps={num_steps}, "
                f"block_window_size={block_window_size}")
    
    config_name = f"kvcache_incremental_s{seqlen_q}x{max_seqlen_k}_d{d}_h{nheads}_kv{nheads_k}_topk{num_topk}_steps{num_steps}"
    if block_window_size > 0:
        config_name += f"_window{block_window_size}"
    
    device = "cuda:0"
    block_size = 64
    torch.random.manual_seed(44)
    
    # Generate random starting sequence lengths for each batch (between max_seqlen_k-20 and max_seqlen_k)
    # This simulates having different prefills of variable lengths
    min_seqlen_k = max(max_seqlen_k - 20, 1)  # Ensure min length is at least 1
    initial_seqlens_k = torch.randint(
        min_seqlen_k, max_seqlen_k + 1, (batch_size,), device=device, dtype=torch.int32
    )
    logger.info(f"Generated variable initial sequence lengths between {min_seqlen_k} and {max_seqlen_k}")
    
    # Create the initial cache with zeros
    kcache = torch.zeros(batch_size, max_seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
    vcache = torch.zeros(batch_size, max_seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
    
    # Fill initial cache with random data up to initial_seqlens_k for each batch
    for i in range(batch_size):
        initial_seqlen = initial_seqlens_k[i].item()
        if initial_seqlen > 0:
            kcache[i, :initial_seqlen] = torch.randn(initial_seqlen, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
            vcache[i, :initial_seqlen] = torch.randn(initial_seqlen, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
    
    cache_batch_idx = torch.arange(batch_size, device=device, dtype=torch.int32)
    
    # For each decoding step
    all_passed = True
    current_seqlens_k = initial_seqlens_k.clone()
    
    for step in range(num_steps):
        logger.info(f"Testing incremental decoding step {step+1}/{num_steps}")
        
        # For each step, we generate a single new token
        seqlen_knew = 1
        k_new = torch.randn(batch_size, seqlen_knew, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
        v_new = torch.randn(batch_size, seqlen_knew, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
        
        # Create a temporary k_full_temp and v_full_temp for the reference implementation
        # These include all tokens up to current length plus the new token
        k_full_temp = torch.zeros(batch_size, max_seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
        v_full_temp = torch.zeros(batch_size, max_seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
        
        for i in range(batch_size):
            current_seqlen = current_seqlens_k[i].item()
            # Copy existing tokens from cache
            if current_seqlen > 0:
                k_full_temp[i, :current_seqlen] = kcache[i, :current_seqlen]
                v_full_temp[i, :current_seqlen] = vcache[i, :current_seqlen]
            # Add the new token at the current position
            if current_seqlen < max_seqlen_k:
                k_full_temp[i, current_seqlen:current_seqlen+seqlen_knew] = k_new[i]
                v_full_temp[i, current_seqlen:current_seqlen+seqlen_knew] = v_new[i]
        
        # Generate query for this step
        q_full = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=False)
        
        # Prepare padding masks
        query_padding_mask = torch.ones(batch_size, seqlen_q, device=device, dtype=torch.bool)
        key_padding_mask = torch.zeros(batch_size, max_seqlen_k, device=device, dtype=torch.bool)
        
        # Create key padding mask that reflects the variable current lengths plus new token
        for i in range(batch_size):
            current_seqlen = current_seqlens_k[i].item()
            new_seqlen = min(current_seqlen + seqlen_knew, max_seqlen_k)
            key_padding_mask[i, :new_seqlen] = True
        
        # We need unpadded tensors but our generate_qkv expects full k, v
        q_unpad, _, _, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, cur_max_seqlen_k, _, _, _, output_pad_fn, _, _ = \
            generate_qkv(q_full, k_full_temp, v_full_temp, query_padding_mask, key_padding_mask, kvpacked=False)
        
        # Generate topk indices for block sparse attention
        topk_idx = generate_batch_topk_indices(nheads_k, batch_size, seqlen_q, max_seqlen_k, num_topk, block_size, device)
        base_blockmask = convert_batch_topk_to_base_blockmask(topk_idx, max_seqlen_k, block_size, device)
        
        head_mask_type = torch.tensor([1] * nheads_k, device=device, dtype=torch.int32)
        streaming_info = None #torch.tensor([0, 0] * nheads_k, device=device, dtype=torch.int32)
        
        # Run the KV cache implementation
        out = flash_attn_with_kvcache(
            q=q_full,                # [batch_size, seqlen_q, nheads, d]
            k_cache=kcache,          # [batch_size, max_seqlen_k, nheads_k, d] 
            v_cache=vcache,          # [batch_size, max_seqlen_k, nheads_k, d]
            topk_idx=topk_idx,
            block_window_size=block_window_size,
            k=k_new,                  # No new keys
            v=v_new,                  # No new values
            cache_seqlens=current_seqlens_k, # Current positions in cache
            rotary_cos=None,         # No rotary embeddings
            rotary_sin=None,         # No rotary embeddings
            cache_batch_idx=cache_batch_idx,
            causal=causal,
            num_splits=16
        )
        
        # Run reference implementation with current cache plus new tokens
        mixed_mask = prepare_batch_mixed_mask(base_blockmask, seqlen_q, max_seqlen_k, 
                                       nheads=nheads, nheads_k=nheads_k, n_block_dim=block_size,
                                       block_window_size=block_window_size)
        
        out_ref, _ = attention_blocksparse_ref(
            q_full,
            k_full_temp,
            v_full_temp,
            mixed_mask,
            query_padding_mask,
            key_padding_mask,
            0.0,  # p_dropout
            None, # dropout_mask
            causal=causal,
        )
        
        # Compare outputs
        max_diff = (out - out_ref).abs().max().item()
        mean_diff = (out - out_ref).abs().mean().item()
        
        print(f"Step {step+1} - Variable seqlens, max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")
        
        # Add visualization for the final step only to avoid too many files
        if step == num_steps - 1:
            step_config_name = f"{config_name}_final_step"
            logger.info(f"Generating visualizations for final step (step {step+1})")
            visualize_output_differences(out, out_ref, out_ref, step_config_name)
            visualize_blockmask_accuracy(out, out_ref, mixed_mask, step_config_name)
            logger.info(f"Final step visualizations saved to {VISUALIZATION_DIR / step_config_name}")
        
        # Tolerance check
        tolerance = 1e-2
        if max_diff <= tolerance:
            print(f"✅ Step {step+1} PASSED")
        else:
            print(f"❌ Step {step+1} FAILED: difference ({max_diff}) exceeds tolerance ({tolerance})")
            all_passed = False
        
        # Update current sequence length for next step
        # Don't exceed max_seqlen_k
        current_seqlens_k = torch.min(current_seqlens_k + seqlen_knew, torch.full_like(current_seqlens_k, max_seqlen_k))
        
        # Update the cache with the new tokens for next step
        for i in range(batch_size):
            current_seqlen = current_seqlens_k[i].item() - seqlen_knew  # Before adding new token
            if current_seqlen < max_seqlen_k:
                kcache[i, current_seqlen:current_seqlen+seqlen_knew] = k_new[i]
                vcache[i, current_seqlen:current_seqlen+seqlen_knew] = v_new[i]
    
    if all_passed:
        print("✅ Incremental decoding test: All steps PASSED")
    else:
        print("❌ Incremental decoding test: Some steps FAILED")
    
    return all_passed

def test_kvcache_variable_seqlens(
    seqlen_q, max_seqlen_k, min_seqlen_k, d, causal, exact_streaming, dtype, num_topk, batch_size, nheads, nheads_k,
    block_window_size=0
):
    """Test KV cache with variable sequence lengths for each batch element."""
    
    logger.info(f"Starting variable sequence length test with parameters: "
                f"seqlen_q={seqlen_q}, max_seqlen_k={max_seqlen_k}, min_seqlen_k={min_seqlen_k}, "
                f"num_topk={num_topk}, batch_size={batch_size}, block_window_size={block_window_size}")
    
    config_name = f"kvcache_varlen_s{seqlen_q}x{min_seqlen_k}-{max_seqlen_k}_d{d}_h{nheads}_kv{nheads_k}_topk{num_topk}"
    if block_window_size > 0:
        config_name += f"_window{block_window_size}"
    
    device = "cuda:0"
    block_size = 64
    torch.random.manual_seed(43)
    
    # Generate random sequence lengths for each batch element
    # Random sequence lengths between min_seqlen_k and max_seqlen_k
    random_seqlens_k = torch.randint(
        min_seqlen_k, max_seqlen_k + 1, (batch_size,), device=device, dtype=torch.int32
    )
    
    # For logging, calculate average sequence length
    avg_seqlen_k = random_seqlens_k.float().mean().item()
    logger.info(f"Generated random sequence lengths between {min_seqlen_k} and {max_seqlen_k}, avg: {avg_seqlen_k:.1f}")
    
    # Create query tensor
    q_full = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=False)
    
    # Create KV cache with maximum size
    kcache = torch.zeros(batch_size, max_seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
    vcache = torch.zeros(batch_size, max_seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
    
    # Create random KV data for each batch with appropriate sequence length
    for i in range(batch_size):
        actual_seqlen_k = random_seqlens_k[i].item()
        k_batch = torch.randn(actual_seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
        v_batch = torch.randn(actual_seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
        
        # Fill the cache with the random data up to the actual sequence length
        kcache[i, :actual_seqlen_k] = k_batch
        vcache[i, :actual_seqlen_k] = v_batch
    
    cache_batch_idx = torch.arange(batch_size, device=device, dtype=torch.int32)
    
    # Create padding masks
    query_padding_mask = torch.ones(batch_size, seqlen_q, device=device, dtype=torch.bool)
    
    # We need to create key padding masks that reflect the variable sequence lengths
    key_padding_mask = torch.zeros(batch_size, max_seqlen_k, device=device, dtype=torch.bool)
    for i in range(batch_size):
        key_padding_mask[i, :random_seqlens_k[i]] = True
    
    # For generating query, key, value tensors, we need to create temporary full k, v tensors
    # that match the random sequence lengths
    k_full_temp = torch.zeros(batch_size, max_seqlen_k, nheads_k, d, device=device, dtype=dtype)
    v_full_temp = torch.zeros(batch_size, max_seqlen_k, nheads_k, d, device=device, dtype=dtype)
    
    for i in range(batch_size):
        actual_seqlen_k = random_seqlens_k[i].item()
        k_full_temp[i, :actual_seqlen_k] = kcache[i, :actual_seqlen_k]
        v_full_temp[i, :actual_seqlen_k] = vcache[i, :actual_seqlen_k]
    
    # Generate topk indices for block sparse attention
    topk_idx = generate_batch_topk_indices(nheads_k, batch_size, seqlen_q, max_seqlen_k, num_topk, block_size, device)
    base_blockmask = convert_batch_topk_to_base_blockmask(topk_idx, max_seqlen_k, block_size, device)
    
    head_mask_type = torch.tensor([1] * nheads_k, device=device, dtype=torch.int32)
    streaming_info = None #torch.tensor([0, 0] * nheads_k, device=device, dtype=torch.int32)
    
    # Run the KV cache implementation
    out = flash_attn_with_kvcache(
        q=q_full,                # [batch_size, seqlen_q, nheads, d]
        k_cache=kcache,          # [batch_size, max_seqlen_k, nheads_k, d] 
        v_cache=vcache,          # [batch_size, max_seqlen_k, nheads_k, d]
        topk_idx=topk_idx,
        block_window_size=block_window_size,
        k=None,                  # No new tokens
        v=None,                  # No new tokens
        cache_seqlens=random_seqlens_k,  # Variable sequence lengths
        rotary_cos=None,
        rotary_sin=None,
        cache_batch_idx=cache_batch_idx,
        causal=causal,
        num_splits=16
    )
    
    # Run reference implementation
    mixed_mask = prepare_batch_mixed_mask(base_blockmask, seqlen_q, max_seqlen_k, 
                                   nheads=nheads, nheads_k=nheads_k, n_block_dim=block_size,
                                   block_window_size=block_window_size)
    
    out_ref, _ = attention_blocksparse_ref(
        q_full,
        k_full_temp,
        v_full_temp,
        mixed_mask,
        query_padding_mask,
        key_padding_mask,
        0.0,  # p_dropout
        None, # dropout_mask
        causal=causal,
    )
    
    # Compare outputs
    max_diff = (out - out_ref).abs().max().item()
    mean_diff = (out - out_ref).abs().mean().item()
    
    print(f"Variable sequence length test - max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")
    
    # Add visualization
    logger.info("Generating output and blockmask visualizations for variable sequence length test")
    visualize_output_differences(out, out_ref, out_ref, config_name)
    visualize_blockmask_accuracy(out, out_ref, mixed_mask, config_name)
    logger.info(f"Visualizations saved to {VISUALIZATION_DIR / config_name}")
    
    # Tolerance check
    tolerance = 1e-2
    if max_diff <= tolerance:
        print("✅ Variable sequence length test PASSED")
        return True
    else:
        print(f"❌ Variable sequence length test FAILED: difference ({max_diff}) exceeds tolerance ({tolerance})")
        return False

if __name__ == "__main__":
    # Define test configurations
    basic_tests = [
        # seqlen_q, seqlen_k, d, causal, exact_streaming, sink_num, local_num, dtype, num_topk, batch_size, nheads, nheads_k, block_window_size
        # Basic tests with variable sequence lengths
        (1, 1024, 128, False, False, 0, 0, torch.float16, 8, 1, 32, 2, 0),  # No window
        
        (1, 2048, 128, False, False, 0, 0, torch.float16, 2, 1, 32, 2, 4),  # With window=4
    ]
    
    multi_batch_tests = [
        # seqlen_q, seqlen_k, d, causal, exact_streaming, dtype, num_topk, batch_size, nheads, nheads_k, block_window_size
        (1, 1024, 128, False, False, torch.float16, 8, 4, 32, 2, 0),  # Multi-batch test with variable lengths
    ]
    
    incremental_tests = [
        # seqlen_q, max_seqlen_k, d, causal, exact_streaming, dtype, num_topk, batch_size, nheads, nheads_k, num_steps, block_window_size
        (1, 1024, 128, False, False, torch.float16, 8, 2, 32, 2, 5, 0),  # Simple autoregressive gen, no window
        (1, 2048, 128, False, False, torch.float16, 8, 2, 32, 2, 5, 4),  # Longer context, with window=4
    ]
    
    # Variable sequence length tests
    variable_seqlen_tests = [
        # seqlen_q, max_seqlen_k, min_seqlen_k, d, causal, exact_streaming, dtype, num_topk, batch_size, nheads, nheads_k, block_window_size
        (1, 4096, 4000, 128, False, False, torch.float16, 8, 2, 32, 2, 4),  # With block_window_size=4
        (1, 4096, 4000, 128, True, False, torch.float16, 8, 2, 32, 2, 8),   # With causal mask, window=8
    ]
    
    # Run basic tests
    print("\n" + "="*80)
    print("BASIC KV CACHE TESTS")
    print("="*80)
    basic_results = []
    for config in basic_tests:
        print("\n" + "-"*80)
        print(f"Running test with config: {config}")
        print("-"*80)
        torch.cuda.empty_cache()
        gc.collect()
        
        result = test_kvcache_correctness(*config)
        basic_results.append(result)
        
        torch.cuda.empty_cache()
        gc.collect()
    
    # Run multi-batch tests
    print("\n" + "="*80)
    print("MULTI-BATCH KV CACHE TESTS")
    print("="*80)
    multi_batch_results = []
    for config in multi_batch_tests:
        print("\n" + "-"*80)
        print(f"Running multi-batch test with config: {config}")
        print("-"*80)
        torch.cuda.empty_cache()
        gc.collect()
        
        result = test_kvcache_multiple_batches(*config)
        multi_batch_results.append(result)
        
        torch.cuda.empty_cache()
        gc.collect()
    
    # Run incremental decoding tests
    print("\n" + "="*80)
    print("INCREMENTAL DECODING TESTS")
    print("="*80)
    incremental_results = []
    for config in incremental_tests:
        print("\n" + "-"*80)
        print(f"Running incremental test with config: {config}")
        print("-"*80)
        torch.cuda.empty_cache()
        gc.collect()
        
        result = test_kvcache_incremental_decoding(*config)
        incremental_results.append(result)
        
        torch.cuda.empty_cache()
        gc.collect()
    
    # Run variable sequence length tests
    print("\n" + "="*80)
    print("VARIABLE SEQUENCE LENGTH TESTS")
    print("="*80)
    variable_seqlen_results = []
    for config in variable_seqlen_tests:
        print("\n" + "-"*80)
        print(f"Running variable sequence length test with config: {config}")
        print("-"*80)
        torch.cuda.empty_cache()
        gc.collect()
        
        result = test_kvcache_variable_seqlens(*config)
        variable_seqlen_results.append(result)
        
        torch.cuda.empty_cache()
        gc.collect()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print("\nBasic KV Cache Tests (Forward Pass Only):")
    for i, (config, result) in enumerate(zip(basic_tests, basic_results)):
        status = "PASSED" if result else "FAILED"
        print(f"Test {i+1}: {status} - seqlen_q={config[0]}, seqlen_k={config[1]}, num_topk={config[8]}, window={config[12]}")
    
    print("\nMulti-Batch KV Cache Tests:")
    for i, (config, result) in enumerate(zip(multi_batch_tests, multi_batch_results)):
        status = "PASSED" if result else "FAILED"
        print(f"Test {i+1}: {status} - batch_size={config[7]}, seqlen_k={config[1]}, num_topk={config[6]}, window={config[10]}")
    
    print("\nIncremental Decoding Tests:")
    for i, (config, result) in enumerate(zip(incremental_tests, incremental_results)):
        status = "PASSED" if result else "FAILED"
        print(f"Test {i+1}: {status} - max_seqlen_k={config[1]}, num_topk={config[6]}, steps={config[10]}, window={config[11]}")
    
    print("\nVariable Sequence Length Tests:")
    for i, (config, result) in enumerate(zip(variable_seqlen_tests, variable_seqlen_results)):
        status = "PASSED" if result else "FAILED"
        print(f"Test {i+1}: {status} - max_seqlen_k={config[1]}, min_seqlen_k={config[2]}, num_topk={config[7]}, window={config[11]}")
    
    # Overall result
    all_passed = all(basic_results) and all(multi_batch_results) and all(incremental_results) and all(variable_seqlen_results)
    if all_passed:
        print("\nAll KV Cache Forward Pass tests PASSED! 🎉")
    else:
        print("\nSome KV Cache Forward Pass tests FAILED! ❌") 