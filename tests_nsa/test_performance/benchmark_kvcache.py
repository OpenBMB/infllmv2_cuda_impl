import torch
import time
import gc
import numpy as np
from typing import Optional, Union

from flash_attn import flash_attn_with_kvcache
HAS_FLASH_ATTN = True
# try:
#     from flash_attn import flash_attn_with_kvcache
#     HAS_FLASH_ATTN = True
# except ImportError:
#     print("Flash Attention not found. Only block sparse will be benchmarked.")
#     HAS_FLASH_ATTN = False
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import (
    generate_batch_topk_indices,
)

from nsa import topk_to_uint64 as cuda_topk_to_uint64  

# 定义默认配置
DEFAULT_CONFIG = {
    "batch_size": 1,
    "seqlen_q": 1,
    "max_seqlen_k": 32768*4,
    "nheads": 32,
    "nheads_k": 2,  # Number of key/value heads
    "headdim": 128,
    "block_size": 64,
    "num_topk": 32,
    "block_window_size": 32,  # 2048/64
    "T": 25
}

if __name__ == "__main__":
    # 使用配置字典
    config = DEFAULT_CONFIG
    
    # 从配置中读取参数
    batch_size = config["batch_size"]
    seqlen_q = config["seqlen_q"]
    max_seqlen_k = config["max_seqlen_k"]
    nheads = config["nheads"]
    nheads_k = config["nheads_k"]
    headdim = config["headdim"]
    block_size = config["block_size"]
    num_topk = config["num_topk"]
    block_window_size = config["block_window_size"]
    T = config["T"]

    # Calculate total blocks in key dimension
    k_blocks = (max_seqlen_k + block_size - 1) // block_size

    
    causal = False
    exact_streaming = False

    dtype = torch.float16
    
    # Print configuration
    print(f"batch_size: {batch_size}, seqlen_q: {seqlen_q}, max_seqlen_k: {max_seqlen_k}")
    print(f"nheads: {nheads}, nheads_k: {nheads_k}, headdim: {headdim}, block_size: {block_size}")
    print(f"num_topk: {num_topk} + block_window_size: {block_window_size}/{k_blocks} blocks ({(num_topk + block_window_size)/k_blocks:.4f} density), dtype: {dtype}, causal: {causal}, block_window_size: {block_window_size}")

    # Create test data
    device = "cuda:0"
    
    # Create query tensor
    q_full = torch.randn(batch_size * T, seqlen_q, nheads, headdim, device=device, dtype=dtype)
    
    # For simplicity, we're using a fixed sequence length for all elements in the batch
    seqlens_k = torch.full((batch_size,), max_seqlen_k - 1, dtype=torch.int32, device=device)
    
    # Create KV cache (simulating a context of max_seqlen_k - 1 tokens)
    kcache = torch.randn(batch_size * T, max_seqlen_k, nheads_k, headdim, device=device, dtype=dtype)
    vcache = torch.randn(batch_size * T, max_seqlen_k, nheads_k, headdim, device=device, dtype=dtype)
    
    # Create new keys and values (1 token per batch element)
    seqlen_knew = 1
    k_new = torch.randn(batch_size * T, seqlen_knew, nheads_k, headdim, device=device, dtype=dtype)
    v_new = torch.randn(batch_size * T, seqlen_knew, nheads_k, headdim, device=device, dtype=dtype)
    
    cache_batch_idx = torch.arange(batch_size, device=device, dtype=torch.int32)
    
    # Calculate fake sparsity to get the correct number of topk blocks
    # This is for backward compatibility with the existing implementation
    fake_sparsity = 1.0 - (num_topk / k_blocks)
    
    # Generate topk indices using the fake sparsity to get the exact number of blocks we want
    topk_idx = generate_batch_topk_indices(
        nheads_k, batch_size, seqlen_q, max_seqlen_k, num_topk, block_size, device, mode="random"
    )
    
    def bench_fwd(func, steps=T):
        """Benchmark forward pass."""
        warmup_steps = steps * 3
        test_steps = steps * 6
        for i in range(warmup_steps):
            out = func(i % T)
        torch.cuda.synchronize()
        st = time.time()
        for i in range(warmup_steps, warmup_steps + test_steps):
            out = func(i % T)
        torch.cuda.synchronize()
        ed = time.time()
        torch.cuda.empty_cache()
        return out, (ed - st) / test_steps

    # ---- Benchmark Block Sparse KV Cache ----
    print("\n--- Benchmarking Block Sparse KV Cache ---")
    
    def run_blocksparse_kvcache(i):
        return flash_attn_with_kvcache(
                q=q_full[i:i+1],
                k_cache=kcache[i:i+1],
                v_cache=vcache[i:i+1],
                topk_idx=topk_idx,
                k=k_new[i:i+1],
                v=v_new[i:i+1],
                cache_seqlens=seqlens_k,
                rotary_cos=None,
                rotary_sin=None,
                cache_batch_idx=cache_batch_idx,
                alibi_slopes=None,
                softmax_scale=None,
                causal=causal,
                block_window_size=block_window_size,
        )
    
    # Run once for compilation
    _ = run_blocksparse_kvcache(T-1)
    torch.cuda.empty_cache()
    
    # Benchmark
    _, bs_kvcache_time = bench_fwd(run_blocksparse_kvcache)
    bs_kvcache_time *= 1000  # Convert to ms
    print(f"Block Sparse KV Cache time: {bs_kvcache_time:.4f} ms")
    
    # ---- Benchmark Flash Attention KV Cache ----
    if HAS_FLASH_ATTN:
        
        print("\n--- Benchmarking Flash Attention KV Cache ---")
        
        # Clean up memory and delete variables only needed for block sparse attention
        gc.collect()
        torch.cuda.empty_cache()

        # Compute softmax scale
        softmax_scale = 1.0 / np.sqrt(headdim)
        
        def run_flash_kvcache(i):
            return flash_attn_with_kvcache(
                q=q_full[i:i+1],
                k_cache=kcache[i:i+1],
                v_cache=vcache[i:i+1],
                k=k_new[i:i+1],
                v=v_new[i:i+1],
                cache_seqlens=seqlens_k,
                rotary_cos=None,
                rotary_sin=None,
                cache_batch_idx=cache_batch_idx,
                alibi_slopes=None,
                softmax_scale=None,
                causal=causal,
                # num_splits=16
            )
        
        # Run once for compilation
        try:
            torch.cuda.empty_cache()
            _ = run_flash_kvcache(T-1)
            
            # Benchmark
            _, flash_kvcache_time = bench_fwd(run_flash_kvcache)
            flash_kvcache_time *= 1000  # Convert to ms
            print(f"Flash Attention KV Cache time: {flash_kvcache_time:.4f} ms")
            
            # ---- Benchmark Flash Attention with shortened KV Cache ----
            print("\n--- Benchmarking Flash Attention with shortened KV Cache ---")
            
            # Create shortened key and value caches based on num_topk * block_size
            shortened_len = (num_topk + block_window_size) * block_size
            k_short_cache = torch.randn(batch_size * T, shortened_len, nheads_k, headdim, device=device, dtype=dtype)
            v_short_cache = torch.randn(batch_size * T, shortened_len, nheads_k, headdim, device=device, dtype=dtype)
            seqlens_k_short = torch.full((batch_size,), shortened_len - 1, dtype=torch.int32, device=device)
            
            def run_flash_short_kvcache(i):
                return flash_attn_with_kvcache(
                    q=q_full[i:i+1],
                    k_cache=k_short_cache[i:i+1],
                    v_cache=v_short_cache[i:i+1],
                    k=k_new[i:i+1],
                    v=v_new[i:i+1],
                    cache_seqlens=seqlens_k_short,
                    rotary_cos=None,
                    rotary_sin=None,
                    cache_batch_idx=cache_batch_idx,
                    alibi_slopes=None,
                    softmax_scale=None,
                    causal=causal,
                )
            
            # Run once for compilation
            torch.cuda.empty_cache()
            _ = run_flash_short_kvcache(T-1)
            
            # Benchmark
            _, flash_short_kvcache_time = bench_fwd(run_flash_short_kvcache)
            flash_short_kvcache_time *= 1000  # Convert to ms
            print(f"Flash Attention (shortened KV Cache) time: {flash_short_kvcache_time:.4f} ms")
            
            # Compare speeds
            print(f"\n--- KV Cache Performance Comparison ---")
            print(f"Block Sparse KV Cache time: {bs_kvcache_time:.4f} ms")
            print(f"Flash Attention KV Cache time: {flash_kvcache_time:.4f} ms")
            print(f"Flash Attention (shortened KV Cache) time: {flash_short_kvcache_time:.4f} ms")
            
            # Block Sparse vs Flash
            flash_over_block = flash_kvcache_time / bs_kvcache_time
            block_over_flash = bs_kvcache_time / flash_kvcache_time
            print(f"\nTime Ratio (Flash/BlockSparse): {flash_over_block:.2f}x")
            
            if flash_over_block < 1.0:
                print(f"Flash Attention KV Cache is {block_over_flash:.2f}x faster than Block Sparse KV Cache")
            else:
                print(f"Block Sparse KV Cache is {1/block_over_flash:.2f}x faster than Flash Attention KV Cache")
            
            
            # Flash vs Flash Short
            flash_over_flash_short = flash_kvcache_time / flash_short_kvcache_time
            flash_short_over_flash = flash_short_kvcache_time / flash_kvcache_time
            print(f"\nTime Ratio (Flash/Flash-shortened): {flash_over_flash_short:.2f}x")
            
            if flash_over_flash_short < 1.0:
                print(f"Full Flash Attention KV Cache is {flash_short_over_flash:.2f}x faster than Shortened Flash Attention KV Cache")
            else:
                print(f"Shortened Flash Attention KV Cache is {flash_over_flash_short:.2f}x faster than Full Flash Attention KV Cache")
            
            # Block Sparse vs Flash Short
            flash_short_over_block = flash_short_kvcache_time / bs_kvcache_time
            block_over_flash_short = bs_kvcache_time / flash_short_kvcache_time
            print(f"\nTime Ratio (Flash-shortened/BlockSparse): {flash_short_over_block:.2f}x")
            
            if flash_short_over_block < 1.0:
                print(f"Shortened Flash Attention KV Cache is {block_over_flash_short:.2f}x faster than Block Sparse KV Cache")
            else:
                print(f"Block Sparse KV Cache is {1/block_over_flash_short:.2f}x faster than Shortened Flash Attention KV Cache")
                
        except Exception as e:
            print(f"Error running Flash Attention KV Cache: {e}")
            print("Skipping performance comparison")
    
 