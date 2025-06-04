# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
# ruff: noqa
import torch
import time
import gc  # Import garbage collector
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import generate_topk_indices, print_topk_idx, move_sliding_to_topk_dix

if __name__ == "__main__":
    config = {
        "B": 1,                 # batch size
        "SEQ_LEN": 131072,      # sequence length
        "H": 2,                 # KV attention heads
        "HQ": 32,               # query attention heads
        "D": 128,               # dimension per head
        "S": 32,                # number of blocks to select per query
        "block_size": 64,       # block size
        "dtype": torch.float16, # data type
        "scale": 0.1,           # scale factor
        "window_size": 32*64,   # window size
        "use_checkpoint": False,  # whether to use gradient checkpoint
    }
    use_move_sliding_to_topk_dix = False

    # Then these parameters can be accessed as follows
    B = config["B"]
    SEQ_LEN = config["SEQ_LEN"]
    H = config["H"]
    HQ = config["HQ"]
    D = config["D"]
    S = config["S"]
    block_size = config["block_size"]
    dtype = config["dtype"]
    scale = config["scale"]
    window_size = config["window_size"]
    use_checkpoint = config["use_checkpoint"]

    # Define sparsity level - controls how many blocks are selected (1.0 - S*block_size/SEQ_LEN)
    sparsity = 1.0 - (S * block_size / SEQ_LEN)
    print(f"batch: {B}, seq_len: {SEQ_LEN}, heads: {H}, head_kv: {HQ}, dim: {D}, selected_blocks: {S}, block_size: {block_size}, sparsity: {sparsity:.4f}, dtype: {dtype}, scale: {scale}, window_size: {window_size}, use_checkpoint: {use_checkpoint}")
    # Import Flash Attention
    from flash_attn import flash_attn_func
    # Import Block Sparse Attention
    from flash_attn import flash_attn_varlen_func

    def bench_fwd(func, warmup_steps=2, test_steps=5):
        for i in range(warmup_steps):
            func()
        torch.cuda.synchronize()
        st = time.time()
        for i in range(test_steps):
            func()
        torch.cuda.synchronize()
        ed = time.time()
        return (ed - st) / test_steps

    # print(f"已分配内存: {torch.cuda.memory_stats()['allocated_bytes.all.current'] / 1024**2:.2f} MB")

    torch.random.manual_seed(0)
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device='cuda').requires_grad_(False)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device='cuda').requires_grad_(False)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device='cuda').requires_grad_(False)

    print(f"Initializing inputs, allocated memory: {torch.cuda.memory_stats()['allocated_bytes.all.current'] / 1024**2:.2f} MB")
    # Prepare Flash Attention inputs - reshape to expected format (B, NH, S, D)
    q_fa = Q.contiguous()
    k_fa = K.contiguous()
    v_fa = V.contiguous()

    # print(f"Compiling Flash Attention..., allocated memory: {torch.cuda.memory_stats()['allocated_bytes.all.current'] / 1024**2:.2f} MB")
    # Run Flash Attention once for compilation
    with torch.no_grad():
        flash_attn_func(q_fa, k_fa, v_fa, causal=True, softmax_scale=scale)
    torch.cuda.empty_cache()
    # print(f"Flash Attention compilation complete, allocated memory: {torch.cuda.memory_stats()['allocated_bytes.all.current'] / 1024**2:.2f} MB")

    # Define the benchmark functions
    def run_fa():
        return flash_attn_func(q_fa, k_fa, v_fa, causal=True, softmax_scale=scale)
    
    def zero_fa_grads():
        Q.grad = None
        K.grad = None
        V.grad = None

    # Benchmark Flash Attention
    print("--- Benchmarking Flash Attention ---")
    with torch.no_grad():
        fa_fwd_time = bench_fwd(run_fa)
    del k_fa, v_fa
    torch.cuda.empty_cache()
    print(f"Flash Attention forward pass, allocated memory: {torch.cuda.memory_stats()['allocated_bytes.all.current'] / 1024**2:.2f} MB")
    
    # Convert to ms
    fa_fwd_time *= 1000

    # Print results
    print(f"Flash Attention forward time: {fa_fwd_time:.4f} ms")

    # Benchmark Flash Attention with shortened key length (S*block_size)
    shortened_len = S * block_size + window_size
    # Create shortened key and value tensors
    k_short = torch.randn((B, shortened_len, H, D), dtype=dtype, device='cuda').requires_grad_(False)
    v_short = torch.randn((B, shortened_len, H, D), dtype=dtype, device='cuda').requires_grad_(False)

    # Prepare for Flash Attention
    k_short_fa = k_short.contiguous()  # [B, H, shortened_len, D]
    v_short_fa = v_short.contiguous()  # [B, H, shortened_len, D]

    # print(f"Compiling Flash Attention..., allocated memory: {torch.cuda.memory_stats()['allocated_bytes.all.current'] / 1024**2:.2f} MB")
    # Run once for compilation
    with torch.no_grad():
        flash_attn_func(q_fa, k_short_fa, v_short_fa, causal=False, softmax_scale=scale)
    torch.cuda.empty_cache()
    # print(f"Flash Attention compilation complete, allocated memory: {torch.cuda.memory_stats()['allocated_bytes.all.current'] / 1024**2:.2f} MB")

    def run_fa_short():
        return flash_attn_func(q_fa, k_short_fa, v_short_fa, causal=False, softmax_scale=scale)

    # Benchmark
    print(f"--- Benchmarking Flash Attention with shortened key length ({shortened_len}) ---")
    with torch.no_grad():
        fa_short_fwd_time = bench_fwd(run_fa_short)
    
    # Convert to ms
    fa_short_fwd_time *= 1000

    # Print comparison
    print(f"Flash Attention (shortened keys) forward time: {fa_short_fwd_time:.4f} ms")
    print(f"Speedup (FA-full/FA-short) forward: {fa_fwd_time/fa_short_fwd_time:.2f}x")

    # Run once for compilation
    with torch.no_grad():
        flash_attn_func(q_fa, k_short_fa, v_short_fa, causal=True, softmax_scale=scale)
    torch.cuda.empty_cache()

    def run_fa_short_causal():
        return flash_attn_func(q_fa, k_short_fa, v_short_fa, causal=True, softmax_scale=scale)

    # Benchmark
    print(f"--- Benchmarking Flash Attention with shortened key length ({shortened_len}) ---")
    with torch.no_grad():
        fa_short_causal_fwd_time = bench_fwd(run_fa_short_causal)
    del q_fa, k_short_fa, v_short_fa
    torch.cuda.empty_cache()
    print(f"Flash Attention shortened keys causal forward pass, allocated memory: {torch.cuda.memory_stats()['allocated_bytes.all.current'] / 1024**2:.2f} MB")
    
    # Convert to ms
    fa_short_causal_fwd_time *= 1000

    # Print comparison
    print(f"Flash Attention (shortened keys causal) forward time: {fa_short_causal_fwd_time:.4f} ms")
    print(f"Speedup (FA-full/FA-short-causal) forward: {fa_fwd_time/fa_short_causal_fwd_time:.2f}x")

    # Free memory before verification
    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark Block Sparse Attention
    print(f"\n--- Benchmarking Block Sparse Attention ---")

    # Calculate sparsity level and print
    print(f"Block Sparse Attention with sparsity={sparsity:.4f}")

    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device='cuda').requires_grad_(False)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device='cuda').requires_grad_(False)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device='cuda').requires_grad_(False)

    # Create unpadded tensors (no actual padding)
    q_unpad = Q.view(-1, HQ, D)  # [B*SEQ_LEN*HQ, D]
    k_unpad = K.view(-1, H, D)  # [B*SEQ_LEN*H, D]
    v_unpad = V.view(-1, H, D)  # [B*SEQ_LEN*H, D]
    torch.cuda.empty_cache()
    print(f"BSA input initialization, allocated memory: {torch.cuda.memory_stats()['allocated_bytes.all.current'] / 1024**2:.2f} MB")

    # Create cu_seqlens (cumulative sequence lengths)
    cu_seqlens_q = torch.arange(0, (B+1)*SEQ_LEN, SEQ_LEN, dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.arange(0, (B+1)*SEQ_LEN, SEQ_LEN, dtype=torch.int32, device='cuda')

    # Setup other required parameters
    head_mask_type = torch.tensor([1] * H, device='cuda', dtype=torch.int32)
    streaming_info = torch.tensor([0, 0] * H, device='cuda', dtype=torch.int32)

    # Generate topk indices for block sparse attention
    total_seq_len = B * SEQ_LEN
    max_seqlen_q = SEQ_LEN
    max_seqlen_k = SEQ_LEN
    
    # Generate topk indices - each query position gets S random blocks to attend to
    topk_idx = generate_topk_indices(H, total_seq_len, max_seqlen_k, sparsity, block_size, window_size, 'cuda')
    if use_move_sliding_to_topk_dix:
        topk_idx, S, window_size = move_sliding_to_topk_dix(topk_idx, H, total_seq_len, S, block_size, window_size, 'cuda')
    # print_topk_idx(topk_idx, block_size)
    print(f"topk_idx: {topk_idx.shape}, S: {S}, window_size: {window_size}")
    print(f"BSA generating topk indices, allocated memory: {torch.cuda.memory_stats()['allocated_bytes.all.current'] / 1024**2:.2f} MB")
    
    # Verify actual sparsity
    num_blocks_k = (max_seqlen_k + block_size - 1) // block_size
    actual_blocks_selected = topk_idx.size(-1)
    real_sparsity = 1.0 - (actual_blocks_selected / num_blocks_k)
    print(f"Number of key blocks: {num_blocks_k}")
    print(f"Blocks selected per query: {actual_blocks_selected}")
    print(f"Actual block sparsity: {real_sparsity:.4f}")

    dropout_p = 0.0

    # print(f"Compiling BSA..., allocated memory: {torch.cuda.memory_stats()['allocated_bytes.all.current'] / 1024**2:.2f} MB")
    # Run once for compilation
    flash_attn_varlen_func(
        q_unpad, k_unpad, v_unpad,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p,
        causal=True,
        block_window_size=window_size//block_size,
        topk_idx=topk_idx,  # Using topk_idx instead of base_blockmask
    )
    torch.cuda.empty_cache()
    # print(f"BSA compilation complete, allocated memory: {torch.cuda.memory_stats()['allocated_bytes.all.current'] / 1024**2:.2f} MB")

    def run_bsa():
        return flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p,
            causal=True,
            block_window_size=window_size//block_size,
            topk_idx=topk_idx,
        )

    # Benchmark
    bsa_fwd_time = bench_fwd(run_bsa)
    del q_unpad, k_unpad, v_unpad
    torch.cuda.empty_cache()
    print(f"BSA forward pass, allocated memory: {torch.cuda.memory_stats()['allocated_bytes.all.current'] / 1024**2:.2f} MB")
    # Convert to ms
    bsa_fwd_time *= 1000

    # Final comparison section
    print(f"\n--- Final Performance Comparison ---")
    
    # Forward pass comparisons
    print(f"\nForward Pass:")
    print(f"Flash Attention forward time: {fa_fwd_time:.4f} ms")
    print(f"Flash Attention (shortened keys) forward time: {fa_short_fwd_time:.4f} ms")
    print(f"Flash Attention (shortened keys causal) forward time: {fa_short_causal_fwd_time:.4f} ms")
    print(f"Block Sparse Attention forward time: {bsa_fwd_time:.4f} ms")
    print(f"Speedup (FA/BSA) forward: {fa_fwd_time/bsa_fwd_time:.2f}x")
    print(f"Speedup (FA-short/BSA) forward: {fa_short_fwd_time/bsa_fwd_time:.2f}x")
    print(f"peak memory: {torch.cuda.max_memory_allocated() / 1e9} GB")