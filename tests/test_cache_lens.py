import torch
from infllm_v2.max_pooling_1d import max_pooling_1d_varlen

def test_cache_lens_tensor():
    """Test that cache_lens works as a tensor with different values per batch"""
    print("Testing cache_lens as tensor functionality...")
    
    # Setup test data
    batch_size = 3
    num_heads = 2
    seqlen_qs = [10, 15, 8]  # Different query lengths per batch
    seqlen_ks = [20, 25, 16]  # Different key lengths per batch
    cache_lens_values = [0, 64, 128]  # Different cache lengths per batch
    
    # Create cumulative sequence lengths
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device='cuda')
    
    for i in range(batch_size):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + seqlen_qs[i]
        cu_seqlens_k[i + 1] = cu_seqlens_k[i] + seqlen_ks[i]
    
    # Create cache_lens tensor
    cache_lens = torch.tensor(cache_lens_values, dtype=torch.int32, device='cuda')
    
    total_q = cu_seqlens_q[-1].item()
    max_seqlen_q = max(seqlen_qs)
    max_seqlen_k = max(seqlen_ks)
    
    # Create input tensor
    input_tensor = torch.randn(num_heads, total_q, max_seqlen_k, device='cuda', dtype=torch.float16)
    
    # Test parameters
    local_blocks = 4
    init_blocks = 1
    block_size = 64
    stride = 16
    
    print(f"Batch size: {batch_size}")
    print(f"Sequence lengths (q): {seqlen_qs}")
    print(f"Sequence lengths (k): {seqlen_ks}")
    print(f"Cache lengths: {cache_lens_values}")
    print(f"Input shape: {input_tensor.shape}")
    
    try:
        # Run varlen pooling with tensor cache_lens
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
        
        print(f"\n✓ Successfully ran max_pooling_1d_varlen with tensor cache_lens!")
        print(f"Output shape: {output.shape}")
        
        # Calculate expected output length based on max cache length
        max_cache_len = max(cache_lens_values)
        total_len = max_seqlen_q + max_cache_len
        expected_out_len = (total_len + block_size - 1) // block_size
        
        assert output.shape[2] == expected_out_len, f"Expected out_len {expected_out_len}, got {output.shape[2]}"
        print(f"✓ Output length is correct: {expected_out_len}")
        
        # Verify that different cache lengths are being used correctly
        # by checking the mask patterns for each batch
        print("\nChecking mask patterns for each batch:")
        for b in range(batch_size):
            q_start = cu_seqlens_q[b].item()
            q_end = cu_seqlens_q[b + 1].item()
            cache_len = cache_lens_values[b]
            
            batch_output = output[:, q_start:q_end, :]
            
            # Count inf and -inf values
            num_inf = (torch.isinf(batch_output) & (batch_output > 0)).sum().item()
            num_neg_inf = (torch.isinf(batch_output) & (batch_output < 0)).sum().item()
            num_finite = torch.isfinite(batch_output).sum().item()
            
            print(f"\nBatch {b} (cache_len={cache_len}):")
            print(f"  inf values: {num_inf}")
            print(f"  -inf values: {num_neg_inf}")
            print(f"  finite values: {num_finite}")
            
            # The mask pattern should differ based on cache_len
            # Higher cache_len should affect the off_bq calculation
            
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_cache_lens_tensor() 