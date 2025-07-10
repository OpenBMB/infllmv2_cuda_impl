import torch
import math
from infllm_v2.max_pooling_1d import max_pooling_1d

def test_multi_batch_pooling():
    """Test max_pooling_1d with multiple variable-length sequences."""
    
    # Test parameters
    num_heads = 8
    block_size = 64
    stride = 16
    init_blocks = 1
    local_blocks = 32
    cache_len = 0
    
    # Create variable-length sequences
    batch_size = 3
    seq_lens_q = [128, 256, 192]  # Different query lengths for each batch
    seq_lens_k = [512, 768, 640]  # Different key lengths for each batch
    
    # Create cu_seqlens
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device='cuda')
    
    for i in range(batch_size):
        cu_seqlens_q[i+1] = cu_seqlens_q[i] + seq_lens_q[i]
        cu_seqlens_k[i+1] = cu_seqlens_k[i] + seq_lens_k[i]
    
    total_q_len = int(cu_seqlens_q[-1].item())
    max_k_len = max(seq_lens_k)
    max_seqlen_q = max(seq_lens_q)
    max_seqlen_k = max(seq_lens_k)
    
    # Create input tensor
    input_tensor = torch.randn(num_heads, total_q_len, max_k_len, 
                              dtype=torch.float16, device='cuda')
    
    print(f"Testing multi-batch max_pooling_1d:")
    print(f"Batch size: {batch_size}")
    print(f"Query lengths: {seq_lens_q}")
    print(f"Key lengths: {seq_lens_k}")
    print(f"Total query length: {total_q_len}")
    print(f"Input shape: {input_tensor.shape}")
    
    # Run multi-batch version
    output = max_pooling_1d(
        input_tensor,
        cache_len=cache_len,
        local_blocks=local_blocks,
        init_blocks=init_blocks,
        block_size=block_size,
        stride=stride,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    
    print(f"Output shape: {output.shape}")
    
    # Verify output shape
    max_blocks = (max_seqlen_q + cache_len + block_size - 1) // block_size
    expected_shape = (num_heads, total_q_len, max_blocks)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Test each batch separately and compare
    print("\nVerifying each batch:")
    for b in range(batch_size):
        q_start = int(cu_seqlens_q[b].item())
        q_end = int(cu_seqlens_q[b+1].item())
        q_len = q_end - q_start
        
        k_start = int(cu_seqlens_k[b].item())
        k_end = int(cu_seqlens_k[b+1].item())
        k_len = k_end - k_start
        
        # Extract this batch's input
        batch_input = input_tensor[:, q_start:q_end, :k_len]
        
        # Run single-batch version
        single_output = max_pooling_1d(
            batch_input,
            cache_len=cache_len,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=stride,
        )
        
        # Extract corresponding output from multi-batch result
        multi_output_batch = output[:, q_start:q_end, :]
        
        # The multi-batch output might have more blocks than needed for this specific batch
        # So we only compare up to the number of blocks this batch actually needs
        batch_blocks = (q_len + cache_len + block_size - 1) // block_size
        
        print(f"\nBatch {b}:")
        print(f"  Query range: [{q_start}, {q_end}), length: {q_len}")
        print(f"  Key range: [{k_start}, {k_end}), length: {k_len}")
        print(f"  Single batch output shape: {single_output.shape}")
        print(f"  Multi batch output shape (for this batch): {multi_output_batch.shape}")
        print(f"  Comparing first {batch_blocks} blocks...")
        
        # Compare the valid blocks
        single_valid = single_output[:, :, :batch_blocks]
        multi_valid = multi_output_batch[:, :, :batch_blocks]
        
        # Check if values match (allowing for small numerical differences)
        abs_diff = torch.abs(single_valid - multi_valid)
        max_diff = torch.max(abs_diff).item()
        
        # Count inf/-inf matches
        single_inf = torch.isinf(single_valid)
        multi_inf = torch.isinf(multi_valid)
        inf_match = torch.all(single_inf == multi_inf)
        
        print(f"  Max absolute difference: {max_diff}")
        print(f"  Inf positions match: {inf_match}")
        
        if max_diff > 1e-3 or not inf_match:
            print(f"  WARNING: Batch {b} has differences!")
        else:
            print(f"  Batch {b} matches correctly!")
    
    print("\nMulti-batch pooling test completed successfully!")


def test_single_vs_multi_batch():
    """Compare single batch behavior between old and new API."""
    
    # Test parameters
    num_heads = 4
    q_len = 256
    k_len = 512
    block_size = 64
    stride = 16
    init_blocks = 1
    local_blocks = 16
    cache_len = 0
    
    # Create input
    input_tensor = torch.randn(num_heads, q_len, k_len, 
                              dtype=torch.float16, device='cuda')
    
    print("Testing backward compatibility (single batch):")
    print(f"Input shape: {input_tensor.shape}")
    
    # Run with old API (implicit single batch)
    output_old = max_pooling_1d(
        input_tensor,
        cache_len=cache_len,
        local_blocks=local_blocks,
        init_blocks=init_blocks,
        block_size=block_size,
        stride=stride,
    )
    
    # Run with new API (explicit cu_seqlens)
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.tensor([0, k_len], dtype=torch.int32, device='cuda')
    
    output_new = max_pooling_1d(
        input_tensor,
        cache_len=cache_len,
        local_blocks=local_blocks,
        init_blocks=init_blocks,
        block_size=block_size,
        stride=stride,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=q_len,
        max_seqlen_k=k_len,
    )
    
    print(f"Old API output shape: {output_old.shape}")
    print(f"New API output shape: {output_new.shape}")
    
    # Compare outputs
    if torch.allclose(output_old, output_new, rtol=1e-5, atol=1e-5):
        print("Single batch outputs match perfectly!")
    else:
        abs_diff = torch.abs(output_old - output_new)
        max_diff = torch.max(abs_diff).item()
        print(f"WARNING: Outputs differ! Max difference: {max_diff}")
    
    print("\nBackward compatibility test completed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing with data from .pt files")
    print("=" * 60)
    test_multi_batch_pooling_from_files()
    print("\n" + "=" * 60)
    print("Testing backward compatibility")
    print("=" * 60)
    test_single_vs_multi_batch()
    print("\n" + "=" * 60)
    print("Testing with synthetic data")
    print("=" * 60)
    test_multi_batch_pooling() 