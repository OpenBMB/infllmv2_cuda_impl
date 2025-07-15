#!/usr/bin/env python3
"""
Test script to verify topk_to_uint64 works correctly in multi-GPU scenarios
"""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from infllm_v2.topk_to_uint64 import topk_to_uint64

def test_single_gpu():
    """Test on single GPU to establish baseline"""
    print("Testing single GPU...")
    device = torch.device("cuda:0")
    
    # Create test data
    batch_size = 2
    num_heads = 4
    total_seqlen = 128
    k = 32
    max_seqlen_k = 1024
    block_size = 64
    
    # Generate random topk indices
    torch.manual_seed(42)
    topk_idx = torch.randint(0, max_seqlen_k // block_size, 
                            (batch_size, num_heads, total_seqlen, k), 
                            dtype=torch.int32, device=device)
    
    # Convert to uint64
    result, k_blocks = topk_to_uint64(topk_idx, max_seqlen_k, block_size)
    
    print(f"Single GPU result shape: {result.shape}")
    print(f"Single GPU k_blocks: {k_blocks}")
    print(f"Single GPU result device: {result.device}")
    
    return result

def test_multi_gpu():
    """Test on multiple GPUs"""
    if torch.cuda.device_count() < 2:
        print("Skipping multi-GPU test - only one GPU available")
        return None
        
    print("\nTesting multi-GPU...")
    results = []
    
    for gpu_id in range(min(2, torch.cuda.device_count())):
        device = torch.device(f"cuda:{gpu_id}")
        print(f"\nTesting on GPU {gpu_id}...")
        
        # Create test data on specific GPU
        batch_size = 2
        num_heads = 4
        total_seqlen = 128
        k = 32
        max_seqlen_k = 1024
        block_size = 64
        
        # Generate the same random topk indices (same seed)
        torch.manual_seed(42)
        topk_idx = torch.randint(0, max_seqlen_k // block_size, 
                                (batch_size, num_heads, total_seqlen, k), 
                                dtype=torch.int32, device=device)
        
        # Convert to uint64
        result, k_blocks = topk_to_uint64(topk_idx, max_seqlen_k, block_size)
        
        print(f"GPU {gpu_id} result shape: {result.shape}")
        print(f"GPU {gpu_id} k_blocks: {k_blocks}")
        print(f"GPU {gpu_id} result device: {result.device}")
        
        # Move to CPU for comparison
        results.append(result.cpu())
    
    # Compare results from different GPUs
    if len(results) == 2:
        if torch.allclose(results[0], results[1]):
            print("\n✓ Multi-GPU test PASSED: Results are consistent across GPUs")
        else:
            print("\n✗ Multi-GPU test FAILED: Results differ across GPUs")
            print(f"Max difference: {(results[0] - results[1]).abs().max()}")
    
    return results[0] if results else None

def test_cross_device_transfer():
    """Test transferring tensors between devices"""
    if torch.cuda.device_count() < 2:
        print("Skipping cross-device test - only one GPU available")
        return
        
    print("\nTesting cross-device transfer...")
    
    # Create data on GPU 0
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    
    batch_size = 2
    num_heads = 4
    total_seqlen = 128
    k = 32
    max_seqlen_k = 1024
    block_size = 64
    
    torch.manual_seed(42)
    topk_idx = torch.randint(0, max_seqlen_k // block_size, 
                            (batch_size, num_heads, total_seqlen, k), 
                            dtype=torch.int32, device=device0)
    
    # Test 1: Process on original device
    result0, _ = topk_to_uint64(topk_idx, max_seqlen_k, block_size)
    
    # Test 2: Transfer to GPU 1 and process
    topk_idx_gpu1 = topk_idx.to(device1)
    result1, _ = topk_to_uint64(topk_idx_gpu1, max_seqlen_k, block_size)
    
    # Compare results
    if torch.allclose(result0.cpu(), result1.cpu()):
        print("✓ Cross-device test PASSED: Results consistent after device transfer")
    else:
        print("✗ Cross-device test FAILED: Results differ after device transfer")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing topk_to_uint64 multi-GPU consistency")
    print("=" * 60)
    
    # Test single GPU
    single_gpu_result = test_single_gpu()
    
    # Test multi GPU
    multi_gpu_result = test_multi_gpu()
    
    # Compare single vs multi GPU results
    if single_gpu_result is not None and multi_gpu_result is not None:
        if torch.allclose(single_gpu_result.cpu(), multi_gpu_result.cpu()):
            print("\n✓ Single vs Multi GPU test PASSED: Results are consistent")
        else:
            print("\n✗ Single vs Multi GPU test FAILED: Results differ")
    
    # Test cross-device transfer
    test_cross_device_transfer()
    
    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)

if __name__ == "__main__":
    main() 