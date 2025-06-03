import torch
import time
from infllm_v2 import blockmask_to_uint64 as cuda_blockmask_to_uint64

def compare_blockmasks(mask1, mask2, name1, name2):
    """Compare two blockmasks using cuda_blockmask_to_uint64"""
    print(f"\nComparing {name1} and {name2}")
    print(f"Shape {name1}: {mask1.shape}")
    print(f"Shape {name2}: {mask2.shape}")
    
    # Move masks to GPU
    mask1_gpu = mask1.cuda()
    mask2_gpu = mask2.cuda()
    
    # Apply CUDA implementation
    print("Running cuda_blockmask_to_uint64 on both masks...")
    
    # Process first mask
    start_time = time.time()
    result1, last_dim1 = cuda_blockmask_to_uint64(mask1_gpu)
    torch.cuda.synchronize()
    time1 = time.time() - start_time
    
    # Process second mask
    start_time = time.time()
    result2, last_dim2 = cuda_blockmask_to_uint64(mask2_gpu)
    torch.cuda.synchronize()
    time2 = time.time() - start_time
    
    print(f"Output shape {name1}: {result1.shape}, Last dim size: {last_dim1}")
    print(f"Output shape {name2}: {result2.shape}, Last dim size: {last_dim2}")
    print(f"Processing time {name1}: {time1*1000:.3f}ms")
    print(f"Processing time {name2}: {time2*1000:.3f}ms")
    
    # Check if the entire tensors match
    if result1.shape == result2.shape:
        result1_cpu = result1.cpu()
        result2_cpu = result2.cpu()
        full_match = torch.all(result1_cpu == result2_cpu).item()
        print(f"\nFull tensor comparison - Complete match: {full_match}")
        
        if not full_match:
            # Find total number of mismatches
            mismatches = (result1_cpu != result2_cpu).sum().item()
            total_elements = result1_cpu.numel()
            mismatch_percentage = (mismatches / total_elements) * 100
            print(f"Found {mismatches} mismatches out of {total_elements} elements ({mismatch_percentage:.4f}%)")
    else:
        print(f"\nFull tensor comparison - Shapes don't match: {result1.shape} vs {result2.shape}")
    
    # Get the common shape dimensions for comparison
    if len(mask1.shape) >= 2 and len(mask2.shape) >= 2:
        # Assuming the shapes have the format [..., head, seq]
        print("\nComparing results for same head and sequence positions:")
        
        # Find maximum comparable dimensions
        max_heads = min(mask1.shape[-2], mask2.shape[-2])
        
        for head_idx in range(max_heads):
            # Extract head results
            head_result1 = result1[..., head_idx, :]
            head_result2 = result2[..., head_idx, :]
            
            # Move results to CPU for easier comparison
            head_result1_cpu = head_result1.cpu()
            head_result2_cpu = head_result2.cpu()
            
            # Check if shapes are compatible for direct comparison
            if head_result1_cpu.shape == head_result2_cpu.shape:
                is_equal = torch.all(head_result1_cpu == head_result2_cpu).item()
                #print(f"Head {head_idx}: Equal = {is_equal}")
                
                if not is_equal:
                    # Find mismatches
                    mismatch = (head_result1_cpu != head_result2_cpu).nonzero(as_tuple=True)
                    if len(mismatch[0]) > 0:
                        print(f"Found {len(mismatch[0])} mismatches for head {head_idx}")
                        for i in range(min(5, len(mismatch[0]))):
                            idx = tuple(tensor[i].item() for tensor in mismatch)
                            print(f"  Mismatch at {idx}: {name1}={head_result1_cpu[idx].item()}, {name2}={head_result2_cpu[idx].item()}")
            else:
                print(f"Head {head_idx}: Shapes not compatible for direct comparison - {head_result1_cpu.shape} vs {head_result2_cpu.shape}")
    else:
        print("Masks don't have enough dimensions for head/sequence comparison")

def load_and_compare_tensor(tp1_dir, tp2_rank0_dir, tp2_rank1_dir, filename):
    """Load and compare a tensor file from all directories"""
    tp1_path = f"{tp1_dir}/{filename}"
    tp2_rank0_path = f"{tp2_rank0_dir}/{filename}"
    tp2_rank1_path = f"{tp2_rank1_dir}/{filename}"
    
    print(f"\nLoading and comparing {filename}...")
    
    try:
        tp1_tensor = torch.load(tp1_path)
        print(f"TP1 {filename}: {tp1_tensor}")
        if hasattr(tp1_tensor, 'shape'):
            print(f"TP1 {filename} shape: {tp1_tensor.shape}")
    except Exception as e:
        print(f"Error loading TP1 {filename}: {e}")
        tp1_tensor = None
    
    try:
        tp2_rank0_tensor = torch.load(tp2_rank0_path)
        print(f"TP2 (rank 0) {filename}: {tp2_rank0_tensor}")
        if hasattr(tp2_rank0_tensor, 'shape'):
            print(f"TP2 (rank 0) {filename} shape: {tp2_rank0_tensor.shape}")
    except Exception as e:
        print(f"Error loading TP2 (rank 0) {filename}: {e}")
        tp2_rank0_tensor = None
    
    try:
        tp2_rank1_tensor = torch.load(tp2_rank1_path)
        print(f"TP2 (rank 1) {filename}: {tp2_rank1_tensor}")
        if hasattr(tp2_rank1_tensor, 'shape'):
            print(f"TP2 (rank 1) {filename} shape: {tp2_rank1_tensor.shape}")
    except Exception as e:
        print(f"Error loading TP2 (rank 1) {filename}: {e}")
        tp2_rank1_tensor = None
    
    # Compare scalars
    if tp1_tensor is not None and tp2_rank0_tensor is not None and tp2_rank1_tensor is not None:
        if not hasattr(tp1_tensor, 'shape'):
            print(f"TP1 vs TP2 rank 0: {tp1_tensor == tp2_rank0_tensor}")
            print(f"TP1 vs TP2 rank 1: {tp1_tensor == tp2_rank1_tensor}")
            return tp1_tensor, tp2_rank0_tensor, tp2_rank1_tensor
    
    return tp1_tensor, tp2_rank0_tensor, tp2_rank1_tensor

def main():
    # Define directories
    tp1_dir = "/home/qiqi/downloads/debug/tp1"
    tp2_rank0_dir = "/home/qiqi/downloads/debug/tp2_rank0"
    tp2_rank1_dir = "/home/qiqi/downloads/debug/tp2_rank1"
    
    # Load the blockmask files
    print("Loading blockmask files...")
    
    try:
        mask_tp1 = torch.load(f"{tp1_dir}/fwd_blockmask_bool.pt")
        print("Successfully loaded TP1 mask")
        print(f"TP1 mask shape: {mask_tp1.shape}")
    except Exception as e:
        print(f"Error loading TP1 mask: {e}")
        return
    
    try:
        mask_tp2_rank0 = torch.load(f"{tp2_rank0_dir}/fwd_blockmask_bool.pt")
        print("Successfully loaded TP2 rank 0 mask")
        print(f"TP2 rank 0 mask shape: {mask_tp2_rank0.shape}")
        print(f"TP2 rank 0 mask device: {mask_tp2_rank0.device}")
    except Exception as e:
        print(f"Error loading TP2 rank 0 mask: {e}")
        return
    
    try:
        mask_tp2_rank1 = torch.load(f"{tp2_rank1_dir}/fwd_blockmask_bool.pt")
        print("Successfully loaded TP2 rank 1 mask")
        print(f"TP2 rank 1 mask shape: {mask_tp2_rank1.shape}")
        print(f"TP2 rank 1 mask device: {mask_tp2_rank1.device}")
    except Exception as e:
        print(f"Error loading TP2 rank 1 mask: {e}")
        return
    
    # Load and compare additional context variables
    max_seqlen_k_tp1, max_seqlen_k_tp2_rank0, max_seqlen_k_tp2_rank1 = load_and_compare_tensor(
        tp1_dir, tp2_rank0_dir, tp2_rank1_dir, "ctx.max_seqlen_k_.pt")
    
    n_block_dim_tp1, n_block_dim_tp2_rank0, n_block_dim_tp2_rank1 = load_and_compare_tensor(
        tp1_dir, tp2_rank0_dir, tp2_rank1_dir, "ctx.n_block_dim.pt")
    
    # Move tensors to the same device before concatenation
    target_device = "cuda:0"
    mask_tp2_rank0 = mask_tp2_rank0.to(target_device)
    mask_tp2_rank1 = mask_tp2_rank1.to(target_device)
    
    # Concatenate TP2 rank 0 and rank 1 masks along dimension 0
    print(f"\nConcatenating TP2 rank 0 and rank 1 masks along dimension 0 (on {target_device})...")
    
    mask_tp2_combined = torch.cat([mask_tp2_rank0, mask_tp2_rank1], dim=0)
    
    print(f"Original TP1 mask shape: {mask_tp1.shape}")
    print(f"TP2 rank 0 mask shape: {mask_tp2_rank0.shape}")
    print(f"TP2 rank 1 mask shape: {mask_tp2_rank1.shape}")
    print(f"Combined TP2 mask shape: {mask_tp2_combined.shape}")
    
    # Compare the TP1 mask with combined TP2 mask
    compare_blockmasks(mask_tp1, mask_tp2_combined, "TP1", "TP2 (combined)")

if __name__ == "__main__":
    main() 