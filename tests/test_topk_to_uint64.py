import torch
import time
import numpy as np
from infllm_v2 import topk_to_uint64
from infllm_v2 import blockmask_to_uint64

def convert_topk_to_base_blockmask(
    topk_idx: torch.Tensor,
    max_seqlen_k: int,
    block_size: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    将topk索引转换为块稀疏注意力掩码，仅处理-1的情况
    
    Args:
        topk_idx: 形状 [num_heads, total_seqlen, k] 的块索引张量
        max_seqlen_k: 最大键序列长度（用于计算键块数量）
        block_size: block_size
        device: 输出设备
    
    Returns:
        mask: 布尔掩码，形状 [num_heads, total_seqlen, k_blocks]
    """
    # 计算键块数量
    k_blocks = (max_seqlen_k + block_size - 1) // block_size  # 向上取整
    num_heads, total_seqlen, k = topk_idx.shape

    # 初始化全False掩码
    mask = torch.zeros(num_heads, total_seqlen, k_blocks, 
                       dtype=torch.bool, device=device)

    # 过滤掉 -1，确保索引合法
    valid_idx = topk_idx[topk_idx != -1]

    # 生成索引掩码
    batch_idx, seq_idx, _ = torch.where(topk_idx != -1)  # 找到非-1索引的 (head, seq) 位置
    mask[batch_idx, seq_idx, valid_idx] = True  # 设置对应位置为 True

    return mask

def topk_to_uint64_numpy(
    topk_idx: np.ndarray,
    max_seqlen_k: int,
    block_size: int
) -> tuple:
    """
    NumPy implementation of topk_to_uint64
    
    Args:
        topk_idx: NumPy array of shape [num_heads, total_seqlen, k] or [batch, num_heads, seq_len, k]
        max_seqlen_k: Maximum key sequence length
        block_size: Block size
        
    Returns:
        uint64_array: NumPy array of uint64 values
        last_dim: Size of the last dimension
    """
    # Check input dimensions
    is_4d = len(topk_idx.shape) == 4
    
    if is_4d:
        batch, num_heads, seq_len, k = topk_idx.shape
        # Reshape to 3D for processing
        topk_idx_3d = topk_idx.reshape(batch * num_heads, seq_len, k)
    else:
        num_heads, seq_len, k = topk_idx.shape
        topk_idx_3d = topk_idx
    
    # Calculate number of blocks
    k_blocks = (max_seqlen_k + block_size - 1) // block_size
    
    # Calculate how many uint64 values we need per row
    uint64_bits = 64
    last_dim = (k_blocks + uint64_bits - 1) // uint64_bits
    
    # Create output array
    if is_4d:
        uint64_output = np.zeros((batch, num_heads, seq_len, last_dim), dtype=np.uint64)
        uint64_output_3d = uint64_output.reshape(batch * num_heads, seq_len, last_dim)
    else:
        uint64_output = np.zeros((num_heads, seq_len, last_dim), dtype=np.uint64)
        uint64_output_3d = uint64_output
    
    # Process each (head, sequence) position
    for head_idx in range(uint64_output_3d.shape[0]):
        for seq_idx in range(seq_len):
            # Get valid indices (not -1)
            indices = topk_idx_3d[head_idx, seq_idx]
            valid_indices = indices[indices >= 0]
            
            # Set bits in uint64 array
            for idx in valid_indices:
                uint64_idx = idx // uint64_bits
                bit_pos = idx % uint64_bits
                # Use np.uint64 explicitly for bitwise operations
                uint64_output_3d[head_idx, seq_idx, uint64_idx] |= np.uint64(1) << np.uint64(bit_pos)
    
    return uint64_output, last_dim

def run_test(shape, max_seqlen_k, block_size, desc):
    """运行单个测试用例"""
    print(f"\n测试 {desc}:")
    print(f"形状: {shape}, max_seqlen_k: {max_seqlen_k}, block_size: {block_size}")
    
    # 生成随机topk索引 (模拟实际使用场景，包含一些-1值)
    num_heads, total_seqlen, k = shape
    k_blocks = (max_seqlen_k + block_size - 1) // block_size
    
    topk_idx = torch.randint(-1, k_blocks, shape, dtype=torch.int32, device="cuda")
    
    # 方法1: 使用NumPy实现
    topk_idx_cpu = topk_idx.cpu().numpy()
    start1 = time.time()
    uint64_np, last_dim_np = topk_to_uint64_numpy(topk_idx_cpu, max_seqlen_k, block_size)
    time1 = time.time() - start1
    
    # 方法2: 使用CUDA实现
    start2 = time.time()
    uint64_cuda, _ = topk_to_uint64(topk_idx, max_seqlen_k, block_size)
    last_dim_cuda = uint64_cuda.shape[-1]
    torch.cuda.synchronize()
    time2 = time.time() - start2
    
    # 验证结果 - 将NumPy结果转换为有符号整数进行比较
    uint64_cuda_cpu = uint64_cuda.cpu().numpy()
    
    # 将NumPy结果(无符号)转换为有符号整数
    uint64_np_signed = uint64_np.view(np.int64)
    
    is_equal = np.array_equal(uint64_np_signed, uint64_cuda_cpu)
    assert last_dim_np == last_dim_cuda, f"最后一维大小不一致: {last_dim_np} vs {last_dim_cuda}"
    
    print(f"结果相同: {is_equal}")
    print(f"最后一维大小: {last_dim_cuda}")
    print(f"输出形状: {uint64_np.shape} (NumPy实现), {uint64_cuda.shape} (CUDA实现)")
    print(f"NumPy执行时间: {time1*1000:.3f}ms")
    print(f"CUDA执行时间: {time2*1000:.3f}ms")
    print(f"加速比: {time1/time2:.2f}x")
    
    if not is_equal:
        # 找出不相等的位置
        mismatch_count = 0
        for idx in np.ndindex(uint64_np.shape):
            if uint64_np_signed[idx] != uint64_cuda_cpu[idx]:
                if mismatch_count < 5:
                    print(f"不匹配位置 {idx}: NumPy unsigned={uint64_np[idx]}, NumPy signed={uint64_np_signed[idx]}, CUDA={uint64_cuda_cpu[idx]}")
                mismatch_count += 1
        print(f"总计 {mismatch_count} 个不匹配位置")
    
    return is_equal

def convert_4d_topk_to_base_blockmask(
    topk_idx: torch.Tensor,
    max_seqlen_k: int,
    block_size: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    将4D形状 [batch, num_heads, seq_len, k] 的topk索引转换为块稀疏注意力掩码
    
    Args:
        topk_idx: 形状 [batch, num_heads, seq_len, k] 的块索引张量
        max_seqlen_k: 最大键序列长度（用于计算键块数量）
        block_size: block_size
        device: 输出设备
    
    Returns:
        mask: 布尔掩码，形状 [batch, num_heads, seq_len, k_blocks]
    """
    # 计算键块数量
    k_blocks = (max_seqlen_k + block_size - 1) // block_size  # 向上取整
    batch, num_heads, seq_len, k = topk_idx.shape

    # 初始化全False掩码
    mask = torch.zeros(batch, num_heads, seq_len, k_blocks, 
                       dtype=torch.bool, device=device)

    # 过滤掉 -1，确保索引合法
    valid_idx = topk_idx[topk_idx != -1]

    # 生成索引掩码
    batch_idx, head_idx, seq_idx, _ = torch.where(topk_idx != -1)  # 找到非-1索引的位置
    mask[batch_idx, head_idx, seq_idx, valid_idx] = True  # 设置对应位置为 True

    return mask

def run_4d_test(shape, max_seqlen_k, block_size, desc):
    """运行4D张量测试用例 - 比较NumPy和CUDA实现"""
    print(f"\n测试 {desc}:")
    print(f"形状: {shape}, max_seqlen_k: {max_seqlen_k}, block_size: {block_size}")
    
    # 生成随机topk索引
    batch, num_heads, seq_len, k = shape
    k_blocks = (max_seqlen_k + block_size - 1) // block_size
    
    topk_idx = torch.randint(-1, k_blocks, shape, dtype=torch.int32, device="cuda")
    
    # 方法1: 使用NumPy实现
    topk_idx_cpu = topk_idx.cpu().numpy()
    start1 = time.time()
    uint64_np, last_dim_np = topk_to_uint64_numpy(topk_idx_cpu, max_seqlen_k, block_size)
    time1 = time.time() - start1
    
    # 方法2: 使用CUDA实现
    start2 = time.time()
    try:
        # 直接使用4D张量
        uint64_cuda, _ = topk_to_uint64(topk_idx, max_seqlen_k, block_size)
        last_dim_cuda = uint64_cuda.shape[-1]
        direct_4d_supported = True
    except Exception as e:
        print(f"4D测试失败: {e}")
        print("回退到3D处理...")
        
        # 回退到3D处理
        topk_idx_3d = topk_idx.reshape(batch * num_heads, seq_len, k)
        uint64_cuda, _ = topk_to_uint64(topk_idx_3d, max_seqlen_k, block_size)
        last_dim_cuda = uint64_cuda.shape[-1]
        uint64_cuda = uint64_cuda.reshape(batch, num_heads, seq_len, last_dim_cuda)
        direct_4d_supported = False
    
    torch.cuda.synchronize()
    time2 = time.time() - start2
    
    # 验证结果 - 将NumPy结果转换为有符号整数进行比较
    uint64_cuda_cpu = uint64_cuda.cpu().numpy()
    
    # 将NumPy结果(无符号)转换为有符号整数
    uint64_np_signed = uint64_np.view(np.int64)
    
    is_equal = np.array_equal(uint64_np_signed, uint64_cuda_cpu)
    assert last_dim_np == last_dim_cuda, f"最后一维大小不一致: {last_dim_np} vs {last_dim_cuda}"
    
    print(f"结果相同: {is_equal}")
    print(f"最后一维大小: {last_dim_cuda}")
    print(f"输出形状: {uint64_np.shape} (NumPy实现), {uint64_cuda.shape} (CUDA实现)")
    print(f"NumPy执行时间: {time1*1000:.3f}ms")
    print(f"CUDA执行时间: {time2*1000:.3f}ms")
    print(f"加速比: {time1/time2:.2f}x")
    print(f"是否直接支持4D: {direct_4d_supported}")
    
    if not is_equal:
        # 找出不相等的位置
        mismatch_count = 0
        for idx in np.ndindex(uint64_np.shape):
            if uint64_np_signed[idx] != uint64_cuda_cpu[idx]:
                if mismatch_count < 5:
                    print(f"不匹配位置 {idx}: NumPy unsigned={uint64_np[idx]}, NumPy signed={uint64_np_signed[idx]}, CUDA={uint64_cuda_cpu[idx]}")
                mismatch_count += 1
        print(f"总计 {mismatch_count} 个不匹配位置")
    
    return is_equal and direct_4d_supported

def main():
    print("开始测试 topk_to_uint64 CUDA实现与NumPy实现的对比")
    
    # 测试用例
    test_cases = [
        ((2, 128, 16), 256, 4, "小规模测试"),
        ((8, 512, 32), 1024, 16, "中等规模测试"),
        ((16, 1024, 64), 2048, 32, "大规模测试"),
        ((32, 2048, 128), 4096, 64, "超大规模测试"),
        ((16, 1024, 64), 8192, 16, "多uint64测试"),
    ]
    
    success = 0
    for shape, max_seqlen_k, block_size, desc in test_cases:
        if run_test(shape, max_seqlen_k, block_size, desc):
            success += 1
    
    print(f"\n总结: {success}/{len(test_cases)} 个测试通过")
    
    # 测试4D张量格式（batch * head * seq * dim）
    print("\n\n开始测试4D张量格式（batch * head * seq * dim）")
    
    # 4D测试用例
    test_cases_4d = [
        ((2, 2, 64, 16), 256, 4, "4D小规模测试"),
        ((4, 8, 128, 32), 1024, 16, "4D中等规模测试"),
    ]
    
    success_4d = 0
    for shape, max_seqlen_k, block_size, desc in test_cases_4d:
        if run_4d_test(shape, max_seqlen_k, block_size, desc):
            success_4d += 1
    
    print(f"\n总结: {success_4d}/{len(test_cases_4d)} 个4D测试通过")

if __name__ == "__main__":
    main() 