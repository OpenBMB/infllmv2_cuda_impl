import torch
import numpy as np
import time
from infllm_v2 import blockmask_to_uint64 as cuda_blockmask_to_uint64

def py_blockmask_to_uint64(blockmask):
    """
    将 PyTorch blockmask 转换为 uint64 表示，完全向量化实现 (Python版本)
    
    参数:
        blockmask: 布尔型 PyTorch 张量
        
    返回:
        uint64_arrays: 与原张量相同高维结构的 uint64 数组
        last_dim_size: 原始最后一维的大小
    """
    # 记录原始形状
    original_shape = blockmask.shape
    last_dim_size = original_shape[-1]
    
    # 计算需要多少个 uint64 来表示最后一维
    n_uint64_per_row = (last_dim_size + 63) // 64
    
    # 将 blockmask 移到 CPU 并转换为 numpy
    blockmask_np = blockmask.cpu().numpy()
    
    # 将输入展平为2D: (所有高维展平, last_dim)
    flat_dims = np.prod(original_shape[:-1], dtype=int)
    blockmask_2d = blockmask_np.reshape(flat_dims, last_dim_size)
    
    # 创建结果数组
    result = np.zeros((flat_dims, n_uint64_per_row), dtype=np.uint64)
    
    # 填充零以使最后一维是64的倍数
    padded_length = n_uint64_per_row * 64
    if padded_length > last_dim_size:
        padded = np.zeros((flat_dims, padded_length), dtype=bool)
        padded[:, :last_dim_size] = blockmask_2d
        blockmask_2d = padded
    
    # 重塑为3D: (flat_dims, n_uint64_per_row, 64)
    blockmask_3d = blockmask_2d.reshape(flat_dims, n_uint64_per_row, 64)
    
    # 创建位权重向量: 2^0, 2^1, ..., 2^63
    bit_weights = np.uint64(1) << np.arange(64, dtype=np.uint64)
    
    # 批量乘以权重并求和
    result = (blockmask_3d * bit_weights).sum(axis=2)
    
    # 重塑回原始高维结构，最后一维是 n_uint64_per_row
    high_dims = original_shape[:-1]
    uint64_shape = high_dims + (n_uint64_per_row,)
    
    # 将 numpy 数组转换回 torch.Tensor，但始终保持在 CPU 设备上
    result_tensor = torch.tensor(result.reshape(uint64_shape), dtype=torch.int64)
    
    return result_tensor, last_dim_size

def run_test(shape, desc):
    """运行单个测试用例"""
    print(f"\n测试 {desc}:")
    print(f"形状: {shape}")
    
    # 生成随机布尔掩码
    cpu_mask = torch.randint(0, 2, shape, dtype=torch.bool)
    gpu_mask = cpu_mask.cuda()
    
    # CUDA 实现
    cuda_start = time.time()
    cuda_result, cuda_last_dim = cuda_blockmask_to_uint64(gpu_mask)
    torch.cuda.synchronize()
    cuda_time = time.time() - cuda_start
    
    # Python 实现
    py_start = time.time()
    py_result, py_last_dim = py_blockmask_to_uint64(gpu_mask)
    py_time = time.time() - py_start
    
    # 验证结果 - 确保两个结果都在 CPU 上
    cuda_cpu = cuda_result.cpu()
    is_equal = torch.all(cuda_cpu == py_result).item()
    
    print(f"结果相同: {is_equal}")
    print(f"原始最后一维大小: {cuda_last_dim} (CUDA), {py_last_dim} (Python)")
    print(f"输出形状: {cuda_result.shape} (CUDA), {py_result.shape} (Python)")
    print(f"CUDA 执行时间: {cuda_time*1000:.3f}ms")
    print(f"Python 执行时间: {py_time*1000:.3f}ms")
    print(f"加速比: {py_time/cuda_time:.2f}x")
    
    if not is_equal:
        # 找出不相等的位置
        mismatch = (cuda_cpu != py_result).nonzero(as_tuple=True)
        if len(mismatch[0]) > 0:
            for i in range(min(5, len(mismatch[0]))):
                idx = tuple(tensor[i].item() for tensor in mismatch)
                print(f"不匹配位置 {idx}: CUDA={cuda_cpu[idx].item()}, Python={py_result[idx].item()}")
    
    return is_equal

def main():
    print("开始对拍测试 blockmask_to_uint64 的 Python 和 CUDA 实现")
    
    # 测试用例
    test_cases = [
        ((100, 64), "刚好一个 uint64 的情况"),
        ((100, 63), "不是 64 倍数"),
        ((100, 65), "略大于一个 uint64"),
        ((100, 128), "两个 uint64"),
        ((100, 200), "超过三个 uint64"),
        ((2, 3, 4, 100), "多维张量"),
        ((1000, 1000), "大规模测试"),
        ((1000, 1000, 1000), "大规模测试"),
    ]
    
    success = 0
    for shape, desc in test_cases:
        if run_test(shape, desc):
            success += 1
    
    print(f"\n总结: {success}/{len(test_cases)} 个测试通过")

if __name__ == "__main__":
    main() 