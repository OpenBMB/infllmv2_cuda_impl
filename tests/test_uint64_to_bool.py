import torch
import numpy as np
import time
from infllm_v2 import blockmask_to_uint64, uint64_to_bool

def py_uint64_to_bool(uint64_array, last_dim_size):
    """
    将 uint64 表示转换回布尔掩码，完全向量化实现 (Python版本)
    
    参数:
        uint64_array: 包含 uint64 值的 PyTorch 张量
        last_dim_size: 原始最后一维的大小
        
    返回:
        bool_mask: 布尔型 PyTorch 张量
    """
    # 记录原始形状
    original_shape = uint64_array.shape
    n_uint64_per_row = original_shape[-1]
    
    # 将 uint64_array 移到 CPU 并转换为 numpy
    uint64_np = uint64_array.cpu().numpy()
    
    # 将输入展平为2D: (所有高维展平, n_uint64_per_row)
    flat_dims = np.prod(original_shape[:-1], dtype=int)
    uint64_2d = uint64_np.reshape(flat_dims, n_uint64_per_row)
    
    # 创建结果数组
    result = np.zeros((flat_dims, last_dim_size), dtype=bool)
    
    # 为每个 uint64 处理 64 位 - 使用Python整数进行位操作
    for row in range(flat_dims):
        for col in range(n_uint64_per_row):
            # 将NumPy值转换为Python整数
            value = int(uint64_2d[row, col])
            for bit in range(64):
                bit_idx = col * 64 + bit
                if bit_idx < last_dim_size:
                    # 在Python整数上执行位操作
                    result[row, bit_idx] = bool((value >> bit) & 1)
    
    # 重塑回原始高维结构，最后一维是 last_dim_size
    high_dims = original_shape[:-1]
    bool_shape = high_dims + (last_dim_size,)
    
    # 将 numpy 数组转换回 torch.Tensor
    result_tensor = torch.tensor(result.reshape(bool_shape), dtype=torch.bool)
    
    return result_tensor

def run_test(shape, last_dim_size, desc):
    """运行单个测试用例"""
    print(f"\n测试 {desc}:")
    print(f"uint64 形状: {shape}, 原始布尔形状: {shape[:-1] + (last_dim_size,)}")
    
    # 生成随机布尔掩码，然后转换为 uint64
    original_shape = shape[:-1] + (last_dim_size,)
    cpu_mask = torch.randint(0, 2, original_shape, dtype=torch.bool)
    gpu_mask = cpu_mask.cuda()
    
    # 先将布尔掩码转换为 uint64
    uint64_result, _ = blockmask_to_uint64(gpu_mask)
    
    # CUDA 实现：将 uint64 转回布尔掩码
    cuda_start = time.time()
    cuda_bool_result = uint64_to_bool(uint64_result, last_dim_size)
    torch.cuda.synchronize()
    cuda_time = time.time() - cuda_start
    
    # Python 实现
    py_start = time.time()
    py_bool_result = py_uint64_to_bool(uint64_result, last_dim_size)
    py_time = time.time() - py_start
    
    # 验证结果 - 确保两个结果都在 CPU 上
    cuda_cpu = cuda_bool_result.cpu()
    
    # 验证恢复后的布尔掩码与原始布尔掩码是否相同
    is_equal_to_original = torch.all(cuda_cpu == cpu_mask).item()
    
    # 验证 CUDA 和 Python 实现的结果是否相同
    is_equal_implementations = torch.all(cuda_cpu == py_bool_result).item()
    
    print(f"与原始布尔掩码相同: {is_equal_to_original}")
    print(f"CUDA 与 Python 实现结果相同: {is_equal_implementations}")
    print(f"输出形状: {cuda_bool_result.shape} (CUDA), {py_bool_result.shape} (Python)")
    print(f"CUDA 执行时间: {cuda_time*1000:.3f}ms")
    print(f"Python 执行时间: {py_time*1000:.3f}ms")
    print(f"加速比: {py_time/cuda_time:.2f}x")
    
    if not is_equal_to_original:
        # 找出不相等的位置
        mismatch = (cuda_cpu != cpu_mask).nonzero(as_tuple=True)
        if len(mismatch[0]) > 0:
            for i in range(min(5, len(mismatch[0]))):
                idx = tuple(tensor[i].item() for tensor in mismatch)
                print(f"与原始掩码不匹配位置 {idx}: 恢复={cuda_cpu[idx].item()}, 原始={cpu_mask[idx].item()}")
    
    if not is_equal_implementations:
        # 找出不相等的位置
        mismatch = (cuda_cpu != py_bool_result).nonzero(as_tuple=True)
        if len(mismatch[0]) > 0:
            for i in range(min(5, len(mismatch[0]))):
                idx = tuple(tensor[i].item() for tensor in mismatch)
                print(f"与Python实现不匹配位置 {idx}: CUDA={cuda_cpu[idx].item()}, Python={py_bool_result[idx].item()}")
    
    return is_equal_to_original and is_equal_implementations

def main():
    print("开始对拍测试 uint64_to_bool 的 Python 和 CUDA 实现")
    
    # 测试用例
    test_cases = [
        ((100, 1), 64, "刚好一个 uint64 的情况"),
        ((100, 1), 63, "不是 64 倍数"),
        ((100, 2), 65, "略大于一个 uint64"),
        ((100, 2), 128, "两个 uint64"),
        ((100, 4), 200, "多个 uint64"),
        ((2, 3, 4, 2), 100, "多维张量"),
        ((1000, 16), 1000, "大规模测试"),
    ]
    
    success = 0
    for shape, last_dim_size, desc in test_cases:
        if run_test(shape, last_dim_size, desc):
            success += 1
    
    print(f"\n总结: {success}/{len(test_cases)} 个测试通过")

if __name__ == "__main__":
    main() 