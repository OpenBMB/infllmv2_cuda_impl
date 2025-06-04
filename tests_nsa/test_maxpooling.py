# 输入
# kvhead_num = 2
# seqlen = 131072
# block_size = 16
# kvhead_num * seqlen * (seqlen / block_size)的fp16 Tensor

# 计算
# 所有元素在GPU上进行计算
# (seqlen / block_size)维度每5个元素做maxpooling, stride为4

# 输出
# maxpooling的耗时

import torch
import torch.nn.functional as F
import time
import math


def maxpooling(input_tensor, kernel_size=5, stride=4, device="cuda:0"):
    """
    执行maxpooling操作并计算理论和实际耗时

    Args:
        input_tensor: 输入张量，形状为[kvhead_num, seqlen, seqlen//block_size]
        kernel_size: 池化核大小
        stride: 池化步长
    
    Returns:
        maxpooling_result: 池化结果
        elapsed_time_ms: 实际执行耗时(ms)
    """
    kvhead_num, seqlen, last_dim = input_tensor.shape
    
    # 计算输入内存大小(GB)
    input_size_bytes = kvhead_num * seqlen * last_dim * 2  # float16是2字节
    input_size_gb = input_size_bytes / (1024**3)
    
    # 计算maxpooling的耗时
    torch.cuda.synchronize(device=device)
    start_time = time.time()
    
    if kernel_size == stride:
        # 当kernel_size等于stride时，使用view + torch.max实现（无需padding）
        # assert确保最后一维能被kernel_size整除
        assert last_dim % kernel_size == 0, "最后一维不能被kernel_size整除"
        
        # 重塑矩阵以便在第三维度上每kernel_size个元素做maxpooling
        input_tensor = input_tensor.view(kvhead_num, seqlen, -1, kernel_size)
        
        # 对每组kernel_size个元素执行最大池化（在最后一维上）
        maxpooling_result, _ = torch.max(input_tensor, dim=3)
        
        # 计算输出大小
        output_size_bytes = kvhead_num * seqlen * (last_dim // kernel_size) * 2  # float16是2字节
    else:
        # 如果kernel_size不等于stride，使用F.max_pool1d实现
        batch_size = kvhead_num * seqlen
        input_tensor = input_tensor.view(batch_size, 1, -1)  # 将其变为(batch, channel, length)的形式
        
        # 使用F.max_pool1d执行maxpooling
        pooled = F.max_pool1d(input_tensor, kernel_size=kernel_size, stride=stride)
        # pooled = input_tensor
        
        # 将结果view回原来的形状
        output_length = pooled.shape[2]
        maxpooling_result = pooled.view(kvhead_num, seqlen, output_length)
        
        # 计算输出大小
        output_size_bytes = kvhead_num * seqlen * output_length * 2  # float16是2字节
    
    torch.cuda.synchronize(device=device)
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000
    
    # 输出大小(GB)
    output_size_gb = output_size_bytes / (1024**3)
    
    # 返回结果和耗时
    return maxpooling_result, elapsed_time_ms, input_size_gb, output_size_gb


def run_benchmark(kvhead_num, seqlen, block_size, kernel_size=5, stride=4, num_runs=10, warm_up_runs=5, device="cuda:0"):
    """运行基准测试"""
    
    # RTX 4090理论内存带宽 (GB/s)
    rtx_4090_bandwidth = 1008.0  # GB/s
    print(f"RTX 4090理论内存带宽: {rtx_4090_bandwidth} GB/s")
    
    # 生成符合要求的三维输入矩阵
    input_tensor = torch.randn((kvhead_num, seqlen, seqlen // block_size), 
                              dtype=torch.float16, device=device)
    print(f"输入张量形状: {input_tensor.shape}")
    
    # Warm up阶段
    print("正在进行warm up...")
    for _ in range(warm_up_runs):
        _ = maxpooling(input_tensor, kernel_size, stride, device=device)
    
    # 多次运行并记录时间
    print(f"开始进行{num_runs}次测试...")
    times = []
    for i in range(num_runs):
        result, elapsed_time, input_size_gb, output_size_gb = maxpooling(
            input_tensor, kernel_size, stride, device=device)
        times.append(elapsed_time)
    torch.cuda.empty_cache()
    
    # 计算统计信息
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print("\n性能统计:")
    print(f"平均耗时: {avg_time:.3f}毫秒")
    print(f"最小耗时: {min_time:.3f}毫秒")
    print(f"最大耗时: {max_time:.3f}毫秒")
    
    # 计算理论时间和带宽利用率
    # 总访存量(GB) = 读取输入 + 写入输出
    total_memory_access_gb = input_size_gb + output_size_gb
    
    # 理论计算时间(基于带宽)
    read_time_ms = (input_size_gb / rtx_4090_bandwidth) * 1000
    write_time_ms = (output_size_gb / rtx_4090_bandwidth) * 1000
    theoretical_time_ms = read_time_ms + write_time_ms
    
    print(f"\n输入数据大小: {input_size_gb:.4f} GB")
    print(f"输出数据大小: {output_size_gb:.4f} GB")
    print(f"总访存量: {total_memory_access_gb:.4f} GB")
    print(f"理论读取时间: {read_time_ms:.3f}毫秒")
    print(f"理论写入时间: {write_time_ms:.3f}毫秒")
    print(f"理论总时间: {theoretical_time_ms:.3f}毫秒")
    
    # 基于最小时间计算带宽利用率（最小时间更接近硬件极限）
    bandwidth_utilization = (theoretical_time_ms / min_time) * 100
    print(f"带宽利用率(基于最小耗时): {bandwidth_utilization:.2f}%")
    
    print(f"结果形状: {result.shape}")
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    device = "cuda:1"
    with torch.no_grad():
        # 使用注释中指定的参数 kernel_size=5, stride=4
        print("=== 测试 kernel_size=5, stride=4 ===")
        run_benchmark(2, 131072, 16, 5, 4, device=device)
        
    with torch.no_grad():
        # 测试kernel_size == stride的情况
        # print("\n=== 测试 kernel_size=4, stride=4 ===")
        # run_benchmark(2, 131072, 16, 4, 4, device=device)
        pass

    # 打印内存使用情况
    # TODO maxpooling内存占用不正常
    print(f"峰值内存: {torch.cuda.max_memory_allocated(device=device) / 1e9} GB")
