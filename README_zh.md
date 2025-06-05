# InfLLM V2 CUDA 内核实现：第二阶段稀疏注意力计算

本仓库包含了 **InfLLM V2 第二阶段：稀疏注意力计算** 的优化 CUDA 内核实现。我们的实现提供了高性能的稀疏注意力内核，使大型语言模型（LLM）能够通过可训练的稀疏模式高效处理长上下文。

## 概述

InfLLM V2 引入了一种新颖的两阶段方法来高效处理长上下文：
- **第一阶段**：块选择和评分（本仓库未包含此实现）
- **第二阶段**：对选定块进行稀疏注意力计算（本实现）

这个 CUDA 内核实现专注于第二阶段，提供优化的稀疏注意力计算：
- 显著降低前向和反向阶段的计算成本
- 与现有 Transformer 架构无缝集成

基于 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 2.4.2 构建，我们的内核利用了高效的内存访问模式和优化的 Top-K 实现。

![InfLLM V2 架构](assets/infllm-v2.png)

## 内核设计特性
- **Token 级别查询，块级别键值**：避免解码时的训练-推理不一致性
- **选择性块注意力**：仅对第一阶段选择的块执行注意力计算
- **线性复杂度**：对于长序列具有 O(l) 复杂度

## 新闻

- [2025/06] InfLLM V2 初始版本发布，支持完整的稀疏注意力
- [2025/06] 与 [MiniCPM4](https://github.com/OpenBMB/MiniCPM) 模型系列集成


## 内核实现细节
- `infllmv2_sparse_attn_fwd`：前向传递内核
- `infllmv2_sparse_attn_bwd`：反向传递内核（用于训练）


## 安装

### 系统要求

- PyTorch 1.12+
- CUDA 11.6+（需要 CUDA 开发工具包）
- Python 3.7+
- Linux 操作系统
- 足够的 GPU 内存用于内核编译
- Ninja 构建系统（用于加速编译）

### 从源码构建

```bash
# 克隆仓库
git clone https://github.com/your-org/infllm-v2-cuda.git
cd infllm-v2-cuda

# 安装并编译 CUDA 内核
pip install -e .

# 或者指定 CUDA 架构进行安装
TORCH_CUDA_ARCH_LIST="8.0;9.0" pip install -e .
```


## 使用方法

### CUDA 内核 API

InfLLM V2 CUDA 内核为第二阶段稀疏注意力计算提供以下主要接口：

```python
from infllm_v2 import infllmv2_sparse_attn_func

# 第二阶段：稀疏注意力计算内核
# 输入：
#   - q_unpad: 查询张量（token 级别）
#   - k_unpad, v_unpad: 键和值张量（块级别）
#   - cu_seqlens_q, cu_seqlens_k: 累积序列长度
#   - topk_idx: 第一阶段选择的块索引
#   - max_seqlen_q, max_seqlen_k: 最大序列长度
#   - block_window_size: 可选的局部注意力窗口大小

out_unpad = infllmv2_sparse_attn_func(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    topk_idx,  # 第一阶段选择的块索引
    max_seqlen_q, max_seqlen_k,
    block_window_size = 0,  # 额外的局部注意力窗口
)
```

### 内核参数

- **q_unpad**：未填充格式的查询张量（bfloat16）
- **k_unpad, v_unpad**：未填充格式的键和值张量
- **topk_idx**：包含第一阶段选择的块索引的整数张量
- **block_window_size**：局部注意力窗口大小（0 表示禁用）

### 性能考虑

- 内核自动处理不同的 GPU 架构（SM80/SM90）
- 针对变长序列的批处理进行了优化
- 通过未填充张量格式实现内存高效
- 支持 bfloat16 精度

## 支持的 GPU 架构

- **SM 80**: A100
- **SM 90**: H100

## 性能基准测试

### 性能对比：InfLLMv2 vs FlashAttention

所有基准测试均采用以下配置：
- **GPU**：NVIDIA H100
- **头维度**：128
- **头数量**：2
- **查询头数**：32
- **块大小**：64
- **选择的块数**：64
- **注意力类型**：因果注意力

#### 详细性能结果

| 序列长度 | 批大小 | 实现方式 | 前向 (ms) | 反向 (ms) | 总计 (ms) | 相比 FlashAttention 的加速比 |
|----------|--------|----------|-----------|-----------|-----------|------------------------------|
| 32,768 | 8 | Flash Attention | 201.46 | 526.62 | 728.08 | 1x |
| 32,768 | 8 | Triton NSA | 169.11 | 343.82 | 512.93 | 1.42x |
| 32,768 | 8 | InfLLMv2 | 133.60 | 330.04 | 463.64 | 1.57x |
| 65,536 | 4 | Flash Attention | 409.29 | 1037.46 | 1446.75 | 1x |
| 65,536 | 4 | Triton NSA | 181.88 | 469.00 | 650.88 | 2.22x |
| 65,536 | 4 | InfLLMv2 | 142.31 | 381.55 | 523.86 | 2.76x |
| 131,072 | 2 | Flash Attention | 831.77 | 2063.11 | 2894.88 | 1x |
| 131,072 | 2 | Triton NSA | 216.10 | 589.66 | 805.76 | 3.59x |
| 131,072 | 2 | InfLLMv2 | 158.42 | 468.90 | 627.32 | 4.61x |

#### 性能总结（总时间）

```
序列长度        批大小    FlashAttention    InfLLMv2    加速比
32,768          8         728.08 ms         463.64 ms   1.57x
65,536          4         1446.75 ms        523.86 ms   2.76x
131,072         2         2894.88 ms        627.32 ms   4.61x
```

### 关键性能亮点

- 对于 128K 序列，相比 FlashAttention **最高可达 4.6 倍加速**
- 性能提升随序列长度和稀疏度增加而扩大
- 内存效率使得在单个 GPU 上处理更长序列成为可能

## 引用

如果您在研究中使用了 InfLLM V2 CUDA 内核，请引用：

```bibtex
@article{minicpm4,
  title={MiniCPM4: Ultra-Efficient LLMs on End Devices},
  author={MiniCPM},
  year={2025}
}
```

## 致谢
- [MiniCPM](https://github.com/OpenBMB/MiniCPM): 团队的模型集成和测试
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)：我们构建的基础 CUDA 内核架构
- [Block Sparse Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)：块稀疏内核设计的灵感来源

## 许可证

* 本仓库中代码依照 [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) 协议开源