# InfLLM V2 CUDA Kernel Implementation: Stage 2 Sparse Attention Computation

[English](README.md) | [中文](README_zh.md)

This repository contains the optimized CUDA kernel implementation for **InfLLM V2's Stage 2: Sparse Attention Computation**. Our implementation provides high-performance sparse attention kernels that enable Large Language Models (LLMs) to efficiently process long contexts with trainable sparse patterns.

## Overview

InfLLM V2 introduces a novel two-stage approach for efficient long-context processing:
- **Stage 1**: Block selection and scoring (implementation not included in this repo)
- **Stage 2**: Sparse attention computation on selected blocks (this implementation)

This CUDA kernel implementation focuses on Stage 2, providing optimized sparse attention computation that:
- Significantly reduces computational costs for both forward and backward phases
- Seamlessly integrates with existing transformer architectures

Built upon [FlashAttention](https://github.com/Dao-AILab/flash-attention) 2.4.2, our kernels leverage efficient memory access patterns and optimized Top-K implementations.

![InfLLM V2 Architecture](assets/infllm-v2.png)

## Kernel Design Features
- **Token-level Query, Block-level Key-Value**: Avoids training-inference inconsistency during decoding
- **Selective Block Attention**: Performs attention only on blocks selected in Stage 1
- **Linear Complexity**: O(l) complexity for long sequences

## News

- [2025/06] Initial release of InfLLM V2 with full sparse attention support
- [2025/06] Integration with [MiniCPM4](https://github.com/OpenBMB/MiniCPM) model family


## Kernel Implementation Details
- `infllmv2_sparse_attn_fwd`: Forward pass kernel
- `infllmv2_sparse_attn_bwd`: Backward pass kernel (for training)


## Installation

### Requirements

- PyTorch 1.12+
- CUDA 11.6+ (with CUDA development toolkit)
- Python 3.7+
- Linux operating system
- Sufficient GPU memory for kernel compilation
- Ninja build system (for faster compilation)

### Build from Source

```bash
# Clone the repository
git clone https://github.com/your-org/infllm-v2-cuda.git
cd infllm-v2-cuda

# Install with CUDA kernel compilation
pip install -e .

# Or install with specific CUDA architecture
TORCH_CUDA_ARCH_LIST="8.0;9.0" pip install -e .
```


## Usage

### CUDA Kernel API

The InfLLM V2 CUDA kernel provides the following main interface for Stage 2 sparse attention computation:

```python
from infllm_v2 import infllmv2_sparse_attn_func

# Stage 2: Sparse Attention Computation Kernel
# Inputs:
#   - q_unpad: Queries tensor (token-level)
#   - k_unpad, v_unpad: Keys and Values tensors (block-level)
#   - cu_seqlens_q, cu_seqlens_k: Cumulative sequence lengths
#   - topk_idx: Selected block indices from Stage 1
#   - max_seqlen_q, max_seqlen_k: Maximum sequence lengths
#   - block_window_size: Optional local attention window size

out_unpad = infllmv2_sparse_attn_func(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    topk_idx,  # Block indices selected in Stage 1
    max_seqlen_q, max_seqlen_k,
    block_window_size = 0,  # Additional local window for attention
)
```

### Kernel Parameters

- **q_unpad**: Query tensor in unpadded format (bfloat16)
- **k_unpad, v_unpad**: Key and Value tensors in unpadded format
- **topk_idx**: Integer tensor containing selected block indices from Stage 1
- **block_window_size**: Size of local attention window (0 to disable)

### Performance Considerations

- The kernel automatically handles different GPU architectures (SM80/SM90)
- Optimized for batch processing with variable sequence lengths
- Memory efficient through unpadded tensor format
- Supports bfloat16 precision

## Supported GPU Architectures

- **SM 80**: A100
- **SM 90**: H100

## Performance Benchmarks

### Performance Comparison: InfLLMv2 vs FlashAttention

All benchmarks were conducted with the following configuration:
- **GPU**: NVIDIA H100
- **Head Dimension**: 128
- **Number of Heads**: 2  
- **Query Heads**: 32
- **Block Size**: 64
- **Selected Blocks**: 64
- **Attention Type**: Causal

#### Detailed Performance Results

| Sequence Length | Batch Size | Implementation | Forward (ms) | Backward (ms) | Combined (ms) | Speedup vs FlashAttention |
|-----------------|------------|----------------|-------------|---------------|---------------|----------------------------|
| 32,768 | 8 | Flash Attention | 201.46 | 526.62 | 728.08 | 1x |
| 32,768 | 8 | Triton NSA | 169.11 | 343.82 | 512.93 | 1.42x |
| 32,768 | 8 | InfLLMv2 | 133.60 | 330.04 | 463.64 | 1.57x |
| 65,536 | 4 | Flash Attention | 409.29 | 1037.46 | 1446.75 | 1x |
| 65,536 | 4 | Triton NSA | 181.88 | 469.00 | 650.88 | 2.22x |
| 65,536 | 4 | InfLLMv2 | 142.31 | 381.55 | 523.86 | 2.76x |
| 131,072 | 2 | Flash Attention | 831.77 | 2063.11 | 2894.88 | 1x |
| 131,072 | 2 | Triton NSA | 216.10 | 589.66 | 805.76 | 3.59x |
| 131,072 | 2 | InfLLMv2 | 158.42 | 468.90 | 627.32 | 4.61x |

#### Performance Summary (Combined Time)

```
Sequence Length    Batch Size    FlashAttention    InfLLMv2    Speedup
32,768             8             728.08 ms         463.64 ms   1.57x
65,536             4             1446.75 ms        523.86 ms   2.76x
131,072            2             2894.88 ms        627.32 ms   4.61x
```

### Key Performance Highlights

- **Up to 4.6x speedup** for 128K sequences compared to FlashAttention  
- Performance gains scale with sequence length and sparsity
- Memory efficiency enables processing longer sequences on single GPU

## Citation

If you use the InfLLM V2 CUDA kernels in your research, please cite:

```bibtex
@article{minicpm4,
  title={MiniCPM4: Ultra-Efficient LLMs on End Devices},
  author={MiniCPM},
  year={2025}
}
```

## Acknowledgments
- [MiniCPM4](https://github.com/OpenBMB/MiniCPM): For model integration and testing
- [FlashAttention](https://github.com/Dao-AILab/flash-attention): The foundational CUDA kernel architecture we built upon
- [Block Sparse Attention](https://github.com/mit-han-lab/Block-Sparse-Attention): Inspiration for block-sparse kernel design



## License

* 本仓库中代码依照 [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) 协议开源

