# InfLLM V2 CUDA Kernel Implementation


This repository contains the optimized CUDA kernel implementation for **InfLLM V2's Two-Stage Sparse Attention Mechanism**. Our implementation provides high-performance kernels for both Stage 1 (Top-K Context Selection) and Stage 2 (Sparse Attention Computation), enabling Large Language Models (LLMs) to efficiently process long contexts with trainable sparse patterns.

## Overview

InfLLM V2 introduces a novel two-stage approach for efficient long-context processing:
- **Stage 1: Top-K Context Selection**: Block scoring and aggregation using semantic kernels (kernel computes and aggregates scores, selection performed externally)
- **Stage 2: Sparse Attention Computation**: Attention calculation on selected blocks

This CUDA kernel implementation includes both stages, providing:
- Optimized relevance score computation and aggregation for Stage 1 (Top-K selection performed externally)
- Efficient sparse attention on selected blocks for Stage 2
- Significant reduction in computational costs for both forward and backward phases


Built upon [FlashAttention](https://github.com/Dao-AILab/flash-attention), our kernels leverage efficient memory access patterns and optimized implementations for both stages.

![InfLLM V2 Architecture](assets/infllm-v2.png)

## Two-Stage Architecture

### Stage 1: Top-K Context Selection
The Top-K selection stage involves three sequential steps:
1. **Relevance Score Computation**: Computing scores between query tokens and each semantic kernel (compressed representations of key-value blocks), followed by softmax normalization
2. **Score Aggregation**: Aggregating relevance scores for each semantic kernel across the query group dimension using dimension reduction (hdim16_reduce)
3. **Block Selection (Post-processing)**: Selecting the top-K context blocks for each query token based on the aggregated scores

Note: The `infllmv2_attn_stage1` kernel handles steps 1 and 2 (score computation and aggregation). Only step 3 (Top-K selection) is performed outside the kernel.

### Stage 2: Sparse Attention Computation
The sparse attention stage performs standard attention computation, but only on the blocks selected in Stage 1:
- Support for both forward and backward passes
- Efficient memory access through block-sparse patterns

## Kernel Design Features
- **Token-level Query, Block-level Key-Value**: Avoids training-inference inconsistency during decoding
- **Trainable Context Selection**: Semantic kernels updated indirectly through token-level key vector optimization
- **Selective Block Attention**: Performs attention only on blocks selected in Stage 1

## Kernel Implementation Details

### Stage 1 Kernels
- `infllmv2_attn_stage1`: Calculates similarity scores between query tokens and compressed key representations
- Performs score aggregation across query group dimension (hdim16_reduce)
- Returns aggregated attention scores for subsequent Top-K selection (selection performed outside the kernel)
- Support for causal masking and variable sequence lengths

### Stage 2 Kernels
- `infllmv2_sparse_attn_fwd`: Forward pass kernel for sparse attention
- `infllmv2_sparse_attn_bwd`: Backward pass kernel for training

## Installation

### Requirements

- PyTorch 1.12+
- CUDA 11.6+ (with CUDA development toolkit)
- Python 3.7+
- Linux operating system
- Sufficient GPU memory for kernel compilation
- Ninja build system (for faster compilation)

### Build from Source

#### For Training / Inference (main branch)

```bash
# Clone the repository and use main branch for training
git clone https://github.com/OpenBMB/infllm_v2_cuda.git
cd infllm_v2_cuda
git checkout main

# Install with CUDA kernel compilation
pip install -e .

```

# Install with CUDA kernel compilation
pip install -e .

```


## Usage

### CUDA Kernel API

The InfLLM V2 CUDA kernel provides the following interfaces for the two-stage sparse attention:

#### Stage 1: Attention Score Computation and Aggregation (feature_infer branch)

```python
from infllm_v2 import infllmv2_attn_stage1

# Stage 1: Compute and aggregate relevance scores between queries and semantic kernels
# This kernel performs:
#   1. LSE approximation using compressed keys
#   2. Full attention score computation
#   3. Score aggregation across query group dimension (hdim16_reduce)
# Top-K selection must be performed separately on the aggregated scores
#
# Inputs:
#   - q: Query tensor (batch_size * n_heads, seqlen_q, head_dim)
#   - k: Compressed key tensor representing semantic kernels
#   - v: Placeholder tensor (not used in score computation)
#   - cu_seqlens_q, cu_seqlens_k: Cumulative sequence lengths
#   - max_seqlen_q, max_seqlen_k: Maximum sequence lengths

# Returns aggregated attention scores for subsequent Top-K selection
aggregated_scores = infllmv2_attn_stage1(
    q, k, v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    causal=True,  # Apply causal masking
    return_attn_probs=True  # Return attention scores
)

# Top-K selection should be performed on the returned aggregated scores
# (This step is not part of the kernel)
```

#### Stage 2: Sparse Attention Computation

```python
from infllm_v2 import infllmv2_attn_varlen_func

# Stage 2: Sparse Attention Computation Kernel
# Inputs:
#   - q_unpad: Queries tensor (token-level)
#   - k_unpad, v_unpad: Keys and Values tensors (block-level)
#   - cu_seqlens_q, cu_seqlens_k: Cumulative sequence lengths
#   - topk_idx: Selected block indices from Stage 1
#   - max_seqlen_q, max_seqlen_k: Maximum sequence lengths

out_unpad = infllmv2_attn_varlen_func(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    topk_idx,  # Block indices selected in Stage 1
    max_seqlen_q, max_seqlen_k
)
```

### Kernel Parameters

#### Stage 1 Parameters
- **q**: Query tensor with shape (batch_size * n_heads, seqlen_q, head_dim)
- **k**: Compressed key tensor representing semantic kernels
- **causal**: Whether to apply causal masking
- **return_attn_probs**: Whether to return attention scores (required for Top-K selection)
- **Output**: Aggregated attention scores matrix (reduced along query group dimension) for external Top-K selection

#### Stage 2 Parameters
- **q_unpad**: Query tensor in unpadded format (bfloat16)
- **k_unpad, v_unpad**: Key and Value tensors in unpadded format
- **topk_idx**: Integer tensor containing selected block indices from Stage 1


### Performance Considerations

- The kernel automatically handles different GPU architectures (SM80/SM90)
- Optimized for batch processing with variable sequence lengths
- Memory efficient through unpadded tensor format and block-sparse patterns
- Supports bfloat16 precision for both stages

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

* This repository is released under the [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) License. 

