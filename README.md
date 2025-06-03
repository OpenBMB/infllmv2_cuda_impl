# InfLLM V2

InfLLM V2 CUDA implementation with Flash Attention and CUTLASS support.

## Features

- Optimized CUDA kernels for efficient attention computation
- Integration with Flash Attention for memory-efficient attention
- CUTLASS library support for high-performance GPU operations
- Support for multiple GPU architectures (V100, T4, A100, RTX series, H100)
- Multiple data types (FP16, BF16) and head dimensions support

## Installation

```bash
pip install -e .
```

## Requirements

- PyTorch
- CUDA 11.6+
- Python 3.7+
- Sufficient GPU memory for compilation

## Usage

```python
import infllm_v2
```

## Build Options

Set the following environment variables to control the build:

- `INFLLM_V2_FORCE_BUILD=TRUE`: Force build from source
- `INFLLM_V2_SKIP_CUDA_BUILD=TRUE`: Skip CUDA compilation
- `INFLLM_V2_FORCE_CXX11_ABI=TRUE`: Force C++11 ABI
- `INFLLM_V2_FAST_BUILD=TRUE`: Enable fast build mode (default: True)

## Supported GPU Architectures

- SM 70: V100
- SM 75: T4, RTX 20 series
- SM 80: A100
- SM 86: RTX 30 series, RTX A6000
- SM 87: RTX A100
- SM 89: RTX 40 series
- SM 90: H100

## References

```bibtex
@misc{guo2024blocksparse,
  author       = {Guo, Junxian and Tang, Haotian and Yang, Shang and Zhang, Zhekai and Liu, Zhijian and Han, Song},
  title        = {{Block Sparse Attention}},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/mit-han-lab/Block-Sparse-Attention}}
}
```

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
