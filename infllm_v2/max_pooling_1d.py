import torch
from . import C

def max_pooling_1d(
    input: torch.Tensor, # num_heads x q_len x k_len
    cache_len: int,
    local_blocks: int,
    init_blocks: int,
    block_size: int = 64,
    stride: int = 16,
) -> torch.Tensor:
    assert input.dtype == torch.float16 or input.dtype == torch.bfloat16
    input = input.contiguous()
    stride = block_size // stride
    kernel_size = stride + 1
    padding = 1
    num_heads = input.shape[0]
    q_len = input.shape[1]
    k_len = input.shape[2]
    total_len = q_len + cache_len
    out_len = (total_len + block_size - 1) // block_size
    output = torch.zeros(num_heads, q_len, out_len, device=input.device, dtype=input.dtype)
    C.max_pooling_1d(
        torch.cuda.current_stream().cuda_stream,
        input.data_ptr(),
        output.data_ptr(),
        input.dtype == torch.bfloat16,
        num_heads,
        q_len,
        k_len,
        out_len,
        cache_len,
        kernel_size,
        stride,
        padding,
        block_size,
        local_blocks,
        init_blocks,
    )
    return output

