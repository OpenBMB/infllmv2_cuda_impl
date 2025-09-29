import torch
from .naive_softmax import torch_naive_softmax
import time

'''
causal False
如果用bfloat16, diff ~ 0.0078左右
如果用float32, diff ~ 1e-7左右
'''

# qi -> [qi * block_q: qi * block_q + block_q]
def apply_causal_mask(S, qi, kj, max_k_len, stride, causal=False):
    block_q, block_k = 16, 16
    for _i, i in enumerate(range(qi * block_q, (qi + 1) * block_q)):
        left_border = 0
        right_border = max(0, (i - stride + 1) // stride) if causal else max_k_len # offset = 0
        for _j, j in enumerate(range(kj * block_k, (kj + 1) * block_k)):
            if j < left_border or j >= right_border:
                S[:, _i, _j] = float('-inf')
    return S

def compresssed_lse_online_softmax(q, k, compressed_k, block_q, block_k, seq_dim=0, causal=False):
    device = q.device
    dtype = q.dtype
    

    head_n_q, head_n_k = q.shape[1], k.shape[1]
    q_len, k_len = q.shape[0], k.shape[0]
    k_compressed_len = compressed_k.shape[0]
    d = q.shape[-1]
    group_size = head_n_q // head_n_k
    # assert group_size == 16, "group_size must be 16, otherwise unsupported"

    phase_1_stride = 64
    if k_len == k_compressed_len:
        phase_1_stride = 16

    # q [seq_len, head_n_q, head_dim]
    # k [seq_len, head_n_k, head_dim]

    max_q_len = q_len
    if q_len % 16 != 0:
        q = torch.cat([q, torch.zeros(16 - q_len % 16, head_n_q, d, device=device, dtype=dtype)], dim=0)
        q_len = q_len + 16 - q_len % 16

    max_k_compressed_len = k_compressed_len
    if k_compressed_len % 16 != 0:
        compressed_k = torch.cat([compressed_k, torch.zeros(16 - k_compressed_len % 16, head_n_k, d, device=device, dtype=dtype)], dim=0)
        k_compressed_len = k_compressed_len + 16 - k_compressed_len % 16

    max_k_len = k_len
    if k_len % 16 != 0:
        k = torch.cat([k, torch.zeros(16 - k_len % 16, head_n_k, d, device=device, dtype=dtype)], dim=0)
        k_len = k_len + 16 - k_len % 16
    
    q = q.transpose(0, 1) # (head_n_q, q_len, head_dim)
    k = k.transpose(0, 1) # (head_n_k, k_len, head_dim)
    k = k.repeat_interleave(group_size, dim=0)
    compressed_k = compressed_k.transpose(0, 1) # (head_n_k, k_compressed_len, head_dim)
    compressed_k = compressed_k.repeat_interleave(group_size, dim=0)

    # breakpoint()
    block_q_n, block_k_n = q_len // block_q, k_len // block_k
    
    # lse compress
    block_k_n_comp = k_compressed_len // block_k
    q_blocks = q.reshape(head_n_q, block_q_n, block_q, d)
    compressed_k_blocks = compressed_k.reshape(head_n_q, block_k_n_comp, block_k, d)
    k_blocks = k.reshape(head_n_q, block_k_n, block_k, d)

    m_blocks = torch.full((head_n_q, block_q_n, block_q), float('-inf'), device=device, dtype=dtype)
    l_blocks = torch.zeros((head_n_q, block_q_n, block_q), device=device, dtype=dtype)

    # pass one 计算 max 和 sum
    # 用压缩的k
    for qi in range(0, block_q_n):
        m_i = torch.full((head_n_q, block_q), float('-inf'), device=device, dtype=dtype)
        l_i = torch.zeros((head_n_q, block_q), device=device, dtype=dtype)
        local_qi = q_blocks[:, qi, :, :] # (head_n_q, block_q, d)
        for kj in range(0, block_k_n_comp):
            local_kj = compressed_k_blocks[:, kj, :, :] # (head_n_q, block_k, d)
            S = local_qi @ local_kj.transpose(-2, -1) / (d ** 0.5) # (head_n_q, block_q, block_k)
            S = apply_causal_mask(S, qi, kj, max_k_compressed_len, phase_1_stride, causal)
            m_i_1 = m_i.clone()
            m_i = torch.maximum(m_i, S.max(dim=-1).values) # (head_n_q, block_q)
            # l_i = l_i-1.adjusted + new_exp_sum
            l_i = l_i * torch.exp(m_i_1 - m_i) + torch.exp(S - m_i.unsqueeze(-1)).sum(dim=-1)
            # breakpoint()
        m_blocks[:, qi, :] = m_i
        # breakpoint()
        l_blocks[:, qi, :] = torch.log(l_i) + m_i

    # pass two 计算 softmax
    # 用原始的k
    out_float = torch.empty((head_n_q, block_q_n, block_k_n, block_q, block_k), device=device, dtype=dtype)
    for qi in range(0, block_q_n):
        local_qi = q_blocks[:, qi, :, :] # (head_n_q, block_q, d)
        for kj in range(0, block_k_n):
            local_kj = k_blocks[:, kj, :, :] # (head_n_q, block_k, d)
            S = local_qi @ local_kj.transpose(-2, -1) / (d ** 0.5) # (head_n_q, block_q, block_k)
            S = apply_causal_mask(S, qi, kj, max_k_len, 16, causal)
            out_float[ :, qi, kj, :, :] = torch.exp(S - l_blocks[:, qi, :].unsqueeze(-1))

    out = out_float.transpose(2,3).reshape(head_n_q, q_len, k_len).to(dtype)
    out = out.reshape(head_n_k, group_size, q_len, k_len).sum(dim=1) [:, :max_q_len, :max_k_len]
    out = torch.where(torch.isnan(out), 0, out)[:, :max_q_len, :max_k_len]

    return out

if __name__ == "__main__":
    device = torch.device("cuda:0")
    q_len, k_len = 256, 15
    # q_len, k_len = 32, 32
    q_head_n, k_head_n = 32, 2
    block_q, block_k = 16, 16
    d = 128
    causal = True
    block_q_n, block_k_n = q_len // block_q, k_len // block_k

    dtype = torch.bfloat16
    q = torch.randn(q_len, q_head_n, d, dtype=dtype, device=device)
    k = torch.randn(k_len, k_head_n, d, dtype=dtype, device=device)

    time_start = time.time()
    out = torch_online_softmax(q, k, block_q, block_k, causal=causal)
    time_end = time.time()
    print(f"time: {time_end - time_start}")

    time_start = time.time()
    std = torch_naive_softmax(q, k, causal=causal)
    time_end = time.time()
    print(f"time: {time_end - time_start}")

    print(std.shape)
    diff = out - std
    breakpoint()
    print(diff)
    print(f"diff: {diff.abs().max()}")
    breakpoint()