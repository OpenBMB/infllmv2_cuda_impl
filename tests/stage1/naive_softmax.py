import torch
import math

def torch_naive_softmax(q, k, causal=False):
        # q_len < k_len
    # expect such mask
    # 1 1 1 0 0 
    # 1 1 1 1 0
    # 1 1 1 1 1
    q_len, k_len = q.shape[0], k.shape[0]
    k_head = k.shape[1]
    group_size = q.shape[1] // k_head
    assert group_size == 16

    q = q.transpose(0, 1) # (n_head, q_len, head_dim)
    k = k.transpose(0, 1) # (n_kv_head, k_len, head_dim)
    k = k.repeat_interleave(group_size, dim=0)

    S = q @ k.transpose(-2, -1)
    scale = 1.0 / math.sqrt(q.size(-1)) # (n_head, q_len, k_len)
    S = S * scale
    # breakpoint()
    shift = max(k_len - q_len, 0)
    if causal:
        q_idx = torch.arange(q_len, device=S.device)
        # A[14] = 0
        # A[15] = 1
        q_compress_idx = ((q_idx - 15) // 16) + k_len - (q_len - 16 + 1) // 16
        q_compress_idx = q_compress_idx.clamp(0, k_len)
        mask = [[0] * q_compress_idx[i] + [1] * (k_len - q_compress_idx[i]) for i in range(q_len)]
        mask = torch.tensor(mask, dtype=torch.bool, device=S.device)
        mask = mask.expand(S.shape[0], q_len, k_len)
        score = S.masked_fill(mask, float('-inf'))
    else:
        score = S
    # softmax
    out = torch.softmax(score, dim=-1)
    out = out.reshape(k_head, group_size, q_len, k_len).sum(dim=1)
    out = torch.where(torch.isnan(out), 0, out)

    return out # (n_head, q_len, k_len)


if __name__ == "__main__":
    causal = False
    q = torch.randn(256, 32, 128, dtype=torch.float32)
    k = torch.randn(15, 2, 128, dtype=torch.float32)
    out1= torch_naive_softmax(q, k, causal=causal)

    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    out2= torch_naive_softmax(q, k, causal=causal)
    
    q = q.to(torch.float16)
    k = k.to(torch.float16)
    out3= torch_naive_softmax(q, k, causal=causal)

    # print(out1)
    # print(out2)
    # print(out3)
    # print(out1 - out2)
    print((out1 - out2).abs().max())
    # print(out1 - out3)
    print((out1 - out3).abs().max())
    breakpoint()