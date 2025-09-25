import torch
from infllm_v2 import infllmv2_attn_stage1


def naive_torch_implementation(q, k, causal=False):
    # q_len < k_len
    # expect such mask
    # 1 1 1 0 0 
    # 1 1 1 1 0
    # 1 1 1 1 1
    q_len, k_len = q.shape[0], k.shape[0]
    k_head = k.shape[1]
    group_size = q.shape[1] // k_head

    q = q.transpose(0, 1) # (n_head, q_len, head_dim)
    k = k.transpose(0, 1) # (n_kv_head, k_len, head_dim)
    k = k.repeat_interleave(group_size, dim=0)

    S = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5) # (n_head, q_len, k_len)
    shift = max(k_len - q_len, 0)
    if causal:
        base_mask = torch.triu(
            torch.ones((q_len, k_len), dtype=torch.bool, device=S.device),
            diagonal=1 + shift,
        )
        mask = base_mask.expand(S.shape[0], q_len, k_len)
        score = S.masked_fill(mask, float('-inf'))
    else:
        score = S
    # softmax
    out = torch.softmax(score, dim=-1)
    out = out.reshape(k_head, group_size, q_len, k_len).sum(dim=1)

    return out # (n_head, q_len, k_len)

def test_naive_torch_implementation(causal=False):
    device = torch.device('cuda:0')
    n_head = 32
    n_kv_head = 2
    q = torch.randn(64, n_head, 128, device=device, dtype=torch.bfloat16) # (seq_len, n_head, head_dim)
    k = torch.randn(128, n_kv_head, 128, device=device, dtype=torch.bfloat16)
    score = naive_torch_implementation(q, k, causal)
    print(score)

    stage1_score = infllmv2_attn_stage1(
        q, 
        k,
        k,
        cu_seqlens_q=torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, k.shape[0]], device=q.device, dtype=torch.int32),
        cu_seqlens_v=torch.tensor([0, k.shape[0]], device=q.device, dtype=torch.int32),
        max_seqlen_q=q.shape[0],
        max_seqlen_k=k.shape[0],
        causal=causal,
    )
    print(stage1_score)

    max_diff = torch.max(torch.abs(score - stage1_score))
    print(max_diff.item())
    breakpoint()


if __name__ == "__main__":
    test_naive_torch_implementation(causal=False)
    test_naive_torch_implementation(causal=True)
    



