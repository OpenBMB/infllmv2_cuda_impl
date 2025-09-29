import torch
from infllm_v2 import infllmv2_attn_stage1
from stage1.naive_softmax import torch_naive_softmax
from test_stage1 import naive_attention

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
    if causal:
        q_idx = torch.arange(q_len, device=S.device)
        q_compress_idx = ((q_idx + 1) // 16) - 1 + k_len - (q_len - 16 + 1) // 16
        q_compress_idx = q_compress_idx.clamp(0, k_len)
        mask = [[0] * q_compress_idx[i] + [1] * (k_len - q_compress_idx[i]) for i in range(q_len)]
        # breakpoint()
        mask = torch.tensor(mask, dtype=torch.bool, device=S.device)
        mask = mask.expand(1, q_len, k_len)
        print(mask)
        breakpoint()
        score = S.masked_fill(mask, float('-inf'))
    else:
        score = S
    # softmax
    out = torch.softmax(score, dim=-1)
    out = out.reshape(k_head, group_size, q_len, k_len).sum(dim=1)
    out = torch.where(torch.isnan(out), 0, out)

    # print(out)
    # breakpoint()
    return out # (n_head, q_len, k_len)

def test_naive_torch_implementation(causal=False):
    device = torch.device('cuda:0')
    n_head = 32
    n_kv_head = 2
    dtype = torch.bfloat16
    dim = 128
    q = torch.randn(256, n_head, dim, device=device, dtype=dtype) # (seq_len, n_head, head_dim)
    k = torch.randn(15, n_kv_head, dim, device=device, dtype=dtype)
    score = torch_naive_softmax(q, k, causal)
    naive_score = naive_torch_implementation(q, k, causal)
    naive_score2 = naive_attention(q.transpose(0, 1), k.transpose(0, 1), k.transpose(0, 1),[0, 256], [0, 15], causal)
    # print(score)
    assert torch.allclose(score, naive_score)
    assert torch.allclose(naive_score, naive_score2)

    stage1_score = infllmv2_attn_stage1(
        q.contiguous(), 
        k.contiguous(),
        torch.tensor([[[1]]], dtype=q.dtype, device=q.device),
        cu_seqlens_q=torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, k.shape[0]], device=q.device, dtype=torch.int32),
        max_seqlen_q=q.shape[0],
        max_seqlen_k=k.shape[0],
        causal=causal,
    )[:, :q.shape[0], :k.shape[0]]
    print(stage1_score)

    print(score[0] - stage1_score[0])
    max_diff = torch.max(torch.abs(score - stage1_score))
    max_head_0_diff = torch.max(torch.abs(score[0, :, :] - stage1_score[0, :, :]))
    max_head_1_diff = torch.max(torch.abs(score[1, :, :] - stage1_score[1, :, :]))
    print(f"max_diff = {max_diff}, max_head_0_diff = {max_head_0_diff}, max_head_1_diff = {max_head_1_diff}")
    print(stage1_score.sum())
    print(score.sum())
    breakpoint()


if __name__ == "__main__":
    test_naive_torch_implementation(causal=False)
    # test_naive_torch_implementation(causal=True)
    



