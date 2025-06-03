import torch
import warnings

tp1_path = '/home/qiqi/downloads/debug/tp1'
tp2_rank0_path  = '/home/qiqi/downloads/debug/tp2_rank0'
tp2_rank1_path  = '/home/qiqi/downloads/debug/tp2_rank1'

tensor_names = ['q','k','v' , 'bwd_blockmask_uint64','fwd_blockmask_bool','out','softmax_lse','dout','dq','dk','dv','cu_seqlens_q','cu_seqlens_k','head_mask_type','streaming_info','ctx.max_seqlen_k_','ctx.n_block_dim','ctx.m_block_dim','ctx.window_size_left','ctx.window_size_right','ctx.p_dropout','ctx.softmax_scale','ctx.is_causal','ctx.exact_streaming','ctx.deterministic','topk_attn_output','compressed_attn_output','gated','query_layer','key_layer','value_layer',]
# 忽略 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
for t in tensor_names:
    tp1_t = torch.load(tp1_path+'/'+ t + '.pt')
    if isinstance(tp1_t, torch.Tensor):

        tp2_rank0_t = torch.load(tp2_rank0_path+'/'+ t + '.pt').to(tp1_t.device)
        tp2_rank1_t = torch.load(tp2_rank1_path+'/'+ t + '.pt').to(tp1_t.device)
    else:
        # 常数的情况
        tp2_rank0_t = torch.load(tp2_rank0_path+'/'+ t + '.pt')
        tp2_rank1_t = torch.load(tp2_rank1_path+'/'+ t + '.pt')
        print(t)
        print(tp1_t==tp2_rank0_t)
        print(tp1_t==tp2_rank1_t)
        continue
    if   t in  ['topk_idx','bwd_blockmask_uint64','fwd_blockmask_bool'] :
        print(t, tp1_t.shape, tp2_rank0_t.shape, tp2_rank1_t.shape)
        tp2 = torch.cat((tp2_rank0_t, tp2_rank1_t), dim=0)
    elif tp1_t.shape == tp2_rank0_t.shape:
        tp2 = tp2_rank0_t
    elif t in ['head_mask_type','streaming_info']:
        tp2 = torch.cat((tp2_rank0_t, tp2_rank1_t), dim=0)
    else:
        tp2 = torch.cat((tp2_rank0_t, tp2_rank1_t), dim=-2)
    print(t)
    if tp1_t.dtype == torch.bool:
        print(torch.sum(tp1_t != tp2))
    else:
        tp1_t = torch.nan_to_num(tp1_t, nan=0.0)
        tp2 = torch.nan_to_num(tp2, nan=0.0)
        print(torch.sum(torch.abs(tp1_t - tp2)))