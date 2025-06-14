#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <pybind11/pybind11.h>
#include "topk.cuh"
#include "get_probs.cuh"
#include "blockmask.cuh"
#include "topk_to_uint64.cuh"
#include "uint64_to_bool.cuh"
#include "max_pooling_1d.cuh"

// Forward declarations for Flash Attention functions
std::vector<at::Tensor> mha_fwd(at::Tensor &q, const at::Tensor &k, const at::Tensor &v, c10::optional<at::Tensor> &out_, c10::optional<at::Tensor> &alibi_slopes_, const float p_dropout, const float softmax_scale, bool is_causal, int window_size_left, int window_size_right, const int n_block_dim, const bool return_softmax, c10::optional<at::Generator> gen_);
std::vector<at::Tensor> mha_varlen_fwd(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, c10::optional<at::Tensor> &out_, const at::Tensor &cu_seqlens_q, const at::Tensor &cu_seqlens_k, c10::optional<at::Tensor> &seqused_k, c10::optional<at::Tensor> &alibi_slopes_, const int max_seqlen_q, const int max_seqlen_k, const float p_dropout, const float softmax_scale, const bool zero_tensors, const bool is_causal, int window_size_left, int window_size_right, const int n_block_dim, const bool return_softmax, c10::optional<at::Generator> gen_);
std::vector<at::Tensor> mha_bwd(const at::Tensor &dout, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &out, const at::Tensor &softmax_lse, c10::optional<at::Tensor> &dq_, c10::optional<at::Tensor> &dk_, c10::optional<at::Tensor> &dv_, c10::optional<at::Tensor> &alibi_slopes_, const float p_dropout, const float softmax_scale, const bool is_causal, int window_size_left, int window_size_right, const int n_block_dim, const bool deterministic, c10::optional<at::Generator> gen_, c10::optional<at::Tensor> &rng_state);
std::vector<at::Tensor> mha_varlen_bwd(const at::Tensor &dout, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &out, const at::Tensor &softmax_lse, c10::optional<at::Tensor> &dq_, c10::optional<at::Tensor> &dk_, c10::optional<at::Tensor> &dv_, const at::Tensor &cu_seqlens_q, const at::Tensor &cu_seqlens_k, c10::optional<at::Tensor> &alibi_slopes_, const int max_seqlen_q, const int max_seqlen_k, const float p_dropout, const float softmax_scale, const bool zero_tensors, const bool is_causal, int window_size_left, int window_size_right, const int n_block_dim, const bool deterministic, c10::optional<at::Generator> gen_, c10::optional<at::Tensor> &rng_state);
std::vector<at::Tensor> mha_fwd_kvcache(at::Tensor &q, const at::Tensor &kcache, const at::Tensor &vcache, c10::optional<const at::Tensor> &k_, c10::optional<const at::Tensor> &v_, c10::optional<const at::Tensor> &seqlens_k_, c10::optional<const at::Tensor> &rotary_cos_, c10::optional<const at::Tensor> &rotary_sin_, c10::optional<const at::Tensor> &cache_batch_idx_, c10::optional<at::Tensor> &alibi_slopes_, c10::optional<at::Tensor> &out_, const float softmax_scale, bool is_causal, int window_size_left, int window_size_right, bool is_rotary_interleaved, int num_splits, const int n_block_dim);
std::vector<at::Tensor> mha_fwd_block(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &cu_seqlens_q, const at::Tensor &cu_seqlens_k, const int m_block_dim, const int n_block_dim, const at::Tensor &head_mask_type, c10::optional<at::Tensor> &streaming_info_, c10::optional<at::Tensor> &row_blockmask_, const int max_seqlen_q_, const int max_seqlen_k_, const float p_dropout, const float softmax_scale, const bool is_causal, const bool exact_streaming, const bool return_softmax, int window_size_left, int window_size_right, int block_window_size, c10::optional<at::Generator> gen_);
std::vector<at::Tensor> mha_bwd_block(const at::Tensor &dout, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &out, const at::Tensor &softmax_lse, c10::optional<at::Tensor> &dq_, c10::optional<at::Tensor> &dk_, c10::optional<at::Tensor> &dv_, const at::Tensor &cu_seqlens_q, const at::Tensor &cu_seqlens_k, const int m_block_dim, const int n_block_dim, const at::Tensor &head_mask_type, c10::optional<at::Tensor> &streaming_info_, c10::optional<at::Tensor> &col_blockmask_, const int max_seqlen_q_, const int max_seqlen_k_, const float p_dropout, const float softmax_scale, const bool zero_tensors, const bool is_causal, int window_size_left, int window_size_right, int block_window_size, const bool deterministic, c10::optional<at::Generator> gen_, c10::optional<at::Tensor> &rng_state);
std::vector<at::Tensor> fwd_block_kvcache(at::Tensor &q, const at::Tensor &kcache, const at::Tensor &vcache, const int m_block_dim, const int n_block_dim, const at::Tensor &head_mask_type, c10::optional<at::Tensor> &streaming_info_, c10::optional<at::Tensor> &row_blockmask_, c10::optional<const at::Tensor> &k_, c10::optional<const at::Tensor> &v_, c10::optional<const at::Tensor> &seqlens_k_, c10::optional<const at::Tensor> &rotary_cos_, c10::optional<const at::Tensor> &rotary_sin_, c10::optional<const at::Tensor> &cache_batch_idx_, c10::optional<at::Tensor> &alibi_slopes_, c10::optional<at::Tensor> &out_, const float softmax_scale, bool is_causal, const bool exact_streaming, int window_size_left, int window_size_right, int block_window_size, bool is_rotary_interleaved, int num_splits);

#define DTYPE_SWITCH(COND, ...)               \
    [&] {                                     \
        if (COND == 0) {                      \
            using elem_type = __half;         \
            return __VA_ARGS__();             \
        } else {                              \
            using elem_type = __nv_bfloat16;  \
            return __VA_ARGS__();             \
        }                                     \
    }()

void topk(
    std::uintptr_t stream,
    int num_tokens, int dim, int top, int dtype,
    std::uintptr_t x,
    std::uintptr_t topk_val, std::uintptr_t topk_pos
) {
    DTYPE_SWITCH(dtype, [&] {
        topk_func<elem_type>(
            reinterpret_cast<cudaStream_t>(stream),
            num_tokens, dim, top, dtype,
            reinterpret_cast<elem_type*>(x),
            reinterpret_cast<elem_type*>(topk_val), reinterpret_cast<int*>(topk_pos)
        );
    });
}

void get_probs(
    std::uintptr_t stream,
    int n, int dim, int dtype,
    std::uintptr_t attn_probs, std::uintptr_t lse, float scale
) {
    DTYPE_SWITCH(dtype, [&] {
        get_probs_func<elem_type>(
            reinterpret_cast<cudaStream_t>(stream),
            reinterpret_cast<elem_type*>(attn_probs), reinterpret_cast<float*>(lse), scale,
            n, dim
        );
    });
}

void blockmask_to_uint64(
    std::uintptr_t stream,
    std::uintptr_t blockmask,
    std::uintptr_t result,
    int batch_size,
    int last_dim_size,
    int n_uint64_per_row
) {
    blockmask_to_uint64_func(
        reinterpret_cast<cudaStream_t>(stream),
        reinterpret_cast<const bool*>(blockmask),
        reinterpret_cast<uint64_t*>(result),
        batch_size,
        last_dim_size,
        n_uint64_per_row
    );
}

void topk_to_uint64(
    std::uintptr_t stream,
    std::uintptr_t topk_idx,
    std::uintptr_t result,
    int batch_size,
    int k,
    int k_blocks,
    int n_uint64_per_row
) {
    topk_to_uint64_func(
        reinterpret_cast<cudaStream_t>(stream),
        reinterpret_cast<const int*>(topk_idx),
        reinterpret_cast<uint64_t*>(result),
        batch_size,
        k,
        k_blocks,
        n_uint64_per_row
    );
}

void uint64_to_bool(
    std::uintptr_t stream,
    std::uintptr_t input,
    std::uintptr_t result,
    int batch_size,
    int last_dim_size,
    int n_uint64_per_row
) {
    uint64_to_bool_func(
        reinterpret_cast<cudaStream_t>(stream),
        reinterpret_cast<const uint64_t*>(input),
        reinterpret_cast<bool*>(result),
        batch_size,
        last_dim_size,
        n_uint64_per_row
    );
}

void max_pooling_1d(
    std::uintptr_t stream,
    std::uintptr_t input,
    std::uintptr_t output,
    int dtype,
    int num_heads,
    int q_len,
    int k_len,
    int out_len,
    int cache_len,
    int kernel_size,
    int stride,
    int padding,
    int block_size,
    int local_blocks,
    int init_blocks
) {
    DTYPE_SWITCH(dtype, [&] {
        max_pooling_1d_func<elem_type>(
            reinterpret_cast<cudaStream_t>(stream),
            reinterpret_cast<const elem_type*>(input),
            reinterpret_cast<elem_type*>(output),
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
            init_blocks
        );
    });
}

PYBIND11_MODULE(C, m) {
    m.doc() = "InfLLM V2 CUDA Implementation with FlashAttention";
    
    // Original functions
    m.def("topk", &topk, "Topk func");
    m.def("get_probs", &get_probs, "Get probs func");
    m.def("blockmask_to_uint64", &blockmask_to_uint64, "Convert boolean mask to uint64 representation");
    m.def("topk_to_uint64", &topk_to_uint64, "Convert topk indices directly to uint64 representation");
    m.def("uint64_to_bool", &uint64_to_bool, "Convert uint64 representation back to boolean mask");
    m.def("max_pooling_1d", &max_pooling_1d, "Max pooling 1d func");
    
    // Flash Attention functions
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable length)");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("varlen_bwd", &mha_varlen_bwd, "Backward pass (variable length)");
    m.def("fwd_kvcache", &mha_fwd_kvcache, "Forward pass, with KV-cache");
    m.def("fwd_block", &mha_fwd_block, "Forward pass, with blockmask");
    m.def("bwd_block", &mha_bwd_block, "Backward pass, with blockmask");
    m.def("fwd_block_kvcache", &fwd_block_kvcache, "Forward pass, with blockmask and KV-cache");
} 