#include <pybind11/pybind11.h>
#include "topk.cuh"
#include "get_probs.cuh"
#include "blockmask.cuh"
#include "topk_to_uint64.cuh"
#include "uint64_to_bool.cuh"
#include "max_pooling_1d.cuh"

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
    m.def("topk", &topk, "Topk func");
    m.def("get_probs", &get_probs, "Get probs func");
    m.def("blockmask_to_uint64", &blockmask_to_uint64, "Convert boolean mask to uint64 representation");
    m.def("topk_to_uint64", &topk_to_uint64, "Convert topk indices directly to uint64 representation");
    m.def("uint64_to_bool", &uint64_to_bool, "Convert uint64 representation back to boolean mask");
    m.def("max_pooling_1d", &max_pooling_1d, "Max pooling 1d func");
} 