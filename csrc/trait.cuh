#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<__half> {
    using half2 = __half2;

    static __inline__ int type_code() {
        return 0;
    }

    static __host__ __device__ __inline__ __half inf() { 
        return __float2half(INFINITY);
    }
};

template <>
struct TypeTraits<__nv_bfloat16> {
    using half2 = __nv_bfloat162;

    static __inline__ int type_code() {
        return 1;
    }

    static __host__ __device__ __inline__ __nv_bfloat16 inf() { 
        return __float2bfloat16(INFINITY);
    }
};
