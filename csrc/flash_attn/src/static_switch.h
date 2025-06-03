// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    using elem_type = cutlass::bfloat16_t;   \
    return __VA_ARGS__();                    \
  }()

#define FWD_HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
    constexpr static int kHeadDim = 128;   \
    return __VA_ARGS__();                  \
  }()

#define FWD_BLOCK_HEADDIM_SWITCH(HEADDIM, ...) \
  [&] {                                        \
    constexpr static int kHeadDim = 128;       \
    return __VA_ARGS__();                      \
  }()

#define BWD_HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
    constexpr static int kHeadDim = 128;   \
    return __VA_ARGS__();                  \
  }()

#define BWD_BLOCK_HEADDIM_SWITCH(HEADDIM, ...) \
  [&] {                                        \
    constexpr static int kHeadDim = 128;       \
    return __VA_ARGS__();                      \
  }()
  
