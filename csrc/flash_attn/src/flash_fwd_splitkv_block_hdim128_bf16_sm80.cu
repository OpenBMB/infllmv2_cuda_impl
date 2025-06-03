#include "flash_fwd_launch_template.h"
#include "static_switch.h"
// Explicit instantiation
template void run_mha_fwd_splitkv_block_dispatch<cutlass::bfloat16_t, 128>(Flash_fwd_params &params, cudaStream_t stream); 