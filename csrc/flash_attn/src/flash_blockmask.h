#pragma once

namespace flash {

class fwdIterator{
    public:
    template<typename Params, typename BlockInfo>
    __device__ fwdIterator(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int n_block_min, int n_block_max) {//row first
        if (params.blockmask == nullptr) {
            blockmask_ptr = nullptr;
            return;
        }
        this->cache_seqlen_k = binfo.actual_seqlen_k - binfo.actual_seqlen_q / params.m_block_dim;
        this->max_block_idx = cute::ceil_div(binfo.actual_seqlen_k, params.n_block_dim);
        this->m_block_dim = params.m_block_dim;
        this->n_block_dim = params.n_block_dim;
        this->n_block_min = n_block_min;
        this->n_block_max = n_block_max;
        this->batch_idx = batch_idx;  // Store batch_idx for debugging
        this->head_idx = head_idx;

        // Calculate the offset for the uint64 blockmask 
        const int num_blocks_m = params.num_blocks_m;
        const int num_blocks_n = params.num_blocks_n;
        const int uint64_per_row = (num_blocks_n + 64 - 1) / 64;
        const int row_offset = params.cu_seqlens_q != nullptr ? binfo.blockmask_q_offset(m_block_dim, batch_idx) : batch_idx * params.num_k_heads * params.num_blocks_m;

        blockmask_ptr = params.blockmask + 
                        head_idx * params.num_blocks_m * uint64_per_row + 
                        row_offset * uint64_per_row +
                        loop_step_idx * uint64_per_row;

        const int q_block_idx = loop_step_idx + cache_seqlen_k;
        this->k_window_right = q_block_idx / n_block_dim;
        this->k_window_left = this->k_window_right - params.block_window_size + 1;
    }

    __device__ int max_no_larger(int target) const {
        if (blockmask_ptr == nullptr) return target;
        if(max_block_idx == 0){
            return -1;
        };
        
        if (k_window_left <= target && target <= k_window_right){
            return target;
        }
        
        // 目标值不能超过最大块索引
        target = min(target, max_block_idx - 1);
        
        // 计算相对于当前q_bit_position的实际位置
        int target_bit_pos = target;
        
        // 确定此块在哪个uint64中
        int uint64_offset = target_bit_pos / 64;
        
        // 确定此块在uint64中的哪一位
        int bit_pos = target_bit_pos % 64;
        
        // 创建一个掩码，保留target及更低位的所有位
        uint64_t mask = bit_pos != 63 ? (1ULL << (bit_pos + 1)) - 1 : 0xFFFFFFFFFFFFFFFFULL;
        
        // 检查当前uint64中target及以下的位
        uint64_t value = blockmask_ptr[uint64_offset] & mask;
        
        // 如果当前uint64中有设置的位
        int result = -1;
        if (value != 0) {
            // 找到最高位的1（即不大于target的最大设置位）
            int highest_bit = 63 - __clzll(value);  // __clzll计算前导0的数量
            result = highest_bit + (uint64_offset * 64);
        } else {
            // 如果当前uint64中没有找到，检查更低的uint64块
            for (int i = uint64_offset - 1; i >= 0; i--) {
                value = blockmask_ptr[i];
                if (value != 0) {
                    // 找到最高位的1
                    int highest_bit = 63 - __clzll(value);
                    // 计算相对于q_bit_position的偏移
                    result = highest_bit + (i * 64);
                    break;
                }
            }
        }

        if (target > k_window_right && result <= k_window_right && k_window_left <= k_window_right)
            return k_window_right;
        
        // 没有找到设置位
        return result;
    }

    uint64_t *blockmask_ptr;
    int row_offset; // 行偏移量
    int uint64_per_row;          // 每行使用的uint64数量
    int cache_seqlen_k;
    int max_block_idx;
    int m_block_dim, n_block_dim;
    int n_block_min, n_block_max;
    int batch_idx, head_idx;
    int k_window_left, k_window_right;
};

class bwdIterator{
    public:
    template<typename Params, typename BlockInfo>
    __device__ bwdIterator(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int m_block_min, int m_block_max) {
        if (params.blockmask == nullptr) {
            blockmask_ptr = nullptr;
            return;
        }
        // For backward, we iterate over K/V dimension, so max_block_idx is based on Q length
        this->max_block_idx = cute::ceil_div(binfo.actual_seqlen_q, params.m_block_dim);
        this->m_block_dim = params.m_block_dim;
        this->n_block_dim = params.n_block_dim;
        this->m_block_min = m_block_min;
        this->m_block_max = m_block_max;
        this->batch_idx = batch_idx;
        this->head_idx = head_idx;
        this->loop_step_idx = loop_step_idx;

        // Calculate the offset for the uint64 blockmask (transposed access pattern)
        const int num_blocks_m = params.num_blocks_m;
        const int num_blocks_n = params.num_blocks_n;
        const int uint64_per_row = (num_blocks_m + 64 - 1) / 64;  // For backward, each column has num_blocks_m bits
        
        // For backward pass, we need to access column loop_step_idx of the blockmask
        // Since blockmask is stored row-major, we need to gather from multiple rows
        const int k_block_idx = loop_step_idx;
        
        // Calculate Q block offset for batch
        const int q_block_offset = params.cu_seqlens_q != nullptr ? binfo.blockmask_q_offset(m_block_dim, batch_idx) : batch_idx * params.num_k_heads * params.num_blocks_m;
        
        // Calculate which uint64 contains the Q offset bits
        const int q_uint64_idx = q_block_offset / 64;
        const int q_bit_position = q_block_offset % 64;
        
        // For backward, blockmask access pattern is transposed
        blockmask_ptr = params.blockmask + 
                        head_idx * params.num_blocks_n * uint64_per_row + 
                        k_block_idx * uint64_per_row +
                        q_uint64_idx;

        this->q_bit_position = q_bit_position;
        this->uint64_per_row = uint64_per_row;
        
        // Window calculation for backward (based on current K block position)
        this->q_window_right = k_block_idx * n_block_dim / m_block_dim;
        this->q_window_left = this->q_window_right - params.block_window_size + 1;
        this->block_window_size = params.block_window_size;
    }

    __device__ int max_no_larger(int target) const {
        if (blockmask_ptr == nullptr) return target;
        if(max_block_idx == 0){
            return -1;
        };
        
        // // Window check for backward pass
        // if (block_window_size > 0) {
        //     // Calculate k position for current step
        //     int k_idx = loop_step_idx * n_block_dim;
        //     int q_idx = target * m_block_dim;
        //     auto round_to_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
            
        //     // Check if target is within window (from k position)
        //     bool is_in_window = (k_idx >= q_idx - (block_window_size * n_block_dim) && 
        //                         k_idx <= round_to_multiple(q_idx, n_block_dim));
        //     if (is_in_window) {
        //         return target;
        //     }
        // }
        
        // 目标值不能超过最大块索引
        target = min(target, max_block_idx - 1);
        
        // 计算包含q_bit_position偏移的实际位置
        int target_bit_pos = q_bit_position + target;
        
        // 确定此块在哪个uint64中
        int uint64_offset = target_bit_pos / 64;
        
        // 确定此块在uint64中的哪一位
        int bit_pos = target_bit_pos % 64;
        
        // 创建一个掩码，保留target及更低位的所有位
        uint64_t mask = bit_pos != 63 ? (1ULL << (bit_pos + 1)) - 1 : 0xFFFFFFFFFFFFFFFFULL;
        
        // 检查当前uint64中target及以下的位
        uint64_t value = blockmask_ptr[uint64_offset] & mask;
        
        // 如果当前uint64中有设置的位
        int result = -1;
        if (value != 0) {
            // 找到最高位的1（即不大于target的最大设置位）
            int highest_bit = 63 - __clzll(value);  // __clzll计算前导0的数量
            result = highest_bit + (uint64_offset * 64) - q_bit_position;
        } else {
            // 如果当前uint64中没有找到，检查更低的uint64块
            for (int i = uint64_offset - 1; i >= 0; i--) {
                value = blockmask_ptr[i];
                if (value != 0) {
                    // 找到最高位的1
                    int highest_bit = 63 - __clzll(value);
                    result = highest_bit + (i * 64) - q_bit_position;
                    break;
                }
            }
        }
        
        // 没有找到设置位
        return result;
    }

    uint64_t *blockmask_ptr;
    int q_bit_position;
    int uint64_per_row;
    int max_block_idx;
    int m_block_dim, n_block_dim;
    int m_block_min, m_block_max;
    int batch_idx, head_idx;
    int loop_step_idx;
    int q_window_left, q_window_right;
    int block_window_size;
};

}  // namespace flash