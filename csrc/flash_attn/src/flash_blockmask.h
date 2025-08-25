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
    }

    __device__ int max_no_larger(int target) const {
        if (blockmask_ptr == nullptr) return target;
        if(max_block_idx == 0){
            return -1;
        };
        
        
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
};

class bwdIterator{
    public:
    template<typename Params, typename BlockInfo>
    __device__ bwdIterator(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int m_block_min, int m_block_max) {
        if (params.blockmask == nullptr) {
            blockmask_ptr = nullptr;
            return;
        }
        this -> max_block_idx = cute::ceil_div(binfo.actual_seqlen_q, params.m_block_dim);
        this -> m_block_dim = params.m_block_dim;
        this -> n_block_dim = params.n_block_dim;
        this -> m_block_min = m_block_min;
        this -> m_block_max = m_block_max;

        this -> loop_step_idx = loop_step_idx;
            

        // 计算块的基本信息
        const int blocks_per_uint64 = 64;  // 每个uint64可以存储64个块信息
        
        // 原始行块的索引起始位置（考虑批次位置）
        const int q_block_offset = binfo.blockmask_q_offset(m_block_dim, batch_idx);
        
        // 计算q_block_offset在uint64表示中的位置
        const int q_uint64_idx = q_block_offset / blocks_per_uint64;  // 确定在第几个uint64
        const int q_bit_position = q_block_offset % blocks_per_uint64; // 确定在uint64中的第几位
        
        // 列块的索引（循环步进位置）
        const int k_block_idx = loop_step_idx;
        
        // 计算每行需要多少个uint64来表示
        const int num_blocks_m = params.num_blocks_m;
        const int uint64_per_row = (num_blocks_m + blocks_per_uint64 - 1) / blocks_per_uint64;
        
        // 确保这里用的是num_blocks_n而不是num_blocks_m，以匹配前向传播中的计算方式
        this->blockmask_ptr = params.blockmask + 
                            head_idx * params.num_blocks_n * uint64_per_row + 
                            k_block_idx * uint64_per_row +
                            q_uint64_idx;
        
        // 存储块在uint64中的位偏移
        this->q_bit_position = q_bit_position;
        
        // 存储每行使用的uint64数量，用于计算偏移
        this->uint64_per_row = uint64_per_row;
    };

    __device__ int max_no_larger(int target) const {
        if (blockmask_ptr == nullptr) {
            return target;
        }
        if(max_block_idx == 0){
            return -1;
        };
        
        // 目标值不能超过最大块索引
        target = min(target, max_block_idx - 1);
        
        
        // 接下来检查blockmask
        const int blocks_per_uint64 = 64;
        int target_bit_pos = q_bit_position + target;
        
        // 确定此块在哪个uint64中
        int uint64_offset = target_bit_pos / blocks_per_uint64;
        
        // 确定此块在uint64中的哪一位
        int bit_pos = target_bit_pos % blocks_per_uint64;
        
        // 创建一个掩码，保留target及更低位的所有位
        uint64_t mask = (1ULL << (bit_pos + 1)) - 1;
        
        // 检查当前uint64中target及以下的位
        uint64_t value = blockmask_ptr[uint64_offset] & mask;
        int blockmask_result = -1;
        
        if (value != 0) {
            // 找到最高位的1（即不大于target的最大设置位）
            int highest_bit = 63 - __clzll(value);  // __clzll计算前导0的数量
            blockmask_result = highest_bit + (uint64_offset * blocks_per_uint64) - q_bit_position;
        } else {
            // 如果当前uint64中没有找到，检查更低的uint64块
            for (int i = uint64_offset - 1; i >= 0; i--) {
                value = blockmask_ptr[i];
                if (value != 0) {
                    // 找到最高位的1
                    int highest_bit = 63 - __clzll(value);
                    // 计算相对于q_bit_position的偏移
                    blockmask_result = highest_bit + (i * blocks_per_uint64) - q_bit_position;
                    break;
                }
            }
        }

        // 返回blockmask结果
        return blockmask_result;
    };


    uint64_t *blockmask_ptr;
    int q_bit_position;
    int uint64_per_row;
    int max_block_idx;
    int m_block_dim, n_block_dim;
    int m_block_min, m_block_max;
    int batch_idx, head_idx;
    int loop_step_idx;
};

}  // namespace flash