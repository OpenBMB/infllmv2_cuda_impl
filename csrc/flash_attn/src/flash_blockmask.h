/******************************************************************************
 * Copyright (c) 2024, Junxian Guo.
 ******************************************************************************/

#pragma once

namespace flash {

class fwdIteratorBase{
};


// ////////////////////////////////////////////////////////////////////////////////////////////////////
class fwdStreaming: public fwdIteratorBase{
    public:
    template<typename Params, typename BlockInfo>
    __device__ fwdStreaming(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int n_block_min, int n_block_max) {//row first
        this -> row_factor = params.m_block_dim / kBlockM;
        this -> col_factor = params.n_block_dim / kBlockN;
        this -> sink_block_num = params.streaming_info[head_idx * 2] * col_factor;
        this -> local_block_num = params.streaming_info[head_idx * 2 + 1] * col_factor;
        this -> m_block_dim = params.m_block_dim;
        this -> n_block_dim = params.n_block_dim;
        this -> mask_type = head_idx;
        this -> n_block_min = n_block_min;
        this -> n_block_max = n_block_max;
        int act_k = binfo.actual_seqlen_k;
        int act_q = binfo.actual_seqlen_q;
        bool causal = params.is_causal;
        if (causal){
            int start_row_idx = max(int((act_q-act_k)/m_block_dim), 0);
            this -> start_block_val = (cute::ceil_div(max(act_k - act_q, 0), n_block_dim) + 1 + loop_step_idx/row_factor - start_row_idx) * col_factor;
        }else{
            this -> start_block_val = max(cute::ceil_div(n_block_max * kBlockN, n_block_dim) * col_factor, 0);
        };
        this -> no_gap = start_block_val - n_block_min < sink_block_num + local_block_num;
        this -> max_block_idx = min(sink_block_num + local_block_num, start_block_val - n_block_min);

        assert(mask_type < 0);
        assert(params.m_block_dim % kBlockM == 0);
        assert(params.n_block_dim % kBlockN == 0);
    };

    __device__ int mask_val(int block_col_idx) const {
        if (block_col_idx > max_block_idx || block_col_idx < 0){
            return -1;
        };
        int ret = 0;
        if (no_gap){
            ret = start_block_val - 1 - block_col_idx;
            return ret >= n_block_min ? ret : -1;
        }else{
            if (block_col_idx < local_block_num){
                return start_block_val - 1 - block_col_idx;
            }else{
                ret = sink_block_num - 1 - (block_col_idx - local_block_num);
                return ret >= n_block_min ? ret : -1;
            };
        };
    };

    __device__ int max_no_larger(int target) const {
        if(max_block_idx == 0){
            return -1;
        };
        int left = 0;
        int right = max_block_idx - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (mask_val(mid) > target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            };
        };
        return (left < max_block_idx && mask_val(left) <= target) ? left : -1;
    };

    int sink_block_num, local_block_num;
    int start_block_val;
    bool no_gap;
    
    int max_block_idx;
    int m_block_dim, n_block_dim;
    int mask_type;
    int n_block_min, n_block_max;
    int row_factor, col_factor;
};


class fwdExactStreaming: public fwdIteratorBase{
    public:
    template<typename Params, typename BlockInfo>
    __device__ fwdExactStreaming(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int n_block_min, int n_block_max) {//row first
        this -> row_factor = params.m_block_dim / kBlockM;
        this -> col_factor = params.n_block_dim / kBlockN;
        int sink_num = params.streaming_info[head_idx * 2];
        int local_num = params.streaming_info[head_idx * 2 + 1];
        this -> m_block_dim = params.m_block_dim;
        this -> n_block_dim = params.n_block_dim;
        this -> sink_block_num = cute::ceil_div(sink_num, n_block_dim) * col_factor;
        this -> local_block_num = (cute::ceil_div(local_num, n_block_dim)+2) * col_factor;

        
        
        this -> mask_type = head_idx;
        this -> n_block_min = n_block_min;
        this -> n_block_max = n_block_max;
        int act_k = binfo.actual_seqlen_k;
        int act_q = binfo.actual_seqlen_q;
        bool causal = params.is_causal;
        if (causal){
            int start_row_idx = max(int((act_q-act_k)/m_block_dim), 0);
            this -> start_block_val = (cute::ceil_div(max(act_k - act_q, 0), n_block_dim) + 1 + loop_step_idx/row_factor - start_row_idx) * col_factor;
        }else{
            this -> start_block_val = max(cute::ceil_div(n_block_max * kBlockN, n_block_dim) * col_factor, 0);
        };
        this -> no_gap = start_block_val - n_block_min < sink_block_num + local_block_num;
        this -> max_block_idx = min(sink_block_num + local_block_num, start_block_val - n_block_min);

        assert(mask_type < 0);
        assert(params.m_block_dim % kBlockM == 0);
        assert(params.n_block_dim % kBlockN == 0);
    };

    __device__ int mask_val(int block_col_idx) const {
        if (block_col_idx > max_block_idx || block_col_idx < 0){
            return -1;
        };
        int ret = 0;
        if (no_gap){
            ret = start_block_val - 1 - block_col_idx;
            return ret >= n_block_min ? ret : -1;
        }else{
            if (block_col_idx < local_block_num){
                return start_block_val - 1 - block_col_idx;
            }else{
                ret = sink_block_num - 1 - (block_col_idx - local_block_num);
                return ret >= n_block_min ? ret : -1;
            };
        };
    };

    __device__ int max_no_larger(int target) const {
        if(max_block_idx == 0){
            return -1;
        };
        int left = 0;
        int right = max_block_idx - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (mask_val(mid) > target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            };
        };
        return (left < max_block_idx && mask_val(left) <= target) ? left : -1;
    };

    int sink_block_num, local_block_num;
    int start_block_val;
    bool no_gap;
    
    int max_block_idx;
    int m_block_dim, n_block_dim;
    int mask_type;
    int n_block_min, n_block_max;
    int row_factor, col_factor;
};

// ////////////////////////////////////////////////////////////////////////////////////////////////////

class fwdBlockmask: public fwdIteratorBase{
    public:
    template<typename Params, typename BlockInfo>
    __device__ fwdBlockmask(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int n_block_min, int n_block_max) {//row first
        this -> row_factor = params.m_block_dim / kBlockM;
        this -> col_factor = params.n_block_dim / kBlockN;
        this -> max_block_idx = cute::ceil_div(binfo.actual_seqlen_k, params.n_block_dim) * col_factor;
        this -> m_block_dim = params.m_block_dim;
        this -> n_block_dim = params.n_block_dim;
        this -> mask_type = head_idx;
        this -> n_block_min = n_block_min;
        this -> n_block_max = n_block_max;
        this -> block_window_size = params.block_window_size;
        this -> current_step = loop_step_idx;

        // assert(mask_type > 0);
        assert(params.m_block_dim % kBlockM == 0);
        assert(params.n_block_dim % kBlockN == 0);
        
        // Calculate the offset for the uint64 blockmask 
        const int blocks_per_uint64 = 64;  // 64 bits per uint64
        const int num_blocks_m = params.num_blocks_m;
        const int num_blocks_n = params.num_blocks_n;
        const int uint64_per_row = (num_blocks_n + blocks_per_uint64 - 1) / blocks_per_uint64;
        const int row_offset = binfo.blockmask_q_offset(m_block_dim, batch_idx);
        const int step_offset = int(loop_step_idx / row_factor);
        
        blockmask_ptr = params.blockmask + 
                        mask_type * params.num_blocks_m * uint64_per_row + 
                        row_offset * uint64_per_row +
                        step_offset * uint64_per_row;
        
        // Store the number of uint64 values per row for bit calculations
        this->uint64_per_row = uint64_per_row;
    };

    __device__ int max_no_larger(int target) const {
        if(max_block_idx == 0){
            return -1;
        };
        
        // Check if the target is within the window size range
        if (block_window_size > 0) {
            // Calculate the actual q block number using round_multiple
            int q_block_idx = current_step;
            int k_idx = target * n_block_dim;
            auto round_to_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
            if (k_idx >= q_block_idx - (block_window_size * n_block_dim) && k_idx <= round_to_multiple(q_block_idx, n_block_dim)){
                return target;
            }
        }
        
        const int blocks_per_uint64 = 64;  // 64位uint64
        int result = -1;
        
        // 目标值不能超过最大块索引
        target = min(target, max_block_idx - 1);
        
        // 计算相对于当前q_bit_position的实际位置
        int target_bit_pos = target;
        
        // 确定此块在哪个uint64中
        int uint64_offset = target_bit_pos / blocks_per_uint64;
        
        // 确定此块在uint64中的哪一位
        int bit_pos = target_bit_pos % blocks_per_uint64;
        
        // 创建一个掩码，保留target及更低位的所有位
        uint64_t mask = (1ULL << (bit_pos + 1)) - 1;
        
        // 检查当前uint64中target及以下的位
        uint64_t value = blockmask_ptr[uint64_offset] & mask;
        
        // 如果当前uint64中有设置的位
        if (value != 0) {
            // 找到最高位的1（即不大于target的最大设置位）
            int highest_bit = 63 - __clzll(value);  // __clzll计算前导0的数量
            result = highest_bit + (uint64_offset * blocks_per_uint64);
        } else {
            // 如果当前uint64中没有找到，检查更低的uint64块
            for (int i = uint64_offset - 1; i >= 0; i--) {
                value = blockmask_ptr[i];
                if (value != 0) {
                    // 找到最高位的1
                    int highest_bit = 63 - __clzll(value);
                    // 计算相对于q_bit_position的偏移
                    result = highest_bit + (i * blocks_per_uint64);
                    break;
                }
            }
        }
        
        // Return result, will be -1 if no bits were found
        return result;
    };

    uint64_t *blockmask_ptr;
    int row_offset;              // 行偏移量
    int blocks_per_uint64;       // 每个uint64包含的块数
    int uint64_per_row;          // 每行使用的uint64数量
    int max_block_idx;
    int m_block_dim, n_block_dim;
    int mask_type;
    int n_block_min, n_block_max;
    int row_factor, col_factor;
    int block_window_size;       // 新增：窗口大小参数
    int current_step;            // 新增：当前步骤索引
};

// ////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Is_streaming, bool Is_exact_streaming, bool Is_batched = false>   
class fwdIterator{};

template<>
struct fwdIterator<false, false, false>: public fwdBlockmask{
    template<typename Params, typename BlockInfo>
    __device__ fwdIterator(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int n_block_min, int n_block_max): fwdBlockmask(params, binfo, kBlockM, kBlockN, batch_idx, head_idx, loop_step_idx, n_block_min, n_block_max) {};
};

template<>
struct fwdIterator<true, false, false>: public fwdStreaming{
    template<typename Params, typename BlockInfo>
    __device__ fwdIterator(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int n_block_min, int n_block_max): fwdStreaming(params, binfo, kBlockM, kBlockN, batch_idx, head_idx, loop_step_idx, n_block_min, n_block_max) {};
};

template<>
struct fwdIterator<true, true, false>: public fwdExactStreaming{
    template<typename Params, typename BlockInfo>
    __device__ fwdIterator(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int n_block_min, int n_block_max): fwdExactStreaming(params, binfo, kBlockM, kBlockN, batch_idx, head_idx, loop_step_idx, n_block_min, n_block_max) {};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

class bwdIteratorBase{
};


struct bwdStreaming: public bwdIteratorBase{
    public:
    template<typename Params, typename BlockInfo>
    __device__ bwdStreaming(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int m_block_min, int m_block_max) {// col first
        this -> row_factor = params.m_block_dim / kBlockM;
        this -> col_factor = params.n_block_dim / kBlockN;
        
        this -> m_block_dim = params.m_block_dim;
        this -> n_block_dim = params.n_block_dim;
        this -> mask_type = head_idx;
        this -> m_block_min = m_block_min;
        this -> m_block_max = m_block_max;

        int mask_block_col = cute::ceil_div(loop_step_idx+1, col_factor);
        int sink = (this -> mask_type) < 0 ? params.streaming_info[head_idx * 2]: cute::ceil_div(binfo.actual_seqlen_k, this -> n_block_dim);
        int local = (this -> mask_type) < 0 ? params.streaming_info[head_idx * 2 + 1]: 0;
        this -> sink_block_num = sink * col_factor;
        this -> local_block_num = local * col_factor;
        int act_q = binfo.actual_seqlen_q;
        int act_k = binfo.actual_seqlen_k;
        bool causal = params.is_causal;

        if(mask_block_col <= sink){
            this -> start_block_val = m_block_max;
            this -> max_block_idx = m_block_max - m_block_min;
        }else{
            if (causal){
                int free_token_num = act_q - min(act_q, act_k - loop_step_idx * kBlockN);
                int end_mask_block_row_idx = free_token_num / params.m_block_dim;//zero based
                int num_mask_block_in_end_row = max(0, cute::ceil_div(act_k - act_q + (end_mask_block_row_idx + 1) * params.m_block_dim, params.n_block_dim));
                int local_col_mask_block_num = max(0, local - (num_mask_block_in_end_row - mask_block_col));
                if(local_col_mask_block_num > 0){
                    this -> start_block_val = min((end_mask_block_row_idx + local_col_mask_block_num) * row_factor, m_block_max);
                    this -> max_block_idx = min(local_col_mask_block_num * row_factor, m_block_max - m_block_min);
                }else{
                    this -> start_block_val = 0;
                    this -> max_block_idx = 0;
                };
            }else{
                int n_mask_block_col = max(cute::ceil_div(act_k, n_block_dim), 0);
                bool in_none_causal_local = !causal && mask_block_col <= n_mask_block_col && mask_block_col > n_mask_block_col - local;
                if(in_none_causal_local){
                    this -> start_block_val = m_block_max;
                    this -> max_block_idx = m_block_max - m_block_min;
                }else{
                    this -> start_block_val = 0;
                    this -> max_block_idx = 0;
                };
            };
        }
        
        // assert(mask_type <= 0); //for blocksparse, mask_type > 0; for streaming, mask_type < 0; for dense, mask_type = 0
        assert(params.m_block_dim % kBlockM == 0);
        assert(params.n_block_dim % kBlockN == 0);
    };

    __device__ int mask_val(int block_row_idx) const {
        if (block_row_idx > max_block_idx || block_row_idx < 0){
            return -1;
        };
        int ret = start_block_val - 1 - block_row_idx;
        return ret >= m_block_min ? ret : -1;
    };

    __device__ int max_no_larger(int target) const {
        if(max_block_idx == 0){
            return -1;
        };
        int left = 0;
        int right = max_block_idx - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (mask_val(mid) > target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            };
        };
        return (left < max_block_idx && mask_val(left) <= target) ? left : -1;
    };

    int sink_block_num, local_block_num;
    int start_block_val;

    int max_block_idx;
    int m_block_dim, n_block_dim;
    int mask_type;
    int m_block_min, m_block_max;
    int row_factor, col_factor;
};

struct bwdBlockmask: public bwdIteratorBase{
    public:
    template<typename Params, typename BlockInfo>
    __device__ bwdBlockmask(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int m_block_min, int m_block_max) {
        this -> row_factor = params.m_block_dim / kBlockM;
        this -> col_factor = params.n_block_dim / kBlockN;
        this -> max_block_idx = cute::ceil_div(binfo.actual_seqlen_q, params.m_block_dim) * row_factor;
        this -> m_block_dim = params.m_block_dim;
        this -> n_block_dim = params.n_block_dim;
        this -> mask_type = head_idx;
        this -> m_block_min = m_block_min;
        this -> m_block_max = m_block_max;
        this -> block_window_size = params.block_window_size;
        this -> current_step = loop_step_idx;
        
        // assert(mask_type > 0);
        assert(params.m_block_dim % kBlockM == 0);
        assert(params.n_block_dim % kBlockN == 0);

        // 计算块的基本信息
        const int blocks_per_uint64 = 64;  // 每个uint64可以存储64个块信息
        
        // 原始行块的索引起始位置（考虑批次位置）
        const int q_block_offset = binfo.blockmask_q_offset(m_block_dim, batch_idx);
        
        // 计算q_block_offset在uint64表示中的位置
        const int q_uint64_idx = q_block_offset / blocks_per_uint64;  // 确定在第几个uint64
        const int q_bit_position = q_block_offset % blocks_per_uint64; // 确定在uint64中的第几位
        
        // 列块的索引（循环步进位置）
        const int k_block_idx = loop_step_idx / col_factor;
        
        // 计算每行需要多少个uint64来表示
        const int num_blocks_m = params.num_blocks_m;
        const int uint64_per_row = (num_blocks_m + blocks_per_uint64 - 1) / blocks_per_uint64;
        
        // 确保这里用的是num_blocks_n而不是num_blocks_m，以匹配前向传播中的计算方式
        this->blockmask_ptr = params.blockmask + 
                            mask_type * params.num_blocks_n * uint64_per_row + 
                            k_block_idx * uint64_per_row +
                            q_uint64_idx;
        
        // 存储块在uint64中的位偏移
        this->q_bit_position = q_bit_position;
        
        // 存储每行使用的uint64数量，用于计算偏移
        this->uint64_per_row = uint64_per_row;
    };

    __device__ int max_no_larger(int target) const {
        if(max_block_idx == 0){
            return -1;
        };
        
        // 目标值不能超过最大块索引
        target = min(target, max_block_idx - 1);
        
        // 窗口计算结果
        int window_result = -1;
        
        // 检查窗口条件
        if (block_window_size > 0) {
            // 计算k的位置
            auto round_to_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
            int k_idx = current_step * n_block_dim;
            
            // 检查target是否在窗口内 (从k往右算)
            bool is_in_window = (k_idx >= target - (block_window_size * n_block_dim) && k_idx <= round_to_multiple(target, n_block_dim));
            if (is_in_window) {
                return target; // 如果在窗口内，直接返回target
            } else if (target > k_idx) {
                // 在k_idx右侧找到不大于target的最大值
                int right_boundary = k_idx + block_window_size * n_block_dim;
                window_result = min(target, right_boundary);
            }
        }
        
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

        // 返回blockmask结果和窗口结果的较大值
        return max(blockmask_result, window_result);
    };

    uint64_t *blockmask_ptr;
    int q_bit_position;          // 在第一个uint64中的位偏移
    int uint64_per_row;          // 每行使用的uint64数量
    int max_block_idx;
    int m_block_dim, n_block_dim;
    int mask_type;
    int m_block_min, m_block_max;
    int row_factor, col_factor;
    int block_window_size;       // 新增：窗口大小参数
    int current_step;            // 新增：当前步骤索引
};



template<bool Is_streaming>   
class bwdIterator{};

template<>
struct bwdIterator<false>: public bwdBlockmask{
    template<typename Params, typename BlockInfo>
    __device__ bwdIterator(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int m_block_min, int m_block_max): bwdBlockmask(params, binfo, kBlockM, kBlockN, batch_idx, head_idx, loop_step_idx, m_block_min, m_block_max) {};
};

template<>
struct bwdIterator<true>: public bwdStreaming{
    template<typename Params, typename BlockInfo>
    __device__ bwdIterator(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int m_block_min, int m_block_max): bwdStreaming(params, binfo, kBlockM, kBlockN, batch_idx, head_idx, loop_step_idx, m_block_min, m_block_max) {};
};


////////////////////////////////////////////////////////////////////////////////////////////////////

class fwdBlockmaskBatch: public fwdIteratorBase{
    public:
    template<typename Params, typename BlockInfo>
    __device__ fwdBlockmaskBatch(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int n_block_min, int n_block_max) {//row first
        this -> row_factor = params.m_block_dim / kBlockM;
        this -> col_factor = params.n_block_dim / kBlockN;
        this -> actual_seqlen_k = binfo.actual_seqlen_k;
        this -> max_block_idx = cute::ceil_div(binfo.actual_seqlen_k, params.n_block_dim) * col_factor;
        this -> m_block_dim = params.m_block_dim;
        this -> n_block_dim = params.n_block_dim;
        this -> mask_type = head_idx;
        this -> n_block_min = n_block_min;
        this -> n_block_max = n_block_max;
        this -> block_window_size = params.block_window_size;
        this -> current_step = loop_step_idx;
        this -> batch_idx = batch_idx;  // Store batch_idx for debugging
        this -> head_idx = head_idx;  // Store head_idx for debugging

        // assert(mask_type > 0);
        assert(params.m_block_dim % kBlockM == 0);
        assert(params.n_block_dim % kBlockN == 0);
        
        // Calculate the offset for the uint64 blockmask 
        const int blocks_per_uint64 = 64;  // 64 bits per uint64
        const int num_blocks_m = params.num_blocks_m;
        const int num_blocks_n = params.num_blocks_n;
        const int uint64_per_row = (num_blocks_n + blocks_per_uint64 - 1) / blocks_per_uint64;
        const int row_offset = binfo.blockmask_q_offset(m_block_dim, batch_idx);
        const int step_offset = int(loop_step_idx / row_factor);
        
        // Store more diagnostic information
        this->row_offset = row_offset;
        this->step_offset = step_offset;
        // Modified blockmask_ptr calculation for batched inputs
        blockmask_ptr = params.blockmask + 
                        (batch_idx * params.num_blocksparse_heads + mask_type) * params.num_blocks_m * uint64_per_row + 
                        step_offset * uint64_per_row;
        
        // Store the number of uint64 values per row for bit calculations
        this->uint64_per_row = uint64_per_row;
        // if(threadIdx.x == 0){
        //     printf("blockmask_ptr: %p, max_block_idx: %d, actual_seqlen_k: %d, uint64_per_row: %d, row_offset: %d, step_offset: %d\n", blockmask_ptr, max_block_idx, actual_seqlen_k, uint64_per_row, row_offset, step_offset);
        // }
        
    };

    __device__ int max_no_larger(int target) const {
        if(max_block_idx == 0){
            return -1;
        };
        
        // Check if the target is within the window size range
        if (block_window_size > 0) {
            // Calculate the actual q block number using round_multiple
            int q_block_idx = current_step + actual_seqlen_k;
            int k_idx = target * n_block_dim;
            auto round_to_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
            
            if (k_idx >= q_block_idx - (block_window_size * n_block_dim) && k_idx <= round_to_multiple(q_block_idx, n_block_dim)){
                // if(threadIdx.x == 0){
                //     printf("max no larger within window: batch_idx: %d, head_idx: %d, target: %d, k_idx: %d, q_block_idx: %d, block_window_size: %d\n", batch_idx, head_idx, target, k_idx, q_block_idx, block_window_size);
                // }
                return target;
            }
        }

        const int blocks_per_uint64 = 64;  // 64位uint64
        int result = -1;
        
        // 目标值不能超过最大块索引
        target = min(target, max_block_idx - 1);
        // if(threadIdx.x == 0 && target != original_target){
        //     printf("max no larger: target min by max_block_idx: batch_idx: %d, head_idx: %d, target: %d, max_block_idx: %d\n", batch_idx, head_idx, target, max_block_idx);
        // }
        
        // 计算相对于当前q_bit_position的实际位置
        int target_bit_pos = target;
        
        // 确定此块在哪个uint64中
        int uint64_offset = target_bit_pos / blocks_per_uint64;
        
        // 确定此块在uint64中的哪一位
        int bit_pos = target_bit_pos % blocks_per_uint64;
        
        // 创建一个掩码，保留target及更低位的所有位
        uint64_t mask = (1ULL << (bit_pos + 1)) - 1;
        
        // 检查当前uint64中target及以下的位
        uint64_t value = blockmask_ptr[uint64_offset] & mask;
        
        // 如果当前uint64中有设置的位
        if (value != 0) {
            // 找到最高位的1（即不大于target的最大设置位）
            int highest_bit = 63 - __clzll(value);  // __clzll计算前导0的数量
            result = highest_bit + (uint64_offset * blocks_per_uint64);
        } else {
            // 如果当前uint64中没有找到，检查更低的uint64块
            for (int i = uint64_offset - 1; i >= 0; i--) {
                value = blockmask_ptr[i];
                if (value != 0) {
                    // 找到最高位的1
                    int highest_bit = 63 - __clzll(value);
                    // 计算相对于q_bit_position的偏移
                    result = highest_bit + (i * blocks_per_uint64);
                    break;
                }
            }
        }
        
        // if(threadIdx.x == 0){
        //     printf("max no larger: batch_idx: %d, head_idx: %d, result: %d, original_target: %d\n", batch_idx, head_idx, result, original_target);
        // }
        
        // Return -1 if no valid bit was found
        return result;
    };

    uint64_t *blockmask_ptr;
    int row_offset;              // 行偏移量
    int step_offset;             // 步骤偏移量
    int blocks_per_uint64;       // 每个uint64包含的块数
    int uint64_per_row;          // 每行使用的uint64数量
    int max_block_idx;
    int m_block_dim, n_block_dim;
    int mask_type;
    int n_block_min, n_block_max;
    int row_factor, col_factor;
    int block_window_size;       // 新增：窗口大小参数
    int current_step;            // 新增：当前步骤索引
    int batch_idx;               // 新增：批次索引，用于调试
    int head_idx;                // 新增：头索引，用于调试
    int actual_seqlen_k;         // 新增：实际序列长度，用于调试
};

template<>
struct fwdIterator<false, false, true>: public fwdBlockmaskBatch{
    template<typename Params, typename BlockInfo>
    __device__ fwdIterator(const Params &params, const BlockInfo &binfo, const int kBlockM, const int kBlockN, const int batch_idx, const int head_idx, const int loop_step_idx, int n_block_min, int n_block_max): fwdBlockmaskBatch(params, binfo, kBlockM, kBlockN, batch_idx, head_idx, loop_step_idx, n_block_min, n_block_max) {};
};

}  // namespace flash
