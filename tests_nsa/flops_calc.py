#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
稀疏注意力机制的计算复杂度分析工具

输入：
    seqlen: 序列长度
    stage1_block_size: 第一阶段块大小
    stage1_stride: 第一阶段步长
    stage2_block_size: 第二阶段块大小
    stage2_selected_block_num: 第二阶段选择的块数量
    stage2_window_size: 第二阶段窗口大小
    head_dim: 注意力头维度
    heads: 注意力头数量
    kv_heads: KV头数量（用于GQA）

输出：
    stage1_flops: 第一阶段浮点运算次数
    stage2_flops: 第二阶段浮点运算次数
    total_flops: 总浮点运算次数
    flops_ratio: 第一阶段与第二阶段的比例
    sparsity: 稀疏度（省略计算的比例）
    efficiency_ratio: 相对于传统注意力的效率比（越大越好）
    standard_flops: 传统注意力计算量
"""

import argparse
import sys
from typing import Dict, List, Tuple, Union, Optional

# 导入pandas用于Excel输出，可选依赖
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def calculate_flops(seqlen: int, stage1_block_size: int, stage1_stride: int, 
                    stage2_block_size: int, stage2_selected_block_num: int, 
                    stage2_window_size: int, head_dim: int = 128, 
                    heads: int = 32, kv_heads: Optional[int] = None) -> Tuple[float, float, float, float, float, float, float]:
    """
    计算两个阶段的计算复杂度和稀疏性
    
    参数:
        seqlen (int): 序列长度
        stage1_block_size (int): 第一阶段块大小
        stage1_stride (int): 第一阶段步长
        stage2_block_size (int): 第二阶段块大小
        stage2_selected_block_num (int): 第二阶段选择的块数量
        stage2_window_size (int): 第二阶段窗口大小
        head_dim (int, optional): 注意力头维度，默认为128
        heads (int, optional): 注意力头数量，默认为32
        kv_heads (int, optional): KV头数量，用于GQA，默认为None（等于heads）
        
    返回:
        tuple: (stage1_flops, stage2_flops, total_flops, flops_ratio, sparsity, efficiency_ratio, standard_flops)
    """
    # 如果未指定kv_heads，则默认等于heads
    if kv_heads is None:
        kv_heads = heads
        
    # GQA参数，每个Q头对应多少个KV头
    kv_ratio = kv_heads / heads
    
    # 传统注意力的FLOPs: QK计算 + 矩阵乘以V (考虑到GQA)
    # QK计算: seqlen^2 * head_dim * heads
    # 矩阵乘以V: seqlen^2 * head_dim * kv_heads
    standard_flops = seqlen * seqlen * head_dim * (heads + kv_heads)
    
    # 计算第一阶段的Flops (稀疏键值搜索)
    # 考虑了head_dim和heads，以及GQA
    stage1_flops = (seqlen * seqlen * head_dim * (heads + kv_heads)) / stage1_stride
    
    # 计算第二阶段的Flops (细粒度注意力计算)
    # 包括选择的块和局部窗口，考虑GQA
    stage2_tokens = (stage2_block_size * stage2_selected_block_num + stage2_window_size)
    stage2_flops = seqlen * stage2_tokens * head_dim * (heads + kv_heads)
    
    # 计算总Flops
    total_flops = stage1_flops + stage2_flops
    
    # 计算两者Flops的比值
    flops_ratio = stage1_flops / stage2_flops if stage2_flops > 0 else float('inf')
    
    # 计算稀疏度（省略计算的比例）
    sparsity = (seqlen - stage2_tokens) / seqlen
    
    # 计算效率比（与传统注意力机制相比）
    efficiency_ratio = standard_flops / total_flops if total_flops > 0 else float('inf')
    
    return stage1_flops, stage2_flops, total_flops, flops_ratio, sparsity, efficiency_ratio, standard_flops


def validate_params(seqlen: int, stage1_block_size: int, stage1_stride: int, 
                    stage2_block_size: int, stage2_selected_block_num: int, 
                    stage2_window_size: int, kv_heads: Optional[int] = None,
                    heads: int = 32) -> List[str]:
    """验证输入参数的合理性，返回警告信息列表"""
    warnings = []
    
    # 检查序列长度是否太小
    if seqlen < 1024:
        warnings.append(f"序列长度({seqlen})较小，稀疏注意力可能不会带来明显优势")
    
    # 检查块大小和步长的关系
    if stage1_stride > stage1_block_size:
        warnings.append(f"第一阶段步长({stage1_stride})大于块大小({stage1_block_size})，可能会漏掉重要信息")
    
    # 检查第二阶段选择的块数量
    max_blocks = seqlen // stage2_block_size
    if stage2_selected_block_num > max_blocks:
        warnings.append(f"第二阶段选择的块数量({stage2_selected_block_num})超过了最大可能的块数量({max_blocks})")
    
    # 检查第二阶段窗口大小
    if stage2_window_size > seqlen:
        warnings.append(f"第二阶段窗口大小({stage2_window_size})超过了序列长度({seqlen})")
    
    # 检查总体稀疏性
    attended_tokens = stage2_block_size * stage2_selected_block_num + stage2_window_size
    if attended_tokens > seqlen:
        warnings.append(f"注意力覆盖的token数({attended_tokens})超过了序列长度({seqlen})，稀疏性计算可能不准确")
    
    # 检查极端稀疏性
    sparsity = (seqlen - attended_tokens) / seqlen
    if sparsity < 0.5:
        warnings.append(f"稀疏度较低({sparsity:.2%})，可能无法充分发挥稀疏注意力的优势")
    elif sparsity > 0.95:
        warnings.append(f"稀疏度过高({sparsity:.2%})，可能会影响模型性能")
    
    # 检查KV头数量
    if kv_heads is not None and kv_heads > heads:
        warnings.append(f"KV头数量({kv_heads})大于Q头数量({heads})，标准GQA应为KV头数量≤Q头数量")
    
    return warnings


def format_flops(flops: float) -> str:
    """将浮点运算数格式化为合适的单位（K/M/G/T）"""
    if flops < 1e3:
        return f"{flops:.2f} Flops"
    elif flops < 1e6:
        return f"{flops/1e3:.2f} KFlops"
    elif flops < 1e9:
        return f"{flops/1e6:.2f} MFlops"
    elif flops < 1e12:
        return f"{flops/1e9:.2f} GFlops"
    else:
        return f"{flops/1e12:.2f} TFlops"


def compare_configurations(base_config: Dict[str, int], 
                          variations: List[Dict[str, Union[int, str]]], 
                          excel_output: Optional[str] = None) -> None:
    """
    比较不同配置下的计算复杂度
    
    参数:
        base_config: 基础配置
        variations: 变化配置列表，每个配置是一个字典，包含变化的参数和描述
        excel_output: Excel输出文件路径，如果为None则不输出
    """
    results = []
    param_names = {
        'seqlen': '序列长度',
        'stage1_block_size': '第一阶段块大小', 
        'stage1_stride': '第一阶段步长',
        'stage2_block_size': '第二阶段块大小', 
        'stage2_selected_block_num': '第二阶段选择的块数量',
        'stage2_window_size': '第二阶段窗口大小',
        'head_dim': '注意力头维度',
        'heads': '注意力头数量',
        'kv_heads': 'KV头数量'
    }
    
    # 计算基础配置
    base_result = calculate_flops(**base_config)
    results.append(('基础配置', base_config, base_result))
    
    # 计算变化配置
    for variation in variations:
        config = base_config.copy()
        desc = variation.pop('description', '')
        for k, v in variation.items():
            if k in config:
                config[k] = v
        
        result = calculate_flops(**config)
        results.append((desc, config, result))
    
    # 打印比较结果
    print("\n配置比较结果:")
    print("-" * 120)
    print(f"{'配置':<20} {'传统注意力':<18} {'总计算量':<18} {'第一阶段计算量':<18} {'第二阶段计算量':<18} {'稀疏度':<10} {'效率比':<10}")
    print("-" * 120)
    
    for desc, config, result in results:
        stage1_flops, stage2_flops, total_flops, _, sparsity, efficiency, standard_flops = result
        print(f"{desc:<20} {format_flops(standard_flops):<18} {format_flops(total_flops):<18} {format_flops(stage1_flops):<18} {format_flops(stage2_flops):<18} {sparsity*100:>6.2f}% {efficiency:>8.2f}x")
    
    print("-" * 120)
    
    # 如果需要导出到Excel
    if excel_output and PANDAS_AVAILABLE:
        try:
            # 创建数据框用于Excel输出
            excel_data = []
            
            for desc, config, result in results:
                stage1_flops, stage2_flops, total_flops, flops_ratio, sparsity, efficiency, standard_flops = result
                stage2_selected_size = config['stage2_block_size'] * config['stage2_selected_block_num']
                stage2_total_size = stage2_selected_size + config['stage2_window_size']
                
                # 组合配置和结果
                row_data = {
                    '配置': desc,
                    # 输入参数
                    '序列长度': config['seqlen'],
                    '第一阶段块大小': config['stage1_block_size'],
                    '第一阶段步长': config['stage1_stride'],
                    '第二阶段块大小': config['stage2_block_size'],
                    '第二阶段选择的块数量': config['stage2_selected_block_num'],
                    '第二阶段窗口大小': config['stage2_window_size'],
                    '注意力头维度': config['head_dim'],
                    '注意力头数量': config['heads'],
                    'KV头数量': config['kv_heads'],
                    '第二阶段选择块总大小': stage2_selected_size,
                    '第二阶段总注意力大小': stage2_total_size,
                    # 计算结果
                    '传统注意力计算量': standard_flops,
                    '第一阶段计算量': stage1_flops,
                    '第二阶段计算量': stage2_flops,
                    '总计算量': total_flops,
                    '两阶段计算比例': flops_ratio,
                    '稀疏度(%)': sparsity * 100,
                    '效率比': efficiency
                }
                excel_data.append(row_data)
            
            # 创建DataFrame并导出
            df = pd.DataFrame(excel_data)
            df.to_excel(excel_output, index=False)
            print(f"\nExcel报表已导出至: {excel_output}")
            
        except Exception as e:
            print(f"\nExcel导出失败: {str(e)}")
    elif excel_output and not PANDAS_AVAILABLE:
        print("\n警告: 无法导出Excel文件。请安装pandas库: pip install pandas")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='稀疏注意力机制的计算复杂度分析工具')
    
    parser.add_argument('--seqlen', type=int, default=131072,
                        help='序列长度')
    parser.add_argument('--stage1-block-size', type=int, default=32,
                        help='第一阶段块大小')
    parser.add_argument('--stage1-stride', type=int, default=16,
                        help='第一阶段步长')
    parser.add_argument('--stage2-block-size', type=int, default=64,
                        help='第二阶段块大小')
    parser.add_argument('--stage2-selected-block-num', type=int, default=32,
                        help='第二阶段选择的块数量')
    parser.add_argument('--stage2-window-size', type=int, default=2048,
                        help='第二阶段窗口大小')
    parser.add_argument('--head-dim', type=int, default=128,
                        help='注意力头维度')
    parser.add_argument('--heads', type=int, default=32, 
                        help='注意力头数量')
    parser.add_argument('--kv-heads', type=int, default=32,
                        help='KV头数量，用于GQA（默认等于注意力头数量）')
    parser.add_argument('--compare', action='store_true',
                        help='比较不同配置')
    parser.add_argument('--excel-output', type=str,
                        help='导出比较结果到Excel文件')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 如果没有命令行参数，使用默认配置
    if len(sys.argv) == 1:
        run_default_analysis()
        return
    
    # 解析命令行参数
    args = parse_args()
    
    # 获取配置
    config = {
        'seqlen': args.seqlen,
        'stage1_block_size': args.stage1_block_size,
        'stage1_stride': args.stage1_stride,
        'stage2_block_size': args.stage2_block_size,
        'stage2_selected_block_num': args.stage2_selected_block_num,
        'stage2_window_size': args.stage2_window_size,
        'head_dim': args.head_dim,
        'heads': args.heads,
        'kv_heads': args.kv_heads
    }
    
    # 验证参数
    warnings = validate_params(
        args.seqlen, args.stage1_block_size, args.stage1_stride,
        args.stage2_block_size, args.stage2_selected_block_num, args.stage2_window_size,
        args.kv_heads, args.heads
    )
    
    if warnings:
        print("参数警告:")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    
    # 添加第二阶段详细信息
    stage2_selected_size = args.stage2_block_size * args.stage2_selected_block_num
    stage2_total_size = stage2_selected_size + args.stage2_window_size
    
    # 计算结果
    stage1_flops, stage2_flops, total_flops, flops_ratio, sparsity, efficiency_ratio, standard_flops = calculate_flops(**config)
    
    # 打印输入参数
    print("输入参数:")
    print(f"  序列长度: {args.seqlen}")
    print(f"  第一阶段块大小: {args.stage1_block_size}")
    print(f"  第一阶段步长: {args.stage1_stride}")
    print(f"  第二阶段块大小: {args.stage2_block_size}")
    print(f"  第二阶段选择的块数量: {args.stage2_selected_block_num}")
    print(f"  第二阶段窗口大小: {args.stage2_window_size}")
    print(f"  注意力头维度: {args.head_dim}")
    print(f"  注意力头数量: {args.heads}")
    print(f"  KV头数量: {args.kv_heads}")
    print()
    
    # 打印计算结果
    print("计算结果:")
    print("备注：计算结果由AI生成，比例正常，但计算量仅供参考")
    print("--------------------------------")
    print(f"  第二阶段窗口大小: {args.stage2_window_size}")
    print(f"  第二阶段选择块总大小: {stage2_selected_size}")
    print(f"  第二阶段总注意力大小: {stage2_total_size}")
    print(f"  传统注意力计算量: {format_flops(standard_flops)}")
    print(f"  第一阶段计算量: {format_flops(stage1_flops)}")
    print(f"  第二阶段计算量: {format_flops(stage2_flops)}")
    print(f"  总计算量: {format_flops(total_flops)}")
    print(f"  两阶段计算比例: {flops_ratio:.2f}")
    print(f"  稀疏度: {sparsity*100:.2f}%")
    print(f"  效率比: {efficiency_ratio:.2f}x（相比传统注意力）")
    
    # 如果需要比较不同配置
    if args.compare:
        # 创建几个有代表性的变化配置
        variations = [
            {'description': '更大的窗口', 'stage2_window_size': args.stage2_window_size * 2},
            {'description': '更多的选择块', 'stage2_selected_block_num': args.stage2_selected_block_num * 2},
            {'description': '更大的步长', 'stage1_stride': args.stage1_stride * 2},
            {'description': '更小的步长', 'stage1_stride': max(1, args.stage1_stride // 2)},
        ]
        # 如果使用了GQA，添加相关对比配置
        if args.kv_heads is not None and args.kv_heads != args.heads:
            variations.append({'description': '不使用GQA', 'kv_heads': args.heads})
            variations.append({'description': '更少的KV头', 'kv_heads': max(1, args.kv_heads // 2)})
        else:
            variations.append({'description': '使用GQA (1:4)', 'kv_heads': max(1, args.heads // 4)})
            variations.append({'description': '使用GQA (1:8)', 'kv_heads': max(1, args.heads // 8)})
            
        compare_configurations(config, variations, args.excel_output)


def run_default_analysis():
    """使用默认参数运行分析"""
    seqlen = 131072
    stage1_block_size = 32
    stage1_stride = 16
    stage2_block_size = 64
    stage2_selected_block_num = 32
    stage2_window_size = 2048
    head_dim = 128
    heads = 32
    kv_heads = 2  # 默认使用GQA
    
    # 验证参数
    warnings = validate_params(
        seqlen, stage1_block_size, stage1_stride,
        stage2_block_size, stage2_selected_block_num, stage2_window_size,
        kv_heads, heads
    )
    
    if warnings:
        print("参数警告:")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    
    # 打印输入参数
    print("输入参数:")
    print(f"  序列长度: {seqlen}")
    print(f"  第一阶段块大小: {stage1_block_size}")
    print(f"  第一阶段步长: {stage1_stride}")
    print(f"  第二阶段块大小: {stage2_block_size}")
    print(f"  第二阶段选择的块数量: {stage2_selected_block_num}")
    print(f"  第二阶段窗口大小: {stage2_window_size}")
    print(f"  注意力头维度: {head_dim}")
    print(f"  注意力头数量: {heads}")
    print(f"  KV头数量: {kv_heads}")
    print()

    # 计算结果
    stage1_flops, stage2_flops, total_flops, flops_ratio, sparsity, efficiency_ratio, standard_flops = calculate_flops(
        seqlen, stage1_block_size, stage1_stride, stage2_block_size, 
        stage2_selected_block_num, stage2_window_size, head_dim, heads, kv_heads
    )
    
    # 添加第二阶段详细信息
    stage2_selected_size = stage2_block_size * stage2_selected_block_num
    stage2_total_size = stage2_selected_size + stage2_window_size
    # 打印计算结果
    print("计算结果:")
    print("备注：计算结果由AI生成，比例正常，但计算量仅供参考")
    print("--------------------------------")
    print(f"  第二阶段窗口大小: {stage2_window_size}")
    print(f"  第二阶段选择块总大小: {stage2_selected_size}")
    print(f"  第二阶段总注意力大小: {stage2_total_size}")
    print(f"  传统注意力计算量: {format_flops(standard_flops)}")
    print(f"  第一阶段计算量: {format_flops(stage1_flops)}")
    print(f"  第二阶段计算量: {format_flops(stage2_flops)}")
    print(f"  总计算量: {format_flops(total_flops)}")
    print(f"  两阶段计算比例: {flops_ratio:.2f}")
    print(f"  稀疏度: {sparsity*100:.2f}%")
    print(f"  效率比: {efficiency_ratio:.2f}x（相比传统注意力）")
    
    print("\n提示: 使用 --help 查看更多选项")


if __name__ == "__main__":
    main()