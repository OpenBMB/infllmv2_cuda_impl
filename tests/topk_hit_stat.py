import re
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# 从文件读取数据
def read_data(filename):
    # 定义多种数据模式的正则表达式
    hit_pattern = r'layer_id:(\d+) q的后1000个里面命中的块  :(\d+)'
    topk_gatesum_pattern = r'layer_id:(\d+)topk_attn_output gate 的sum :(\d+\.\d+)'
    compressed_gatesum_pattern = r'layer_id:(\d+)compressed_attn_output的 gate 的sum :(\d+\.\d+)'
    
    data = []
    topk_gatesum_data = []
    compressed_gatesum_data = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                # 匹配命中块数据
                hit_match = re.search(hit_pattern, line)
                if hit_match:
                    layer_id = int(hit_match.group(1))
                    value = int(hit_match.group(2))
                    data.append((layer_id, value))
                
                # 匹配topk_attn_output的gatesum
                topk_match = re.search(topk_gatesum_pattern, line)
                if topk_match:
                    layer_id = int(topk_match.group(1))
                    value = float(topk_match.group(2))
                    topk_gatesum_data.append((layer_id, value))
                
                # 匹配compressed_attn_output的gatesum
                compressed_match = re.search(compressed_gatesum_pattern, line)
                if compressed_match:
                    layer_id = int(compressed_match.group(1))
                    value = float(compressed_match.group(2))
                    compressed_gatesum_data.append((layer_id, value))
    except FileNotFoundError:
        print(f"文件未找到: {filename}")
    
    return data, topk_gatesum_data, compressed_gatesum_data

# 计算统计数据
def calculate_stats(data):
    values = [v for _, v in data]
    non_zero_values = [v for v in values if v > 0]
    
    # 基本统计量
    stats = {
        "总数": len(values),
        "非零样本数": len(non_zero_values),
        "非零比例": len(non_zero_values) / len(values) if values else 0,
        "平均值": np.mean(values) if values else 0,
        "非零平均值": np.mean(non_zero_values) if non_zero_values else 0,
        "中位数": np.median(values) if values else 0,
        "方差": np.var(values) if values else 0,
        "标准差": np.std(values) if values else 0,
        "最大值": max(values) if values else 0,
        "最小值": min(values) if values else 0,
        "众数": Counter(values).most_common(1)[0][0] if values else None
    }
    
    # 按层统计
    layers = {}
    for layer_id, value in data:
        if layer_id not in layers:
            layers[layer_id] = []
        layers[layer_id].append(value)
    
    layer_stats = {}
    for layer_id, values in layers.items():
        non_zero = [v for v in values if v > 0]
        layer_stats[layer_id] = {
            "样本数": len(values),
            "非零样本数": len(non_zero),
            "非零比例": len(non_zero) / len(values) if values else 0,
            "总和": sum(values),
            "平均值": np.mean(values),
            "非零平均值": np.mean(non_zero) if non_zero else 0,
            "方差": np.var(values),
            "标准差": np.std(values)
        }
    
    return stats, layer_stats

# 比较两个数据集
def compare_datasets(file1, file2):
    # 读取数据
    data1, topk_gatesum1, compressed_gatesum1 = read_data(file1)
    data2, topk_gatesum2, compressed_gatesum2 = read_data(file2)
    
    # 计算命中块统计值
    stats1, layer_stats1 = calculate_stats(data1)
    stats2, layer_stats2 = calculate_stats(data2)
    
    # 打印总体统计对比
    print(f"对比结果 - 文件1: {file1.split('/')[-1]}, 文件2: {file2.split('/')[-1]}")
    print("\n=== 命中块统计比较 ===")
    
    # 创建对比表格
    stats_df = pd.DataFrame({
        '指标': list(stats1.keys()),
        '文件1': list(stats1.values()),
        '文件2': list(stats2.values()),
        '差值': [stats2[k] - stats1[k] if isinstance(stats1[k], (int, float)) and isinstance(stats2[k], (int, float)) 
               else 'N/A' for k in stats1]
    })
    print(stats_df.to_string(index=False))
    
    # 按层统计对比
    print("\n命中块按层统计对比:")
    
    all_layers = sorted(set(layer_stats1.keys()) | set(layer_stats2.keys()))
    
    for layer in all_layers:
        print(f"\n层 {layer}:")
        if layer in layer_stats1 and layer in layer_stats2:
            layer_df = pd.DataFrame({
                '指标': list(layer_stats1[layer].keys()),
                '文件1': list(layer_stats1[layer].values()),
                '文件2': list(layer_stats2[layer].values()),
                '差值': [layer_stats2[layer][k] - layer_stats1[layer][k] 
                       for k in layer_stats1[layer]]
            })
            print(layer_df.to_string(index=False))
        elif layer in layer_stats1:
            print("  仅在文件1中存在")
            for k, v in layer_stats1[layer].items():
                print(f"  {k}: {v}")
        else:
            print("  仅在文件2中存在")
            for k, v in layer_stats2[layer].items():
                print(f"  {k}: {v}")
    
    # Gate Sum 对比
    compare_gatesum(topk_gatesum1, topk_gatesum2, compressed_gatesum1, compressed_gatesum2)

# 比较gate sum数据
def compare_gatesum(topk1, topk2, compressed1, compressed2):
    print("\n=== Gate Sum 统计 ===")
    
    # 聚合每层的gate sum数据
    def aggregate_by_layer(data):
        result = {}
        for layer_id, value in data:
            if layer_id not in result:
                result[layer_id] = []
            result[layer_id].append(value)
        return result
    
    topk_by_layer1 = aggregate_by_layer(topk1)
    topk_by_layer2 = aggregate_by_layer(topk2)
    compressed_by_layer1 = aggregate_by_layer(compressed1)
    compressed_by_layer2 = aggregate_by_layer(compressed2)
    
    # 获取所有层ID
    all_layers = sorted(set(topk_by_layer1.keys()) | set(topk_by_layer2.keys()) | 
                       set(compressed_by_layer1.keys()) | set(compressed_by_layer2.keys()))
    
    print("\nTopK Gate Sum 对比:")
    for layer in all_layers:
        topk_avg1 = np.mean(topk_by_layer1.get(layer, [0])) if layer in topk_by_layer1 else 0
        topk_avg2 = np.mean(topk_by_layer2.get(layer, [0])) if layer in topk_by_layer2 else 0
        
        if topk_avg1 > 0 or topk_avg2 > 0:
            print(f"层 {layer}: 文件1 = {topk_avg1:.2f}, 文件2 = {topk_avg2:.2f}, 差值 = {topk_avg2 - topk_avg1:.2f}")
    
    print("\nCompressed Gate Sum 对比:")
    for layer in all_layers:
        comp_avg1 = np.mean(compressed_by_layer1.get(layer, [0])) if layer in compressed_by_layer1 else 0
        comp_avg2 = np.mean(compressed_by_layer2.get(layer, [0])) if layer in compressed_by_layer2 else 0
        
        if comp_avg1 > 0 or comp_avg2 > 0:
            print(f"层 {layer}: 文件1 = {comp_avg1:.2f}, 文件2 = {comp_avg2:.2f}, 差值 = {comp_avg2 - comp_avg1:.2f}")
    
    # Gate 比例对比
    print("\nCompressed/TopK Gate Sum 比例对比:")
    for layer in all_layers:
        topk_avg1 = np.mean(topk_by_layer1.get(layer, [0])) if layer in topk_by_layer1 and topk_by_layer1[layer] else 0
        comp_avg1 = np.mean(compressed_by_layer1.get(layer, [0])) if layer in compressed_by_layer1 and compressed_by_layer1[layer] else 0
        
        topk_avg2 = np.mean(topk_by_layer2.get(layer, [0])) if layer in topk_by_layer2 and topk_by_layer2[layer] else 0
        comp_avg2 = np.mean(compressed_by_layer2.get(layer, [0])) if layer in compressed_by_layer2 and compressed_by_layer2[layer] else 0
        
        ratio1 = comp_avg1 / topk_avg1 if topk_avg1 > 0 else 0
        ratio2 = comp_avg2 / topk_avg2 if topk_avg2 > 0 else 0
        
        if ratio1 > 0 or ratio2 > 0:
            print(f"层 {layer}: 文件1 = {ratio1:.4f}, 文件2 = {ratio2:.4f}, 差值 = {ratio2 - ratio1:.4f}")

# 可视化比较
def compare_visually(data1, data2):
    # 提取每个层的值
    layers1 = {}
    for layer_id, value in data1:
        if layer_id not in layers1:
            layers1[layer_id] = []
        layers1[layer_id].append(value)
    
    layers2 = {}
    for layer_id, value in data2:
        if layer_id not in layers2:
            layers2[layer_id] = []
        layers2[layer_id].append(value)
    
    # 计算每个层的平均值
    layer_means1 = {layer: np.mean(values) for layer, values in layers1.items()}
    layer_means2 = {layer: np.mean(values) for layer, values in layers2.items()}
    
    # 所有层ID
    all_layers = sorted(set(layer_means1.keys()) | set(layer_means2.keys()))
    
    # 准备绘图数据
    means1 = [layer_means1.get(layer, 0) for layer in all_layers]
    means2 = [layer_means2.get(layer, 0) for layer in all_layers]
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    x = np.arange(len(all_layers))
    width = 0.35
    
    plt.bar(x - width/2, means1, width, label='文件1')
    plt.bar(x + width/2, means2, width, label='文件2')
    
    plt.xlabel('Layer ID')
    plt.ylabel('平均值')
    plt.title('各层命中块平均值对比')
    plt.xticks(x, all_layers)
    plt.legend()
    
    # 保存图表到文件
    plt.savefig('layer_comparison.png')
    print("\n对比图已保存为: layer_comparison.png")

if __name__ == "__main__":
    file1 = "/user/zhouzihan/workspace/RULER/showtopk.org.sg2.log"
    file2 = "/user/zhouzihan/workspace/RULER/showtopk.train_topk32.sg2.log"
    
    compare_datasets(file1, file2)