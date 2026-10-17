"""
visualization.py：关系网络可视化程序

本模块提供多种可视化，
用于检查关系推理的表现与局限。

功能：
1. 泛化曲线：绘制准确率随对象数 N 的变化。
2. 聚合器比较：展示求和与均值的计数偏置。
3. 关系热图：查看内部 g_theta 输出的幅值。
4. 失败样例定位：分析模型出错位置的空间分布。

用法：
    python visualization.py --compare-aggregators
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from implementation import RelationNetwork
from train_minimal import RelationalDataset, run_evaluation

# ============================================================================
# 实验图表
# ============================================================================

def plot_generalization_curve(model, device, train_n: int, max_n: int = 20):
    """
    绘制模型准确率随对象数 N 增长的变化。
    检查 RN 能否泛化到未见过的集合大小。
    """
    n_values = range(2, max_n + 1)
    accuracies = []
    
    print(f"正在生成泛化曲线（训练 N={train_n})...")
    
    for n in n_values:
        test_dataset = RelationalDataset(mode='count', num_samples=500, num_objects=n)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
        
        # 说明：计数任务需要 f_phi 具有 num_objects+1 个输出。
        # 本可视化假定模型按最大 N 构建输出空间。
        acc = run_evaluation(model, test_loader, device)
        accuracies.append(acc)
        
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, accuracies, 'bo-', linewidth=2, label='RN 表现')
    plt.axvline(train_n, color='r', linestyle='--', label=f'训练 N={train_n}')
    
    plt.xlabel('对象数量 (N)')
    plt.ylabel('准确率（计数任务）')
    plt.title('集合大小泛化：准确率与对象数量', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.05)
    
    save_path = "viz_generalization.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Output] 泛化曲线已保存至 {save_path}")


def plot_aggregator_comparison(results_dict: dict):
    """
    可视化求和与均值聚合器在计数任务上的表现。
    """
    labels = list(results_dict.keys())
    accs = [results_dict[l] for l in labels]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, accs, color=['#3498db', '#e74c3c', '#2ecc71'])
    
    plt.ylabel('准确率')
    plt.title('关系归纳偏置：求和与均值聚合', fontsize=14)
    plt.ylim(0, 1.0)
    
    # 在柱形顶部添加准确率标签
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')
        
    plt.savefig("viz_aggregator_comp.png", dpi=150)
    plt.close()
    print('  [Output] 聚合器比较已保存至 viz_aggregator_comp.png')


def plot_failure_heatmap(points, model, device):
    """
    可视化失败样例中 g_theta 的幅值；原文称“注意力”，实际不是归一化注意力权重。
    帮助检查模型是否忽略某些对象。
    """
    model.eval()
    with torch.no_grad():
        points_t = torch.tensor(points).float().unsqueeze(0).to(device)
        pairs = model.generate_pairs(points_t)
        g_out = model.g_theta(pairs.view(-1, pairs.shape[-1]))
        rel_mag = torch.norm(g_out, dim=-1).view(len(points), len(points)).cpu().numpy()
        
    plt.figure(figsize=(8, 7))
    plt.imshow(rel_mag, cmap='viridis')
    plt.colorbar(label='关系幅值')
    plt.title('内部关系幅值（类比第 4.1 节分析）')
    plt.xlabel('对象索引 i')
    plt.ylabel('对象索引 j')
    plt.savefig("viz_internal_heatmap.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    # 冒烟检查：模拟可视化程序
    device = torch.device('cpu')
    dummy_model = RelationNetwork(object_dim=2, output_dim=21, aggregator='sum')
    
    # 1. 泛化
    plot_generalization_curve(dummy_model, device, train_n=10, max_n=20)
    
    # 2. 聚合器比较（模拟数据）
    mock_results = {'sum': 0.92, 'mean': 0.45, 'max': 0.61}
    plot_aggregator_comparison(mock_results)
    
    print('\n可视化冒烟检查完成。')
