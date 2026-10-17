"""
第 20 天：关系 RNN 可视化
===================================================================

用于理解关系记忆的图表：
- 注意力热图：观察模型关注自身记忆的哪些位置
- 损失比较：RMC 与 LSTM 学习曲线

运行 train_minimal.py --visualize 生成图表。
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def plot_attention_heatmap(attention_weights, save_path="attention_map.png"):
    """
    可视化 RMC 自注意力权重随时间的变化。
    
    参数：
        attention_weights: (seq_len, num_heads, mem_slots, mem_slots)
                           绘制各注意力头的平均权重。
        save_path：图像保存路径。
    """
    # 转换为 numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
        
    seq_len, num_heads, slots, _ = attention_weights.shape
    
    # 对注意力头取平均以简化可视化：(seq_len, slots, slots)
    avg_attention = np.mean(attention_weights, axis=1)
    
    # 观察注意力矩阵随时间的变化。
    # 绘制热图网格，每个时间步或选定子集对应一张。
    
    # 若序列较长，最多显示前 10 步或抽样
    plot_steps = min(seq_len, 10)
    
    fig, axes = plt.subplots(1, plot_steps, figsize=(2 * plot_steps, 2.5))
    if plot_steps == 1:
        axes = [axes]
        
    for t in range(plot_steps):
        ax = axes[t]
        # 热图
        im = ax.imshow(avg_attention[t], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"t={t}")
        ax.axis('off')
        
    # 添加色条
    plt.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    plt.suptitle('RMC 记忆自注意力随时间的变化（各头平均）', y=0.85)
    
    plt.savefig(save_path)
    plt.close()
    print(f"注意力热图已保存至 {save_path}")

def plot_comparison(rmc_losses, lstm_losses, save_path="comparison.png"):
    """
    绘制 RMC 与 LSTM 的训练损失比较。
    """
    plt.figure(figsize=(10, 6))
    
    # 必要时可轻微平滑以便阅读；此处保留原始曲线。
    plt.plot(rmc_losses, label='RMC（关系记忆）', alpha=0.8)
    plt.plot(lstm_losses, label='LSTM（标准）', alpha=0.8)
    
    plt.title('训练损失：RMC 与 LSTM（本章距离任务）')
    plt.xlabel('步数')
    plt.ylabel('均方误差损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path)
    plt.close()
    print(f"比较图已保存至 {save_path}")
