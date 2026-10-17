import torch
import matplotlib.pyplot as plt
import numpy as np

"""
NTM 可视化工具
目标：在模型处理序列时，将可微寻址的关注分布
（即权重）可视化，类似 Graves 等人（2014）的图 4。
"""

def plot_memory_weights(weights_list, title='存储寻址权重（读头）'):
    """
    可视化读写头的关注位置如何随时间在存储位置间移动。
    X 轴：时间步
    Y 轴：存储索引（N）
    """
    # 提取批次中第一个样本的权重
    # weights_list 是由形状为 [batch, N] 的张量构成的列表
    weights = torch.stack(weights_list).detach().cpu().numpy()[:, 0, :]
    
    plt.figure(figsize=(12, 6))
    plt.imshow(weights.T, aspect='auto', cmap='hot')
    plt.xlabel('时间步 (t)')
    plt.ylabel('存储位置索引 (i)')
    plt.title(title)
    plt.colorbar(label='寻址权重 w_t(i)')
    plt.tight_layout()
    plt.savefig("papers/17_neural_turing_machines/addressing_focus.png")
    plt.show()

def plot_recall_performance(inputs, targets, outputs, title='复制任务：输入与回忆'):
    """
    定性检查：比特掩码的回忆结果与目标是否一致？
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 输入阶段
    axes[0].imshow(inputs.detach().cpu().numpy().T, aspect='auto', cmap='binary')
    axes[0].set_title('输入序列（第 9 通道为分隔符）')
    
    # 目标
    axes[1].imshow(targets.detach().cpu().numpy().T, aspect='auto', cmap='binary')
    axes[1].set_title('期望输出（目标）')
    
    # 预测
    preds = (outputs > 0).float().detach().cpu().numpy()
    axes[2].imshow(preds.T, aspect='auto', cmap='binary')
    axes[2].set_title('模型实际回忆（预测）')
    
    for ax in axes:
        ax.set_ylabel('比特索引')
    
    plt.tight_layout()
    plt.savefig("papers/17_neural_turing_machines/recall_comparison.png")
    plt.show()
