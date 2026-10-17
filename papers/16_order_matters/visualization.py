"""
指针网络可视化工具
============================================

用于可视化注意力机制与几何问题求解过程的工具。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

def plot_attention_heatmap(
    attention_weights: List[torch.Tensor],
    input_labels: List[str],
    output_labels: List[str],
    title: str = '注意力热力图',
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    将注意力权重可视化为热力图。
    """
    attention_matrix = torch.stack(attention_weights).cpu().numpy()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attention_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, label='注意力权重')
    
    ax.set_xticks(range(len(input_labels)))
    ax.set_xticklabels(input_labels)
    ax.set_yticks(range(len(output_labels)))
    ax.set_yticklabels(output_labels)
    
    ax.set_xlabel('输入元素')
    ax.set_ylabel('输出步骤')
    ax.set_title(title)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def visualize_sorting(
    input_values: torch.Tensor,
    pointers: torch.Tensor,
    attention_weights: List[torch.Tensor],
    save_path: Optional[str] = None
):
    input_values = input_values.cpu().numpy().squeeze()
    pointers = pointers.cpu().numpy()
    fig = plt.figure(figsize=(16, 5))
    
    # 输入面板
    ax1 = plt.subplot(1, 3, 1)
    ax1.bar(np.arange(len(input_values)), input_values)
    ax1.set_title('未排序输入')
    
    # 注意力面板
    ax2 = plt.subplot(1, 3, 2)
    matrix = torch.stack(attention_weights).cpu().numpy()
    ax2.imshow(matrix, cmap='YlOrRd')
    ax2.set_title('注意力权重')
    
    # 输出面板
    ax3 = plt.subplot(1, 3, 3)
    ax3.bar(np.arange(len(input_values)), input_values[pointers])
    ax3.set_title('排序后输出')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


if __name__ == "__main__":
    print('可视化工具已加载。')
