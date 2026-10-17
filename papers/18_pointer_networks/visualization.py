"""
指针网络可视化工具
按 30u30 第 18 天的格式组织

支持以下可视化：
1. 注意力热图（指针分布）
2. 排序结果验证
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_attention(input_values, attention_probs, save_path="attention_viz.png"):
    """
    绘制指针概率矩阵。
    
    参数：
        input_values（list）：原始输入值，例如数字或文本。
        attention_probs（ndarray）：形状为 (seq_len, seq_len) 的概率矩阵。
        save_path（str）：生成图像的保存路径。
    """
    seq_len = len(input_values)
    
    # 创建热图
    plt.figure(figsize=(10, 8))
    im = plt.imshow(attention_probs, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, label='指针强度')
    
    # 坐标轴标签
    # X 轴：模型可指向的输入项
    plt.xticks(range(seq_len), [f"{v:.2f}" for v in input_values])
    # Y 轴：输出步骤，即模型依次选择的元素
    plt.yticks(range(seq_len), [f"输出步骤 {i+1}" for i in range(seq_len)])
    
    plt.title('指针网络：注意力热图', fontsize=14, fontweight='bold')
    plt.xlabel('输入序列元素', fontsize=12)
    plt.ylabel('解码时间步', fontsize=12)
    
    # 添加网格线以便阅读
    plt.vlines(np.arange(-0.5, seq_len), -0.5, seq_len-0.5, colors='grey', alpha=0.1)
    plt.hlines(np.arange(-0.5, seq_len), -0.5, seq_len-0.5, colors='grey', alpha=0.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[OK] 可视化已保存至 {save_path}")
    plt.close()

if __name__ == "__main__":
    print('指针网络可视化工具')
    print("=" * 60)
    
    # 用于演示的模拟数据
    # 序列：0.7、0.2、0.9、0.1、0.5
    # 排序：0.1（索引 3）、0.2（索引 1）、0.5（索引 4）、0.7（索引 0）、0.9（索引 2）
    inputs = [0.7, 0.2, 0.9, 0.1, 0.5]
    
    # 创建看起来像已训练模型的尖锐矩阵；并非实际训练结果
    dummy_attn = np.zeros((5, 5))
    dummy_attn[0, 3] = 0.9  # 选择 0.1
    dummy_attn[1, 1] = 0.95 # 选择 0.2
    dummy_attn[2, 4] = 0.92 # 选择 0.5
    dummy_attn[3, 0] = 0.98 # 选择 0.7
    dummy_attn[4, 2] = 0.99 # 选择 0.9
    
    # 加噪声模拟真实模型行为
    dummy_attn += np.random.rand(5, 5) * 0.05
    # 重新归一化，使每行总和为 1
    dummy_attn /= dummy_attn.sum(axis=1, keepdims=True)
    
    plot_attention(inputs, dummy_attn)
