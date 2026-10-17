"""
字符级 RNN 可视化工具
===========================================

可视化功能：
1. 训练损失曲线
2. 隐藏状态演化
3. 字符概率热力图
4. 梯度随时间的传播
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_curve(losses, title="训练损失", save_path=None):
    """
    绘制训练损失随迭代的变化。
    
    参数：
        losses：损失值列表
        title：图标题
        save_path：可选的图片保存路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('损失', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 添加平滑曲线
    if len(losses) > 100:
        window = 100
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(losses)), smoothed, 
                color='red', linewidth=2, label='平滑曲线', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存到 {save_path}")
    else:
        plt.show()


def plot_hidden_states(hidden_states, char_sequence, save_path=None):
    """
    可视化隐藏状态沿序列的变化。
    
    参数：
        hidden_states：形状为 (seq_len, hidden_size) 的数组
        char_sequence：与各时间步对应的字符序列
        save_path：可选的图片保存路径
    """
    plt.figure(figsize=(12, 6))
    
    # 转置后，行对应隐藏单元，列对应时间步
    hidden_states_T = hidden_states.T
    
    sns.heatmap(hidden_states_T, cmap='RdBu_r', center=0, 
                xticklabels=list(char_sequence),
                yticklabels=False,
                cbar_kws={'label': '激活值'})
    
    plt.xlabel('字符', fontsize=12)
    plt.ylabel('隐藏单元', fontsize=12)
    plt.title('隐藏状态沿序列的变化', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存到 {save_path}")
    else:
        plt.show()


def plot_probability_distribution(probs, chars, top_k=10, save_path=None):
    """
    绘制单个时间步的字符概率分布。
    
    参数：
        probs：形状为 (vocab_size,) 的概率数组
        chars：字符列表
        top_k：展示概率最高的字符数量
        save_path：可选的图片保存路径
    """
    # 取概率最高的 k 个字符
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_probs = probs[top_indices]
    top_chars = [chars[i] for i in top_indices]
    
    # 替换特殊字符以便显示
    display_chars = []
    for ch in top_chars:
        if ch == '\n':
            display_chars.append('\\n')
        elif ch == ' ':
            display_chars.append('空格')
        else:
            display_chars.append(ch)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(top_k), top_probs, color='skyblue', edgecolor='navy')
    
    # 突出显示概率最高的预测
    bars[0].set_color('coral')
    
    plt.xlabel('字符', fontsize=12)
    plt.ylabel('概率', fontsize=12)
    plt.title('字符概率分布', fontsize=14, fontweight='bold')
    plt.xticks(range(top_k), display_chars, fontsize=10)
    plt.ylim(0, max(top_probs) * 1.1)
    
    # 在条形图上标注数值
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存到 {save_path}")
    else:
        plt.show()


def plot_sequence_predictions(text, predictions, chars, save_path=None):
    """
    展示模型在序列各字符之后的预测。
    
    参数：
        text：输入文本字符串
        predictions：包含概率、形状为 (seq_len, vocab_size) 的数组
        chars：词表中所有字符的列表
        save_path：可选的图片保存路径
    """
    seq_len = len(text)
    
    # 取各步概率最高的 3 个预测
    fig, axes = plt.subplots(seq_len, 1, figsize=(12, 2*seq_len))
    
    if seq_len == 1:
        axes = [axes]
    
    for t, (char, probs, ax) in enumerate(zip(text, predictions, axes)):
        # 取概率最高的 3 项
        top_indices = np.argsort(probs)[-3:][::-1]
        top_probs = probs[top_indices]
        top_chars = [chars[i] for i in top_indices]
        
        # 显示
        display_chars = [ch if ch not in ['\n', ' '] else ('\\n' if ch == '\n' else '空格') for ch in top_chars]
        
        bars = ax.barh(range(3), top_probs, color=['green', 'orange', 'red'])
        ax.set_yticks(range(3))
        ax.set_yticklabels(display_chars)
        ax.set_xlim(0, 1)
        ax.set_title(f"位于 '{char}' 之后 → 预测下一个字符", fontsize=10)
        ax.set_xlabel('概率')
        
        # 添加数值标注
        for bar, prob in zip(bars, top_probs):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{prob:.3f}',
                   ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存到 {save_path}")
    else:
        plt.show()


def plot_gradient_flow(grad_norms, save_path=None):
    """
    可视化梯度大小随时间的变化，以诊断梯度消失或爆炸。
    
    参数：
        grad_norms：各时间步梯度范数的数组
        save_path：可选的图片保存路径
    """
    plt.figure(figsize=(10, 6))
    
    time_steps = range(len(grad_norms))
    plt.plot(time_steps, grad_norms, marker='o', linewidth=2, markersize=6)
    plt.axhline(y=1.0, color='green', linestyle='--', label='梯度范数 = 1', alpha=0.7)
    plt.axhline(y=0.1, color='orange', linestyle='--', label='消失阈值', alpha=0.7)
    plt.axhline(y=10.0, color='red', linestyle='--', label='爆炸阈值', alpha=0.7)
    
    plt.xlabel('时间步（反向）', fontsize=12)
    plt.ylabel('梯度范数', fontsize=12)
    plt.title('梯度随时间的传播', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存到 {save_path}")
    else:
        plt.show()


def plot_temperature_comparison(samples_dict, save_path=None):
    """
    比较不同温度下生成的样本。
    
    参数：
        samples_dict：温度 → 生成文本的映射字典
        save_path：可选的图片保存路径
    """
    temps = sorted(samples_dict.keys())
    n_temps = len(temps)
    
    fig, ax = plt.subplots(figsize=(12, max(8, n_temps * 1.5)))
    ax.axis('off')
    
    title = "温度对采样的影响"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # 创建文本展示区域
    y_pos = 0.95
    for temp in temps:
        sample = samples_dict[temp]
        
        # 文本过长时截断
        if len(sample) > 200:
            sample = sample[:200] + "..."
        
        # 添加温度标签
        ax.text(0.05, y_pos, f"温度 = {temp:.1f}:",
               fontsize=12, fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.03
        
        # 添加自动换行的样本文本
        import textwrap
        wrapped = textwrap.fill(sample, width=80)
        ax.text(0.05, y_pos, wrapped,
               fontsize=10, family='monospace', transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        y_pos -= 0.08 * (len(wrapped.split('\n')) + 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存到 {save_path}")
    else:
        plt.show()


def plot_learning_rate_comparison(lr_losses_dict, save_path=None):
    """
    比较不同学习率下的训练曲线。
    
    参数：
        lr_losses_dict：学习率 → 损失列表的映射字典
        save_path：可选的图片保存路径
    """
    plt.figure(figsize=(12, 6))
    
    for lr, losses in sorted(lr_losses_dict.items()):
        # 平滑曲线
        if len(losses) > 100:
            window = 100
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(losses)), smoothed, 
                    linewidth=2, label=f'学习率 = {lr}', alpha=0.8)
        else:
            plt.plot(losses, linewidth=2, label=f'学习率 = {lr}', alpha=0.8)
    
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('损失', fontsize=12)
    plt.title('学习率比较', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存到 {save_path}")
    else:
        plt.show()


def visualize_training_run(losses, samples_dict, save_dir=None):
    """
    为一次训练创建综合可视化。
    
    参数：
        losses：训练损失列表
        samples_dict：迭代次数 → 生成样本的映射字典
        save_dir：保存全部图像的目录
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 1. 损失曲线
    save_path = os.path.join(save_dir, 'loss_curve.png') if save_dir else None
    plot_loss_curve(losses, save_path=save_path)
    
    # 2. 样本随训练的变化
    fig, axes = plt.subplots(len(samples_dict), 1, figsize=(12, 3*len(samples_dict)))
    if len(samples_dict) == 1:
        axes = [axes]
    
    for ax, (iteration, sample) in zip(axes, sorted(samples_dict.items())):
        ax.axis('off')
        ax.text(0.05, 0.5, f"迭代 {iteration}:\n{sample[:200]}",
               fontsize=10, family='monospace', transform=ax.transAxes,
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'sample_evolution.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"样本变化图已保存到 {save_dir}")
    else:
        plt.show()
    
    print("可视化完成！")


if __name__ == "__main__":
    print("RNN 可视化工具")
    print("=" * 50)
    print("\n可用函数：")
    print("  - plot_loss_curve")
    print("  - plot_hidden_states")
    print("  - plot_probability_distribution")
    print("  - plot_sequence_predictions")
    print("  - plot_gradient_flow")
    print("  - plot_temperature_comparison")
    print("  - plot_learning_rate_comparison")
    print("  - visualize_training_run")
    print("\n使用示例：")
    print("  from visualization import plot_loss_curve")
    print("  plot_loss_curve(losses, save_path='loss.png')")
