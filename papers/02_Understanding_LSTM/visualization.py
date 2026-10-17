"""
LSTM 网络可视化工具
=====================================

可视化和理解 LSTM 行为的函数：
1. 门激活模式（遗忘门、输入门、输出门）
2. 细胞状态随时间的演化
3. 梯度流比较（LSTM 与基本 RNN）
4. 长程依赖测试
5. 温度对采样的影响
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_gate_activations(gates_dict, sequence_text, save_path=None):
    """
    可视化整个序列上遗忘门、输入门和输出门的激活值。
    
    展示 LSTM 何时选择记忆、遗忘或输出信息。
    
    参数：
        gates_dict：含 'forget'、'input'、'output' 键的字典，每个值
                   都是形状为 (seq_len, hidden_size) 的数组
        sequence_text：与各时间步对应的字符字符串
        save_path：可选的图片保存路径
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    gates = ['forget', 'input', 'output']
    titles = ['遗忘门（1=保留，0=遗忘）',
              '输入门（1=写入，0=忽略）',
              '输出门（1=输出，0=隐藏）']
    
    for ax, gate_name, title in zip(axes, gates, titles):
        gate_data = np.array(gates_dict[gate_name]).T  # (hidden_size, seq_len)
        
        # 绘制热力图
        im = ax.imshow(gate_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # 设置标签
        ax.set_xlabel('时间步（字符）', fontsize=12)
        ax.set_ylabel('隐藏单元', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 字符数不太多时，在 x 轴显示字符
        if len(sequence_text) <= 50:
            ax.set_xticks(range(len(sequence_text)))
            ax.set_xticklabels(list(sequence_text), fontsize=8)
        
        # 颜色条
        plt.colorbar(im, ax=ax, label='门激活值')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存至 {save_path}")
    else:
        plt.show()


def plot_cell_state_evolution(cell_states, sequence_text, save_path=None):
    """
    可视化细胞状态在序列中的演化。
    
    细胞状态是 LSTM 的“记忆”；此图展示哪些信息
    能够跨时间步保留。
    
    参数：
        cell_states：形状为 (seq_len, hidden_size) 的数组
        sequence_text：字符字符串
        save_path：可选的图片保存路径
    """
    plt.figure(figsize=(14, 7))
    
    # 转置后，每行为隐藏单元，每列为时间步
    cell_states_T = np.array(cell_states).T  # (hidden_size, seq_len)
    
    # 绘制热力图
    im = plt.imshow(cell_states_T, cmap='RdBu_r', aspect='auto', 
                    vmin=-1, vmax=1, interpolation='nearest')
    
    plt.xlabel('时间步（字符）', fontsize=12)
    plt.ylabel('隐藏单元', fontsize=12)
    plt.title('LSTM 细胞状态演化\n（红色=正值记忆，蓝色=负值记忆）',
             fontsize=14, fontweight='bold')
    
    # 字符数不太多时，在 x 轴显示字符
    if len(sequence_text) <= 50:
        plt.xticks(range(len(sequence_text)), list(sequence_text), fontsize=9)
    
    plt.colorbar(im, label='细胞状态值')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存至 {save_path}")
    else:
        plt.show()


def plot_gradient_flow_comparison(lstm_grads, rnn_grads=None, save_path=None):
    """
    比较 LSTM 与基本 RNN 中的梯度流。
    
    这有助于理解 LSTM 为什么能够缓解梯度消失。
    
    参数：
        lstm_grads：LSTM 各时间步的梯度范数列表
        rnn_grads：基本 RNN 的梯度范数列表（可选）
        save_path：可选的图片保存路径
    """
    plt.figure(figsize=(12, 6))
    
    time_steps = range(len(lstm_grads))
    
    # 绘制 LSTM 梯度
    plt.plot(time_steps, lstm_grads, marker='o', linewidth=2, 
            label='LSTM', color='green', markersize=4, alpha=0.8)
    
    # 如果提供了 RNN 梯度，则绘制
    if rnn_grads is not None:
        plt.plot(time_steps, rnn_grads, marker='s', linewidth=2, 
                label='Vanilla RNN', color='red', markersize=4, alpha=0.8)
    
    # 参考线
    plt.axhline(y=1.0, color='gray', linestyle='--', 
               label='梯度范数 = 1', alpha=0.5)
    plt.axhline(y=0.1, color='orange', linestyle='--', 
               label='衰减参考阈值（0.1）', alpha=0.5)
    
    plt.xlabel('时间步（反向）', fontsize=12)
    plt.ylabel('梯度范数', fontsize=12)
    plt.title('沿时间的梯度流：LSTM 与 RNN',
             fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存至 {save_path}")
    else:
        plt.show()


def analyze_gate_patterns(gates_dict, sequence_text):
    """
    打印门激活模式的统计分析。
    
    参数：
        gates_dict：包含 'forget'、'input'、'output' 门的字典
        sequence_text：字符字符串
    """
    print("=" * 60)
    print('门激活分析')
    print("=" * 60)
    
    for gate_name in ['forget', 'input', 'output']:
        gate_data = np.array(gates_dict[gate_name])  # (seq_len, hidden_size)
        
        print(f"\n{gate_name.upper()} 门：")
        print(f"  平均激活值：{gate_data.mean():.3f}")
        print(f"  标准差：{gate_data.std():.3f}")
        print(f"  完全开启的比例（>0.9）：{(gate_data > 0.9).mean() * 100:.1f}%")
        print(f"  完全关闭的比例（<0.1）：{(gate_data < 0.1).mean() * 100:.1f}%")
        print(f"  中间区间比例（0.4-0.6）：{((gate_data > 0.4) & (gate_data < 0.6)).mean() * 100:.1f}%")

    # 分析特定位置
    print("\n" + "=" * 60)
    print('逐字符分析')
    print("=" * 60)
    
    forget_data = np.array(gates_dict['forget'])
    input_data = np.array(gates_dict['input'])
    output_data = np.array(gates_dict['output'])
    
    print(f"\n{'字符':<6} | {'遗忘门':<8} | {'输入门':<8} | {'输出门':<8} | {'解释'}")
    print("-" * 70)
    
    for t, char in enumerate(sequence_text[:20]):  # 前 20 个字符
        f_mean = forget_data[t].mean()
        i_mean = input_data[t].mean()
        o_mean = output_data[t].mean()
        
        # 解释结果
        if f_mean > 0.7 and i_mean < 0.3:
            interp = '保留旧信息'
        elif f_mean < 0.3 and i_mean > 0.7:
            interp = '替换记忆'
        elif f_mean > 0.7 and i_mean > 0.7:
            interp = '累积信息'
        elif o_mean > 0.7:
            interp = '较强输出'
        else:
            interp = '混合信号'
        
        char_display = char if char not in ['\n', ' '] else ('\\n' if char == '\n' else '_')
        print(f"{char_display:<6} | {f_mean:>6.3f}   | {i_mean:>6.3f}   | {o_mean:>6.3f}   | {interp}")


def plot_long_range_dependency_test(distances, accuracies, save_path=None):
    """
    绘制长程依赖任务的准确率与距离关系。
    
    测试：“经过 N 步后，LSTM 是否还记得第一个字符？”
    
    参数：
        distances：测试的序列长度列表
        accuracies：各距离上的准确率
        save_path：可选的图片保存路径
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(distances, accuracies, marker='o', linewidth=2, markersize=8, color='blue')
    
    plt.xlabel('距离（中间字符数）', fontsize=12)
    plt.ylabel('准确率（%）', fontsize=12)
    plt.title('长程依赖测试\n“经过 N 步后记住第一个字符”',
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # 添加标注
    if len(distances) > 0:
        best_idx = np.argmax(accuracies)
        plt.annotate(f'最佳：{accuracies[best_idx]:.1f}%，距离 {distances[best_idx]} 步',
                    xy=(distances[best_idx], accuracies[best_idx]),
                    xytext=(distances[best_idx] + 5, accuracies[best_idx] - 10),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存至 {save_path}")
    else:
        plt.show()


def plot_training_comparison(lstm_losses, rnn_losses=None, save_path=None):
    """
    比较 LSTM 与基本 RNN 的训练曲线。
    
    参数：
        lstm_losses：LSTM 损失列表
        rnn_losses：RNN 损失列表（可选）
        save_path：可选的图片保存路径
    """
    plt.figure(figsize=(12, 6))
    
    # 平滑曲线
    window = min(100, len(lstm_losses) // 10)
    if window > 1:
        lstm_smoothed = np.convolve(lstm_losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(lstm_losses)), lstm_smoothed, 
                linewidth=2, label='LSTM', color='green')
    else:
        plt.plot(lstm_losses, linewidth=2, label='LSTM', color='green')
    
    if rnn_losses is not None:
        if window > 1:
            rnn_smoothed = np.convolve(rnn_losses, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rnn_losses)), rnn_smoothed, 
                    linewidth=2, label='Vanilla RNN', color='red', alpha=0.7)
        else:
            plt.plot(rnn_losses, linewidth=2, label='Vanilla RNN', color='red', alpha=0.7)
    
    plt.xlabel('迭代', fontsize=12)
    plt.ylabel('损失', fontsize=12)
    plt.title('训练损失：LSTM 与基本 RNN', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存至 {save_path}")
    else:
        plt.show()


def visualize_attention_to_memory(cell_states, query_indices, sequence_text, save_path=None):
    """
    可视化特定时间步上细胞状态的哪些部分处于“活跃”状态。
    
    帮助理解 LSTM 正在关注哪些信息。
    
    参数：
        cell_states：形状为 (seq_len, hidden_size) 的数组
        query_indices：需要突出显示的时间步列表
        sequence_text：字符字符串
        save_path：可选的图片保存路径
    """
    fig, axes = plt.subplots(len(query_indices), 1, 
                             figsize=(12, 3*len(query_indices)))
    
    if len(query_indices) == 1:
        axes = [axes]
    
    for ax, t in zip(axes, query_indices):
        # 获取当前时间步的细胞状态
        C_t = cell_states[t]  # (hidden_size,)
        
        # 绘制柱状图
        ax.bar(range(len(C_t)), C_t, color=['red' if x < 0 else 'blue' for x in C_t])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        char = sequence_text[t] if t < len(sequence_text) else '?'
        char_display = char if char not in ['\n', ' '] else ('\\n' if char == '\n' else 'SPACE')
        
        ax.set_xlabel('隐藏单元', fontsize=10)
        ax.set_ylabel('细胞状态值', fontsize=10)
        ax.set_title(f"细胞状态，t={t}（字符：'{char_display}')",
                    fontsize=12, fontweight='bold')
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存至 {save_path}")
    else:
        plt.show()


def create_comprehensive_report(lstm_data, save_dir='lstm_analysis'):
    """
    生成包含多种可视化的综合分析报告。

    参数：
        lstm_data：包含以下内容的字典：
            - 'gates'：门激活值字典
            - 'cell_states'：细胞状态数组
            - 'sequence'：文本序列
            - 'losses'：训练损失
            - （可选）'rnn_losses'：用于比较的 RNN 损失
            - （可选）'gradient_norms'：梯度范数
        save_dir：所有图像的保存目录
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"正在以下目录生成 LSTM 综合分析：{save_dir}/...")
    
    # 1. 门激活值
    if 'gates' in lstm_data and 'sequence' in lstm_data:
        print('  - 正在绘制门激活值……')
        plot_gate_activations(lstm_data['gates'], lstm_data['sequence'],
                             save_path=f"{save_dir}/gate_activations.png")
    
    # 2. 细胞状态演化
    if 'cell_states' in lstm_data and 'sequence' in lstm_data:
        print('  - 正在绘制细胞状态演化……')
        plot_cell_state_evolution(lstm_data['cell_states'], lstm_data['sequence'],
                                  save_path=f"{save_dir}/cell_state_evolution.png")
    
    # 3. 训练曲线
    if 'losses' in lstm_data:
        print('  - 正在绘制训练曲线……')
        rnn_losses = lstm_data.get('rnn_losses', None)
        plot_training_comparison(lstm_data['losses'], rnn_losses,
                                save_path=f"{save_dir}/training_comparison.png")
    
    # 4. 梯度流
    if 'gradient_norms' in lstm_data:
        print('  - 正在绘制梯度流……')
        rnn_grads = lstm_data.get('rnn_gradient_norms', None)
        plot_gradient_flow_comparison(lstm_data['gradient_norms'], rnn_grads,
                                     save_path=f"{save_dir}/gradient_flow.png")
    
    # 5. 门分析
    if 'gates' in lstm_data and 'sequence' in lstm_data:
        print('  - 正在分析门模式……')
        with open(f"{save_dir}/gate_analysis.txt", 'w') as f:
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            analyze_gate_patterns(lstm_data['gates'], lstm_data['sequence'])
            sys.stdout = old_stdout
    
    print(f"\n分析完成！结果已保存至 {save_dir}/")
    print(f"  - gate_activations.png")
    print(f"  - cell_state_evolution.png")
    print(f"  - training_comparison.png")
    print(f"  - gradient_flow.png（提供数据时）")
    print(f"  - gate_analysis.txt")


if __name__ == "__main__":
    print('LSTM 可视化工具')
    print("=" * 60)
    print('\n可用函数：')
    print("  - plot_gate_activations()")
    print("  - plot_cell_state_evolution()")
    print("  - plot_gradient_flow_comparison()")
    print("  - analyze_gate_patterns()")
    print("  - plot_long_range_dependency_test()")
    print("  - plot_training_comparison()")
    print("  - visualize_attention_to_memory()")
    print("  - create_comprehensive_report()")
    print('\n用法示例：')
    print("  from visualization import plot_gate_activations")
    print("  plot_gate_activations(gates, text, save_path='gates.png')")
