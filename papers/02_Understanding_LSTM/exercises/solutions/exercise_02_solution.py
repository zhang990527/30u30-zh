"""
练习 2 解答：门分析
=====================================

分析 LSTM 门行为的完整实现。
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from lstm_implementation import LSTM


def capture_gate_activations(lstm, text, char_to_idx):
    """
    使用 LSTM 处理文本，记录所有门的激活值。
    
    返回：
        字典键为 'forget'、'input'、'cell_candidate'、'output'、'cell'、'hidden'
        每个值都是随时间变化的激活值列表。
    """
    # 初始化状态
    h = np.zeros(lstm.hidden_size)
    c = np.zeros(lstm.hidden_size)
    
    # 存储激活值
    activations = {
        'forget': [],
        'input': [],
        'cell_candidate': [],
        'output': [],
        'cell': [],
        'hidden': []
    }
    
    # 处理每个字符
    for char in text:
        if char not in char_to_idx:
            continue
        
        idx = char_to_idx[char]
        x = np.zeros(lstm.vocab_size)
        x[idx] = 1
        
        # 拼接输入与隐藏状态
        combined = np.concatenate([h, x])
        
        # 计算各门与候选状态
        f = sigmoid(np.dot(lstm.Wf, combined) + lstm.bf)  # 遗忘门
        i = sigmoid(np.dot(lstm.Wi, combined) + lstm.bi)  # 输入门
        c_tilde = np.tanh(np.dot(lstm.Wc, combined) + lstm.bc)  # 候选细胞状态
        c = f * c + i * c_tilde  # 新的细胞状态
        o = sigmoid(np.dot(lstm.Wo, combined) + lstm.bo)  # 输出门
        h = o * np.tanh(c)  # 新的隐藏状态
        
        # 存储激活值（在隐藏维度上取平均）
        activations['forget'].append(np.mean(f))
        activations['input'].append(np.mean(i))
        activations['cell_candidate'].append(np.mean(c_tilde))
        activations['output'].append(np.mean(o))
        activations['cell'].append(np.mean(c))
        activations['hidden'].append(np.mean(h))
    
    return activations


def sigmoid(x):
    """数值稳定的 sigmoid。"""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def analyze_gate_patterns(activations, text):
    """
    分析门激活模式。
    
    需要分析的内容：
    1. 遗忘门何时激活？遗忘了什么？
    2. 输入门何时激活？存储了什么？
    3. 各门如何协作？
    4. 与输入文本结构的关系
    """
    print("="*70)
    print('门激活分析')
    print("="*70)
    
    # 1. 基本统计量
    print('\n[1] 门统计量')
    print("-" * 70)
    for gate_name in ['forget', 'input', 'output']:
        values = activations[gate_name]
        print(f"\n{gate_name.upper()} 门：")
        print(f"  均值：{np.mean(values):.3f}")
        print(f"  标准差：{np.std(values):.3f}")
        print(f"  最小值：{np.min(values):.3f}")
        print(f"  最大值：{np.max(values):.3f}")

    # 2. 相关性分析
    print('\n[2] 门相关性')
    print("-" * 70)
    
    forget = np.array(activations['forget'])
    input_gate = np.array(activations['input'])
    output = np.array(activations['output'])
    
    # 遗忘门与输入门（原文预期为负相关，需用数据验证）
    corr_fi = np.corrcoef(forget, input_gate)[0, 1]
    print(f"遗忘门 ↔ 输入门：{corr_fi:+.3f}")
    if corr_fi < -0.3:
        print('  ✓ 各门呈互补关系：写入新信息时遗忘旧信息')
    else:
        print('  原文预期各门应有更强负相关；该预期并非理论保证')
    
    # 输入门与输出门
    corr_io = np.corrcoef(input_gate, output)[0, 1]
    print(f"输入门 ↔ 输出门：{corr_io:+.3f}")
    
    # 3. 识别关键时刻
    print('\n[3] 关键时刻')
    print("-" * 70)
    
    # 遗忘门值较高（>0.8）时；高门值实际表示保留较多
    high_forget = np.where(forget > 0.8)[0]
    if len(high_forget) > 0:
        print(f"\n遗忘门高值出现在 {len(high_forget)} 个位置：")
        for idx in high_forget[:5]:  # 显示前 5 项
            if idx < len(text):
                context = text[max(0, idx-10):idx+10]
                print(f"  位置 {idx}: ...{context}...")
    
    # 输入门值较高（>0.8）时
    high_input = np.where(input_gate > 0.8)[0]
    if len(high_input) > 0:
        print(f"\n输入门高值出现在 {len(high_input)} 个位置：")
        for idx in high_input[:5]:
            if idx < len(text):
                context = text[max(0, idx-10):idx+10]
                print(f"  位置 {idx}: ...{context}...")
    
    # 4. 结构模式
    print('\n[4] 结构模式')
    print("-" * 70)
    
    # 检查句子边界处的活动
    sentence_ends = [i for i, char in enumerate(text) if char in '.!?']
    
    if sentence_ends:
        avg_forget_at_end = np.mean([forget[i] for i in sentence_ends if i < len(forget)])
        avg_forget_overall = np.mean(forget)
        
        print(f"\n句末的遗忘门值：{avg_forget_at_end:.3f}")
        print(f"整体遗忘门值：{avg_forget_overall:.3f}")
        
        if avg_forget_at_end > avg_forget_overall + 0.1:
            print('  ✓ 原文判断：LSTM 在句子边界遗忘更多；需注意遗忘门高值实际表示保留')
        else:
            print('  → LSTM 未在句末呈现明显的遗忘变化')


def visualize_gates(activations, text, save_path='gate_analysis.png'):
    """
    创建门激活值的综合四面板可视化。
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    time_steps = range(len(activations['forget']))
    
    # 面板 1：三个门
    axes[0].plot(time_steps, activations['forget'], label='遗忘门',
                color='red', alpha=0.7, linewidth=1.5)
    axes[0].plot(time_steps, activations['input'], label='输入门',
                color='green', alpha=0.7, linewidth=1.5)
    axes[0].plot(time_steps, activations['output'], label='输出门',
                color='blue', alpha=0.7, linewidth=1.5)
    axes[0].set_ylabel('门激活值', fontsize=12)
    axes[0].set_title('门激活值随时间变化', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # 面板 2：细胞状态演化
    axes[1].plot(time_steps, activations['cell'], color='purple', linewidth=2)
    axes[1].set_ylabel('细胞状态', fontsize=12)
    axes[1].set_title('细胞状态演化', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 面板 3：门的互补性（遗忘门与输入门）
    axes[2].plot(time_steps, activations['forget'], label='遗忘门',
                color='red', alpha=0.6, linewidth=1.5)
    axes[2].plot(time_steps, activations['input'], label='输入门',
                color='green', alpha=0.6, linewidth=1.5)
    axes[2].plot(time_steps, 
                np.array(activations['forget']) + np.array(activations['input']),
                label='遗忘门 + 输入门', color='black', linestyle='--', linewidth=2)
    axes[2].set_ylabel('Activation', fontsize=12)
    axes[2].set_title('门的互补性（遗忘门与输入门）', fontsize=14, fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 2])
    
    # 面板 4：隐藏状态活动
    axes[3].plot(time_steps, activations['hidden'], color='orange', linewidth=2)
    axes[3].set_ylabel('隐藏状态', fontsize=12)
    axes[3].set_xlabel('时间步', fontsize=12)
    axes[3].set_title('隐藏状态活动', fontsize=14, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化已保存至 {save_path}")
    
    return fig


def compare_gates_across_contexts(lstm, char_to_idx):
    """
    比较不同上下文中的门行为。
    
    测试用例：
    1. 重复文本："the the the the"
    2. 长程依赖："The dog, which was big, barked."
    3. 上下文变化："It was sunny. It was rainy."
    """
    test_cases = {
        'Repetitive': "the the the the the the",
        'Long-range dependency': "The cat that was sleeping on the mat stood up",
        'Context switch': "It was hot. Now it is cold."
    }
    
    print("\n" + "="*70)
    print('不同上下文中的门行为')
    print("="*70)
    
    for name, text in test_cases.items():
        print(f"\n{name.upper()}: '{text}'")
        print("-" * 70)
        
        activations = capture_gate_activations(lstm, text, char_to_idx)
        
        forget_vals = activations['forget']
        input_vals = activations['input']
        
        print(f"遗忘门——均值：{np.mean(forget_vals):.3f}，标准差：{np.std(forget_vals):.3f}")
        print(f"输入门——均值：{np.mean(input_vals):.3f}，标准差：{np.std(input_vals):.3f}")
        
        # 分析
        if 'Repetitive' in name:
            if np.mean(forget_vals) > 0.7:
                print('  → 原文判断：遗忘门高值表示丢弃重复信息 ✓；核对：高值实际表示保留')
        
        elif 'Long-range' in name:
            if np.std(input_vals) > 0.1:
                print('  → 输入门变化：LSTM 有选择地存储信息 ✓')
        
        elif 'Context switch' in name:
            mid = len(forget_vals) // 2
            early_forget = np.mean(forget_vals[:mid])
            late_forget = np.mean(forget_vals[mid:])
            if late_forget > early_forget + 0.1:
                print('  → 原文判断：上下文切换后遗忘更多 ✓；核对：此条件实际检测门值升高')


# ============================================================================
# 主分析流程
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    
    print("\n" + "="*70)
    print('LSTM 门分析工具集')
    print("="*70)
    
    print('\n本解答提供：')
    print('  1. 前向传播时记录门激活值')
    print('  2. 门行为的统计分析')
    print('  3. 门之间的相关性分析')
    print('  4. 识别关键时刻，即高激活时刻')
    print('  5. 结构模式检测，例如句子边界')
    print('  6. 四面板可视化')
    print('  7. 跨上下文比较')
    
    print("\n" + "-"*70)
    print('原仓库的门分析预期（需实验核验）')
    print("-"*70)
    
    print('\n1. 遗忘门：')
    print('   • 原文称旧信息无关时激活；核对：门值低才表示遗忘较多')
    print('   • 原文预期句子边界处门值高')
    print('   • 原文预期上下文切换时门值高')
    print('   • 原文预期与输入门负相关')
    
    print('\n2. 输入门：')
    print('   • 新的重要信息出现时激活')
    print('   • 实词（名词、动词）处门值高')
    print('   • 功能词（the、a、of）处门值低')
    print('   • 与遗忘门互补')
    
    print('\n3. 输出门：')
    print('   • 控制输出哪些信息')
    print('   • 预测较确定时门值高')
    print('   • 调节隐藏状态量级')
    
    print('\n4. 细胞状态：')
    print('   • 随时间累积信息')
    print('   • 在有意义的上下文中保持稳定')
    print('   • 在主题边界处重置')
    
    print("\n" + "="*70)
    print('在训练好的 LSTM 上运行这些函数，观察门行为。')
    print("="*70)
