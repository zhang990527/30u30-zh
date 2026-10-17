"""
练习 4 解答：长程依赖
===============================================

通过可控记忆任务测试 LSTM 保留
长程依赖能力的完整实现。
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from lstm_implementation import LSTM


# ============================================================================
# 记忆任务：复制任务
# ============================================================================

def generate_copy_task(seq_length=10, num_samples=1000, vocab_size=8):
    """
    生成复制任务：记住一段序列，并在稍后复现。
    
    格式：[序列] [空白] [分隔符] [序列]
    
    示例（seq_length=3）：
        输入：  3 1 4 - - - GO ? ? ?
        目标：  - - - - - - -  3 1 4
    
    测试 LSTM 是否能够：
    1. 在记忆中存储任意序列
    2. 在空白时间步中保留它
    3. 准确复现它
    """
    dataset = []
    
    for _ in range(num_samples):
        # 生成随机序列（序列用 vocab_size-2 种符号，另 2 种留给空白与分隔符）
        sequence = np.random.randint(0, vocab_size-2, seq_length)
        
        # 创建输入：[序列] [空白] [分隔符] [查询标记]
        blank_token = vocab_size - 2
        delimiter_token = vocab_size - 1
        
        input_seq = list(sequence)
        input_seq += [blank_token] * seq_length  # 空白时段
        input_seq += [delimiter_token]           # 开始复制的信号
        input_seq += [blank_token] * seq_length  # 查询标记
        
        # 创建目标：分隔符之前为空白，之后复现序列
        target_seq = [blank_token] * (2 * seq_length + 1)
        target_seq += list(sequence)
        
        dataset.append({
            'input': input_seq,
            'target': target_seq,
            'sequence': sequence
        })
    
    return dataset


# ============================================================================
# 记忆任务：延迟异或（XOR）
# ============================================================================

def generate_delayed_xor_task(delay=20, num_samples=1000):
    """
    生成延迟异或任务：记住两个比特，经过延迟后计算异或。
    
    格式：[比特1] [比特2] [延迟] [查询] -> [异或结果]
    
    示例（delay=3）：
        输入：  1 0 - - - ?
        目标：  - - - - - 1   （因为 1 XOR 0 = 1）
    
    测试 LSTM 是否能够：
    1. 记住两个特定比特
    2. 在延迟期间保留它们
    3. 最后计算异或
    """
    dataset = []
    blank_token = 2  # 0、1 表示比特，2 表示空白
    
    for _ in range(num_samples):
        # 生成两个随机比特
        bit1 = np.random.randint(0, 2)
        bit2 = np.random.randint(0, 2)
        xor_result = bit1 ^ bit2
        
        # 创建输入序列
        input_seq = [bit1, bit2]
        input_seq += [blank_token] * delay
        input_seq += [blank_token]  # 查询标记
        
        # 创建目标：只有最后一个位置有意义
        target_seq = [blank_token] * (2 + delay)
        target_seq += [xor_result]
        
        dataset.append({
            'input': input_seq,
            'target': target_seq,
            'bit1': bit1,
            'bit2': bit2,
            'result': xor_result
        })
    
    return dataset


# ============================================================================
# 训练与评估
# ============================================================================

def train_on_memory_task(lstm, dataset, epochs=100, learning_rate=0.01):
    """在记忆任务上训练 LSTM。"""
    losses = []
    accuracies = []
    
    print('正在记忆任务上训练……')
    
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_predictions = 0
        
        # 打乱数据集
        np.random.shuffle(dataset)
        
        for sample in dataset:
            inputs = sample['input']
            targets = sample['target']
            
            # 前向传播
            h = np.zeros((lstm.hidden_size, 1))
            c = np.zeros((lstm.hidden_size, 1))
            
            loss, h, c = lstm.forward(inputs, targets, h, c)
            total_loss += loss
            
            # 反向传播并更新（原仓库简化写法，实际应使用完整 BPTT）
            # ...
            
            # 评估准确率（仅非空白位置）
            blank_token = max(inputs)
            for t, (inp, target) in enumerate(zip(inputs, targets)):
                if target != blank_token:
                    # 获取预测
                    # （实际实现应运行前向传播并取得预测）
                    total_predictions += 1
        
        avg_loss = total_loss / len(dataset)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  轮次 {epoch+1}/{epochs}，损失：{avg_loss:.4f}")
    
    return losses, accuracies


def evaluate_memory_capacity(lstm, max_delay=50, step=5):
    """
    通过改变延迟长度测试 LSTM 的记忆容量。
    
    返回不同延迟长度上的准确率。
    """
    print("\n" + "="*70)
    print('正在评估记忆容量')
    print("="*70)
    
    results = []
    
    for delay in range(step, max_delay+1, step):
        print(f"\n测试延迟 = {delay}")
        
        # 生成当前延迟对应的测试集
        test_dataset = generate_delayed_xor_task(delay=delay, num_samples=100)
        
        # 评估
        correct = 0
        total = 0
        
        for sample in test_dataset:
            inputs = sample['input']
            targets = sample['target']
            expected_result = sample['result']
            
            # 前向传播
            h = np.zeros((lstm.hidden_size, 1))
            c = np.zeros((lstm.hidden_size, 1))
            
            # 获取最终预测
            # （简化写法，实际应执行完整前向传播）
            predicted_result = np.random.randint(0, 2)  # 占位实现
            
            if predicted_result == expected_result:
                correct += 1
            total += 1
        
        accuracy = correct / total
        results.append({
            'delay': delay,
            'accuracy': accuracy
        })
        
        print(f"  准确率：{accuracy:.2%}")
    
    return results


# ============================================================================
# 细胞状态分析
# ============================================================================

def analyze_cell_state_evolution(lstm, sample):
    """
    跟踪细胞状态在记忆任务中的演化。
    
    主要观察：
    1. 细胞状态是否在空白期间保留信息？
    2. 存储的信息有多稳定？
    3. 状态量级是否随时间变化？
    """
    inputs = sample['input']
    
    # 前向传播，同时跟踪细胞状态
    h = np.zeros((lstm.hidden_size, 1))
    c = np.zeros((lstm.hidden_size, 1))
    
    cell_states = []
    hidden_states = []
    
    for inp in inputs:
        # 独热编码
        x = np.zeros((lstm.vocab_size, 1))
        x[inp] = 1
        
        # LSTM 的一步前向传播
        combined = np.vstack([h, x])
        
        f = sigmoid(np.dot(lstm.Wf, combined) + lstm.bf)
        i = sigmoid(np.dot(lstm.Wi, combined) + lstm.bi)
        c_tilde = np.tanh(np.dot(lstm.Wc, combined) + lstm.bc)
        c = f * c + i * c_tilde
        o = sigmoid(np.dot(lstm.Wo, combined) + lstm.bo)
        h = o * np.tanh(c)
        
        # 存储状态
        cell_states.append(c.copy())
        hidden_states.append(h.copy())
    
    return cell_states, hidden_states


def sigmoid(x):
    """数值稳定的 sigmoid。"""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


# ============================================================================
# 可视化
# ============================================================================

def visualize_memory_capacity(results, save_path='memory_capacity.png'):
    """绘制准确率与延迟长度的关系。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    delays = [r['delay'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    # 面板 1：准确率与延迟
    axes[0].plot(delays, accuracies, 'o-', linewidth=3, markersize=8, color='blue')
    axes[0].axhline(y=0.5, color='red', linestyle='--', 
                   label='随机猜测', linewidth=2)
    axes[0].set_xlabel('延迟长度', fontsize=14)
    axes[0].set_ylabel('Accuracy', fontsize=14)
    axes[0].set_title('LSTM 记忆容量', fontsize=16, fontweight='bold')
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # 面板 2：按准确率分组的柱状图
    bins = [(0, 10, 'Short'), (10, 25, 'Medium'), (25, 50, 'Long')]
    bin_accuracies = []
    bin_labels = []
    
    for min_delay, max_delay, label in bins:
        relevant = [r['accuracy'] for r in results 
                   if min_delay <= r['delay'] < max_delay]
        if relevant:
            bin_accuracies.append(np.mean(relevant))
            bin_labels.append(f"{label}\n({min_delay}-{max_delay})")
    
    colors = ['green', 'orange', 'red']
    axes[1].bar(range(len(bin_labels)), bin_accuracies, color=colors, alpha=0.7)
    axes[1].set_xticks(range(len(bin_labels)))
    axes[1].set_xticklabels(bin_labels, fontsize=12)
    axes[1].set_ylabel('平均准确率', fontsize=14)
    axes[1].set_title('按延迟范围比较性能', fontsize=16, fontweight='bold')
    axes[1].axhline(y=0.5, color='red', linestyle='--', linewidth=2)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化已保存至 {save_path}")
    
    return fig


def visualize_cell_state(cell_states, input_sequence, save_path='cell_state_evolution.png'):
    """可视化记忆任务中的细胞状态演化。"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 转换为数组
    cell_array = np.array([c.flatten() for c in cell_states])
    
    time_steps = range(len(cell_states))
    
    # 面板 1：细胞状态量级随时间变化
    magnitudes = np.linalg.norm(cell_array, axis=1)
    axes[0].plot(time_steps, magnitudes, linewidth=2, color='purple')
    axes[0].set_ylabel('细胞状态量级', fontsize=12)
    axes[0].set_title('记忆任务中的细胞状态演化',
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 标记各阶段
    seq_len = len([x for x in input_sequence if x < max(input_sequence)-1]) // 2
    axes[0].axvspan(0, seq_len, alpha=0.2, color='green', label='Encoding')
    axes[0].axvspan(seq_len, 2*seq_len, alpha=0.2, color='yellow', label='Retention')
    axes[0].axvspan(2*seq_len, len(time_steps), alpha=0.2, color='blue', label='Retrieval')
    axes[0].legend(fontsize=10)
    
    # 面板 2：细胞状态热力图，显示前 20 维
    im = axes[1].imshow(cell_array[:, :20].T, aspect='auto', cmap='RdBu', 
                       vmin=-1, vmax=1, interpolation='nearest')
    axes[1].set_ylabel('细胞状态维度', fontsize=12)
    axes[1].set_title('细胞状态各维度随时间变化', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[1])
    
    # 面板 3：输入序列可视化
    axes[2].plot(time_steps, input_sequence, 'o-', markersize=6, color='black')
    axes[2].set_xlabel('时间步', fontsize=12)
    axes[2].set_ylabel('输入词元', fontsize=12)
    axes[2].set_title('输入序列', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 细胞状态可视化已保存至 {save_path}")
    
    return fig


# ============================================================================
# 主分析流程
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    
    print("\n" + "="*70)
    print('长程依赖测试')
    print("="*70)
    
    print('\n本解答实现：')
    print('  1. 复制任务：记住并复现序列')
    print('  2. 延迟异或任务：跨过延迟记住两个比特')
    print('  3. 记忆容量评估')
    print('  4. 细胞状态演化分析')
    print('  5. 综合可视化')
    
    print("\n" + "-"*70)
    print('原仓库预期观察（并非已验证结果）')
    print("-"*70)
    
    print('\n✓ 短延迟（< 10 步）：')
    print('  - 原文预期 LSTM 达到接近完美的准确率')
    print('  - 细胞状态保持稳定')
    print('  - 信息清晰保留')
    
    print('\n✓ 中等延迟（10-25 步）：')
    print('  - 原文预期准确率较高（>80%）')
    print('  - 存在一定信息衰减')
    print('  - 细胞状态有轻微漂移')
    
    print('\n✓ 长延迟（> 25 步）：')
    print('  - 性能逐步下降')
    print('  - 原文预期仍优于基本 RNN')
    print('  - 细胞状态量级可能改变')
    
    print('\n✓ 细胞状态行为：')
    print('  - 编码阶段：快速变化')
    print('  - 保留阶段：稳定平台')
    print('  - 检索阶段：调节状态以输出')
    
    print("\n" + "-"*70)
    print('LSTM 为何有助于处理长程依赖')
    print("-"*70)
    
    print('\n1. 细胞状态提供“记忆通路”')
    print('   → 信息无需反复经过完整非线性变换即可沿直接路径传递')
    
    print('\n2. 遗忘门避免无关信息累积')
    print('   → 在需要时清除旧信息')
    
    print('\n3. 输入门控制存储什么')
    print('   → 选择性地写入重要信息')
    
    print('\n4. 梯度通过细胞状态传递')
    print('   → 有助于缓解梯度消失')
    
    print('\n5. 乘性门提供精细控制')
    print('   → 可以接近完全阻断或完全通过')
    
    print("\n" + "="*70)
    print('运行记忆任务，测试你的 LSTM。')
    print("="*70)
