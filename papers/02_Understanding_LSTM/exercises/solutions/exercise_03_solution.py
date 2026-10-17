"""
练习 3 解答：消融研究
======================================

比较 LSTM 与修改版本的完整实现。
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('..')

from lstm_implementation import LSTM


# ============================================================================
# LSTM 变体
# ============================================================================

class VanillaRNN:
    """没有门控的标准 RNN（基线）。"""
    
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # 初始化参数
        scale = 0.01
        self.Wxh = np.random.randn(hidden_size, vocab_size) * scale
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale
        self.Why = np.random.randn(vocab_size, hidden_size) * scale
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))
    
    def forward(self, inputs, targets, h_prev):
        """前向传播。"""
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0
        
        for t, (inp, target) in enumerate(zip(inputs, targets)):
            # 独热编码
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inp] = 1
            
            # 隐藏状态：h = tanh(Wxh*x + Whh*h_prev)
            hs[t] = np.tanh(
                np.dot(self.Wxh, xs[t]) + 
                np.dot(self.Whh, hs[t-1]) + 
                self.bh
            )
            
            # 输出
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            
            # 损失
            loss += -np.log(ps[t][target, 0])
        
        return loss, hs[len(inputs)-1]
    
    def get_description(self):
        return '基本 RNN：h_t = tanh(Wx*x_t + Wh*h_{t-1})'


class LSTMNoForgetGate:
    """无遗忘门的 LSTM（f 始终为 1）。"""
    
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # 初始化参数（无遗忘门）
        scale = 0.01
        input_size = hidden_size + vocab_size
        
        self.Wi = np.random.randn(hidden_size, input_size) * scale
        self.Wc = np.random.randn(hidden_size, input_size) * scale
        self.Wo = np.random.randn(hidden_size, input_size) * scale
        
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        
        self.Why = np.random.randn(vocab_size, hidden_size) * scale
        self.by = np.zeros((vocab_size, 1))
    
    def forward(self, inputs, targets, h_prev, c_prev):
        """无遗忘门的前向传播。"""
        xs, hs, cs, ys, ps = {}, {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        cs[-1] = np.copy(c_prev)
        loss = 0
        
        for t, (inp, target) in enumerate(zip(inputs, targets)):
            # 独热编码
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inp] = 1
            
            # 拼接 h 与 x
            combined = np.vstack([hs[t-1], xs[t]])
            
            # 门与候选状态（无遗忘门）
            i = sigmoid(np.dot(self.Wi, combined) + self.bi)
            c_tilde = np.tanh(np.dot(self.Wc, combined) + self.bc)
            cs[t] = cs[t-1] + i * c_tilde  # f = 1（始终保留旧细胞状态）
            o = sigmoid(np.dot(self.Wo, combined) + self.bo)
            hs[t] = o * np.tanh(cs[t])
            
            # 输出
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            
            # 损失
            loss += -np.log(ps[t][target, 0])
        
        return loss, hs[len(inputs)-1], cs[len(inputs)-1]
    
    def get_description(self):
        return '无遗忘门的 LSTM：c_t = c_{t-1} + i_t * c̃_t'


class LSTMNoInputGate:
    """无输入门的 LSTM（i 始终为 1）。"""
    
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # 初始化参数（无输入门）
        scale = 0.01
        input_size = hidden_size + vocab_size
        
        self.Wf = np.random.randn(hidden_size, input_size) * scale
        self.Wc = np.random.randn(hidden_size, input_size) * scale
        self.Wo = np.random.randn(hidden_size, input_size) * scale
        
        self.bf = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        
        self.Why = np.random.randn(vocab_size, hidden_size) * scale
        self.by = np.zeros((vocab_size, 1))
    
    def forward(self, inputs, targets, h_prev, c_prev):
        """无输入门的前向传播。"""
        xs, hs, cs, ys, ps = {}, {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        cs[-1] = np.copy(c_prev)
        loss = 0
        
        for t, (inp, target) in enumerate(zip(inputs, targets)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inp] = 1
            
            combined = np.vstack([hs[t-1], xs[t]])
            
            # 门与候选状态（无输入门）
            f = sigmoid(np.dot(self.Wf, combined) + self.bf)
            c_tilde = np.tanh(np.dot(self.Wc, combined) + self.bc)
            cs[t] = f * cs[t-1] + c_tilde  # i = 1（始终接受新候选值）
            o = sigmoid(np.dot(self.Wo, combined) + self.bo)
            hs[t] = o * np.tanh(cs[t])
            
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            
            loss += -np.log(ps[t][target, 0])
        
        return loss, hs[len(inputs)-1], cs[len(inputs)-1]
    
    def get_description(self):
        return '无输入门的 LSTM：c_t = f_t * c_{t-1} + c̃_t'


class LSTMNoOutputGate:
    """无输出门的 LSTM（o 始终为 1）。"""
    
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # 初始化参数（无输出门）
        scale = 0.01
        input_size = hidden_size + vocab_size
        
        self.Wf = np.random.randn(hidden_size, input_size) * scale
        self.Wi = np.random.randn(hidden_size, input_size) * scale
        self.Wc = np.random.randn(hidden_size, input_size) * scale
        
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        
        self.Why = np.random.randn(vocab_size, hidden_size) * scale
        self.by = np.zeros((vocab_size, 1))
    
    def forward(self, inputs, targets, h_prev, c_prev):
        """无输出门的前向传播。"""
        xs, hs, cs, ys, ps = {}, {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        cs[-1] = np.copy(c_prev)
        loss = 0
        
        for t, (inp, target) in enumerate(zip(inputs, targets)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inp] = 1
            
            combined = np.vstack([hs[t-1], xs[t]])
            
            # 门与候选状态（无输出门）
            f = sigmoid(np.dot(self.Wf, combined) + self.bf)
            i = sigmoid(np.dot(self.Wi, combined) + self.bi)
            c_tilde = np.tanh(np.dot(self.Wc, combined) + self.bc)
            cs[t] = f * cs[t-1] + i * c_tilde
            hs[t] = np.tanh(cs[t])  # o = 1（始终输出经 tanh 变换后的完整细胞状态）
            
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            
            loss += -np.log(ps[t][target, 0])
        
        return loss, hs[len(inputs)-1], cs[len(inputs)-1]
    
    def get_description(self):
        return '无输出门的 LSTM：h_t = tanh(c_t)'


def sigmoid(x):
    """数值稳定的 sigmoid。"""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


# ============================================================================
# 消融研究
# ============================================================================

def run_ablation_study(data, char_to_idx, hidden_size=64, seq_length=25, 
                       num_iterations=1000, sample_every=100):
    """
    训练 5 种模型并比较性能。

    变体：
    1. 完整 LSTM（基线）
    2. 基本 RNN
    3. 无遗忘门 LSTM
    4. 无输入门 LSTM
    5. 无输出门 LSTM
    """
    print("="*70)
    print('LSTM 消融研究')
    print("="*70)
    
    vocab_size = len(char_to_idx)
    
    # 初始化所有模型
    models = {
        'Full LSTM': LSTM(vocab_size, hidden_size, vocab_size),
        'Vanilla RNN': VanillaRNN(vocab_size, hidden_size),
        'No Forget Gate': LSTMNoForgetGate(vocab_size, hidden_size),
        'No Input Gate': LSTMNoInputGate(vocab_size, hidden_size),
        'No Output Gate': LSTMNoOutputGate(vocab_size, hidden_size)
    }
    
    # 记录指标
    results = {name: {'losses': [], 'times': []} for name in models}
    
    # 训练各模型
    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"正在训练：{name}")
        print(f"描述：{model.get_description()}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # 训练循环
        for iteration in range(num_iterations):
            # 随机选择位置
            p = np.random.randint(0, len(data) - seq_length - 1)
            inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]]
            targets = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]
            
            # 前向传播（RNN 与 LSTM 变体接口不同）
            if name == 'Vanilla RNN':
                h = np.zeros((hidden_size, 1))
                loss, h = model.forward(inputs, targets, h)
            else:
                h = np.zeros((hidden_size, 1))
                c = np.zeros((hidden_size, 1))
                loss, h, c = model.forward(inputs, targets, h, c)
            
            # 记录日志
            results[name]['losses'].append(loss)
            
            if (iteration + 1) % sample_every == 0:
                print(f"  迭代 {iteration+1}/{num_iterations}，损失：{loss:.4f}")
        
        results[name]['times'].append(time.time() - start_time)
        print(f"  训练耗时：{results[name]['times'][-1]:.2f}s")
    
    return results


def analyze_results(results):
    """分析并比较消融研究结果。"""
    print("\n" + "="*70)
    print('结果分析')
    print("="*70)
    
    for name, metrics in results.items():
        losses = metrics['losses']
        final_loss = np.mean(losses[-100:])  # 最后 100 个值的平均值
        
        print(f"\n{name}:")
        print(f"  最终损失：{final_loss:.4f}")
        print(f"  训练耗时：{metrics['times'][0]:.2f}s")
        print(f"  收敛：{'快' if final_loss < 1.5 else '慢'}")
    
    # 按性能排名
    print("\n" + "-"*70)
    print('排名（按最终损失）：')
    print("-"*70)
    
    ranking = sorted(results.items(), 
                    key=lambda x: np.mean(x[1]['losses'][-100:]))
    
    for rank, (name, metrics) in enumerate(ranking, 1):
        final_loss = np.mean(metrics['losses'][-100:])
        print(f"  {rank}. {name:20s} - 损失：{final_loss:.4f}")


def visualize_comparison(results, save_path='ablation_comparison.png'):
    """创建所有变体的比较图。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {
        'Full LSTM': 'blue',
        'Vanilla RNN': 'red',
        'No Forget Gate': 'green',
        'No Input Gate': 'orange',
        'No Output Gate': 'purple'
    }
    
    # 面板 1：损失曲线
    ax = axes[0, 0]
    for name, metrics in results.items():
        losses = metrics['losses']
        # 使用移动平均平滑
        window = 50
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=name, color=colors[name], linewidth=2, alpha=0.8)
    
    ax.set_xlabel('迭代', fontsize=12)
    ax.set_ylabel('损失（平滑后）', fontsize=12)
    ax.set_title('训练损失比较', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 面板 2：最终损失比较（柱状图）
    ax = axes[0, 1]
    names = list(results.keys())
    final_losses = [np.mean(metrics['losses'][-100:]) for metrics in results.values()]
    
    bars = ax.bar(range(len(names)), final_losses, color=[colors[n] for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('最终损失', fontsize=12)
    ax.set_title('最终性能比较', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 面板 3：训练速度
    ax = axes[1, 0]
    times = [metrics['times'][0] for metrics in results.values()]
    bars = ax.bar(range(len(names)), times, color=[colors[n] for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('训练耗时（秒）', fontsize=12)
    ax.set_title('训练速度比较', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 面板 4：文字概述
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = '原仓库预期结论（需实际实验核验）：\n\n'
    summary_text += '1. 完整 LSTM 表现最好\n'
    summary_text += '   → 三个门都重要\n\n'
    summary_text += '2. 遗忘门最关键\n'
    summary_text += '   → 移除后性能下降\n'
    summary_text += '   → 细胞状态不断累积\n\n'
    summary_text += '3. 输入门有助于防止饱和\n'
    summary_text += '   → 控制信息流\n\n'
    summary_text += '4. 输出门调节信息暴露\n'
    summary_text += '   → 原文称其防止梯度消失；该因果表述需谨慎\n\n'
    summary_text += '5. 基本 RNN 表现最差\n'
    summary_text += '   → 原文据此强调门控作用\n'
    
    ax.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
           family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化已保存至 {save_path}")
    
    return fig


# ============================================================================
# 主流程
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    
    print("\n" + "="*70)
    print('LSTM 消融研究')
    print("="*70)
    
    print('\n本解答比较 5 种模型：')
    print('  1. 完整 LSTM（三个门）')
    print('  2. 基本 RNN（无门）')
    print('  3. 无遗忘门的 LSTM')
    print('  4. 无输入门的 LSTM')
    print('  5. 无输出门的 LSTM')
    
    print("\n" + "-"*70)
    print('原仓库预期观察（并非已验证结果）')
    print("-"*70)
    
    print('\n✓ 遗忘门：原文认为最重要')
    print('  - 没有它可能出现梯度爆炸')
    print('  - 细胞状态可能无界增长')
    print('  - 无法主动遗忘无关信息')
    
    print('\n✓ 输入门：帮助避免饱和')
    print('  - 没有它则写入不受门控')
    print('  - 细胞状态可能剧烈变化')
    print('  - 训练可能不稳定')
    
    print('\n✓ 输出门：调节信息暴露')
    print('  - 没有它则输出所有经 tanh 变换的细胞信息')
    print('  - 对隐藏状态的控制减少')
    print('  - 原文预期性能适度下降')
    
    print("\n" + "="*70)
    print('运行实验，取得定量比较结果。')
    print("="*70)
