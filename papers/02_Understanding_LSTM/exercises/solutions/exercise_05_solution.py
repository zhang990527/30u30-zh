"""
练习 5 解答：LSTM 与 GRU 对比
===================================

GRU 的完整实现，以及与 LSTM 的详细比较。
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('..')

from lstm_implementation import LSTM


# ============================================================================
# GRU 实现
# ============================================================================

class GRU:
    """
    门控循环单元（GRU）的实现。

    GRU 是 LSTM 的简化变体，有两个门，而 LSTM 有三个门：
    - 重置门（r）：控制在候选状态计算中使用多少过去信息
    - 更新门（z）：控制接受多少新信息

    与 LSTM 的主要区别：
    - 没有独立细胞状态，只有隐藏状态
    - 参数更少，训练通常更快
    - 表现通常与 LSTM 接近

    方程：
        z_t = σ(W_z [h_{t-1}, x_t])      # 更新门
        r_t = σ(W_r [h_{t-1}, x_t])      # 重置门
        h̃_t = tanh(W [r_t ⊙ h_{t-1}, x_t])  # 候选隐藏状态
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  # 新隐藏状态
    """
    
    def __init__(self, vocab_size, hidden_size, output_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化参数
        scale = 0.01
        input_size = hidden_size + vocab_size
        
        # 更新门参数
        self.Wz = np.random.randn(hidden_size, input_size) * scale
        self.bz = np.zeros((hidden_size, 1))
        
        # 重置门参数
        self.Wr = np.random.randn(hidden_size, input_size) * scale
        self.br = np.zeros((hidden_size, 1))
        
        # 候选隐藏状态参数
        self.Wh = np.random.randn(hidden_size, input_size) * scale
        self.bh = np.zeros((hidden_size, 1))
        
        # 输出层参数
        self.Why = np.random.randn(output_size, hidden_size) * scale
        self.by = np.zeros((output_size, 1))
    
    def forward(self, inputs, targets, h_prev):
        """
        通过 GRU 前向传播。
        
        参数：
            inputs：输入索引列表
            targets：目标索引列表
            h_prev：上一隐藏状态
            
        返回：
            loss：交叉熵损失
            h：最终隐藏状态
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        zs, rs, h_tildes = {}, {}, {}  # 存储门值与候选状态
        
        hs[-1] = np.copy(h_prev)
        loss = 0
        
        for t, (inp, target) in enumerate(zip(inputs, targets)):
            # 对输入进行独热编码
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inp] = 1
            
            # 拼接上一隐藏状态与输入
            combined = np.vstack([hs[t-1], xs[t]])
            
            # 更新门：z_t = σ(W_z [h_{t-1}, x_t])
            zs[t] = sigmoid(np.dot(self.Wz, combined) + self.bz)
            
            # 重置门：r_t = σ(W_r [h_{t-1}, x_t])
            rs[t] = sigmoid(np.dot(self.Wr, combined) + self.br)
            
            # 候选隐藏状态：h̃_t = tanh(W [r_t ⊙ h_{t-1}, x_t])
            reset_combined = np.vstack([rs[t] * hs[t-1], xs[t]])
            h_tildes[t] = np.tanh(np.dot(self.Wh, reset_combined) + self.bh)
            
            # 新隐藏状态：h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
            hs[t] = (1 - zs[t]) * hs[t-1] + zs[t] * h_tildes[t]
            
            # 输出层
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            
            # Softmax
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            
            # 交叉熵损失
            loss += -np.log(ps[t][target, 0])
        
        # 存储反向传播所需的中间值
        self.cache = (xs, hs, zs, rs, h_tildes, ys, ps, targets)
        
        return loss, hs[len(inputs)-1]
    
    def backward(self):
        """
        通过 GRU 执行反向传播（BPTT）。
        
        返回梯度字典。
        """
        xs, hs, zs, rs, h_tildes, ys, ps, targets = self.cache
        
        # 初始化梯度
        dWz, dbz = np.zeros_like(self.Wz), np.zeros_like(self.bz)
        dWr, dbr = np.zeros_like(self.Wr), np.zeros_like(self.br)
        dWh, dbh = np.zeros_like(self.Wh), np.zeros_like(self.bh)
        dWhy, dby = np.zeros_like(self.Why), np.zeros_like(self.by)
        
        dh_next = np.zeros_like(hs[0])
        
        # 沿时间反向传播
        for t in reversed(range(len(targets))):
            # 输出层梯度
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            
            # 隐藏状态梯度
            dh = np.dot(self.Why.T, dy) + dh_next
            
            # 候选隐藏状态梯度
            dh_tilde = dh * zs[t]
            dh_tilde_raw = dh_tilde * (1 - h_tildes[t]**2)  # tanh 导数
            
            combined_reset = np.vstack([rs[t] * hs[t-1], xs[t]])
            dWh += np.dot(dh_tilde_raw, combined_reset.T)
            dbh += dh_tilde_raw
            
            # 重置门梯度
            dr = np.dot(self.Wh.T, dh_tilde_raw)[:self.hidden_size] * hs[t-1]
            dr_raw = dr * rs[t] * (1 - rs[t])  # sigmoid 导数
            
            combined = np.vstack([hs[t-1], xs[t]])
            dWr += np.dot(dr_raw, combined.T)
            dbr += dr_raw
            
            # 更新门梯度
            dz = dh * (h_tildes[t] - hs[t-1])
            dz_raw = dz * zs[t] * (1 - zs[t])  # sigmoid 导数
            
            dWz += np.dot(dz_raw, combined.T)
            dbz += dz_raw
            
            # 传回上一时间步的梯度
            dh_next = dh * (1 - zs[t])
            dh_next += np.dot(self.Wz.T, dz_raw)[:self.hidden_size]
            dh_next += np.dot(self.Wr.T, dr_raw)[:self.hidden_size]
            dh_next += np.dot(self.Wh.T, dh_tilde_raw)[:self.hidden_size] * rs[t]
        
        # 裁剪梯度
        for dparam in [dWz, dWr, dWh, dWhy, dbz, dbr, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        return {
            'dWz': dWz, 'dbz': dbz,
            'dWr': dWr, 'dbr': dbr,
            'dWh': dWh, 'dbh': dbh,
            'dWhy': dWhy, 'dby': dby
        }
    
    def parameter_count(self):
        """统计参数总数。"""
        count = 0
        for param in [self.Wz, self.bz, self.Wr, self.br, 
                     self.Wh, self.bh, self.Why, self.by]:
            count += param.size
        return count


def sigmoid(x):
    """数值稳定的 sigmoid。"""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


# ============================================================================
# 比较研究
# ============================================================================

def compare_lstm_gru(data, char_to_idx, hidden_size=128, seq_length=25,
                    num_iterations=1000, learning_rate=0.01):
    """
    在同一数据上训练 LSTM 与 GRU 并比较。

    比较指标：
    1. 训练损失收敛
    2. 训练速度：每次迭代耗时
    3. 参数量
    4. 内存用量
    5. 生成质量
    """
    print("="*70)
    print('LSTM 与 GRU 比较')
    print("="*70)
    
    vocab_size = len(char_to_idx)
    
    # 初始化模型
    lstm = LSTM(vocab_size, hidden_size, vocab_size)
    gru = GRU(vocab_size, hidden_size, vocab_size)
    
    # 比较参数量
    print('\n[1] 参数量')
    print("-" * 70)
    print(f"LSTM 参数量：{lstm.parameter_count():,}")
    print(f"GRU 参数量：{gru.parameter_count():,}")
    print(f"差异：{lstm.parameter_count() - gru.parameter_count():,}")
    print(f"GRU 的参数少 {(1 - gru.parameter_count()/lstm.parameter_count())*100:.1f}%")
    
    # 比较训练
    print('\n[2] 训练比较')
    print("-" * 70)
    
    results = {
        'LSTM': {'losses': [], 'times': []},
        'GRU': {'losses': [], 'times': []}
    }
    
    # 训练 LSTM
    print('\n正在训练 LSTM……')
    lstm_start = time.time()
    
    for iteration in range(num_iterations):
        # 抽取一批数据
        p = np.random.randint(0, len(data) - seq_length - 1)
        inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]
        
        # 前向传播
        h = np.zeros((hidden_size, 1))
        c = np.zeros((hidden_size, 1))
        loss, h, c = lstm.forward(inputs, targets, h, c)
        
        # 反向传播
        grads = lstm.backward()
        
        # 更新（原仓库的简化写法，实际应使用合适的优化器）
        for param_name in ['Wf', 'Wi', 'Wc', 'Wo', 'Why', 
                          'bf', 'bi', 'bc', 'bo', 'by']:
            param = getattr(lstm, param_name)
            grad = grads['d' + param_name]
            param -= learning_rate * grad
        
        results['LSTM']['losses'].append(loss)
        
        if (iteration + 1) % 100 == 0:
            print(f"  迭代 {iteration+1}/{num_iterations}，损失：{loss:.4f}")
    
    lstm_time = time.time() - lstm_start
    results['LSTM']['times'].append(lstm_time)
    print(f"LSTM 训练耗时：{lstm_time:.2f}s")
    
    # 训练 GRU
    print('\n正在训练 GRU……')
    gru_start = time.time()
    
    for iteration in range(num_iterations):
        # 抽取一批数据
        p = np.random.randint(0, len(data) - seq_length - 1)
        inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]
        
        # 前向传播
        h = np.zeros((hidden_size, 1))
        loss, h = gru.forward(inputs, targets, h)
        
        # 反向传播
        grads = gru.backward()
        
        # 更新
        for param_name in ['Wz', 'Wr', 'Wh', 'Why', 'bz', 'br', 'bh', 'by']:
            param = getattr(gru, param_name)
            grad = grads['d' + param_name]
            param -= learning_rate * grad
        
        results['GRU']['losses'].append(loss)
        
        if (iteration + 1) % 100 == 0:
            print(f"  迭代 {iteration+1}/{num_iterations}，损失：{loss:.4f}")
    
    gru_time = time.time() - gru_start
    results['GRU']['times'].append(gru_time)
    print(f"GRU 训练耗时：{gru_time:.2f}s")
    
    return results, lstm, gru


def analyze_comparison(results):
    """分析并报告比较结果。"""
    print("\n" + "="*70)
    print('比较分析')
    print("="*70)
    
    lstm_losses = results['LSTM']['losses']
    gru_losses = results['GRU']['losses']
    
    print('\n[1] 最终性能')
    print("-" * 70)
    print(f"LSTM 最终损失：{np.mean(lstm_losses[-100:]):.4f}")
    print(f"GRU 最终损失：{np.mean(gru_losses[-100:]):.4f}")
    
    diff = np.mean(lstm_losses[-100:]) - np.mean(gru_losses[-100:])
    if abs(diff) < 0.1:
        print('→ 表现接近！')
    elif diff > 0:
        print(f"→ GRU 的损失低 {diff:.4f}")
    else:
        print(f"→ LSTM 的损失低 {-diff:.4f}")
    
    print('\n[2] 训练速度')
    print("-" * 70)
    lstm_time = results['LSTM']['times'][0]
    gru_time = results['GRU']['times'][0]
    
    print(f"LSTM：{lstm_time:.2f}s")
    print(f"GRU：{gru_time:.2f}s")
    print(f"加速比：{lstm_time/gru_time:.2f}x")
    
    if gru_time < lstm_time:
        print('→ GRU 更快！')
    
    print('\n[3] 收敛速度')
    print("-" * 70)
    
    # 寻找损失低于阈值的迭代位置
    threshold = 1.5
    lstm_converge = next((i for i, loss in enumerate(lstm_losses) if loss < threshold), None)
    gru_converge = next((i for i, loss in enumerate(gru_losses) if loss < threshold), None)
    
    if lstm_converge and gru_converge:
        print(f"LSTM 达到收敛阈值的迭代次数：{lstm_converge}")
        print(f"GRU 达到收敛阈值的迭代次数：{gru_converge}")
        if gru_converge < lstm_converge:
            print('→ GRU 收敛更快！')
        else:
            print('→ LSTM 收敛更快！')
    
    print('\n[4] 总结')
    print("-" * 70)
    print('✓ GRU 的常见特点（需按具体任务核验）：')
    print('  - 参数较少（少 25-30%）')
    print('  - 训练较快（快 10-20%）')
    print('  - 最终表现接近')
    print('  - 架构更简单，更容易理解')

    print('\n✓ LSTM 的常见特点（需按具体任务核验）：')
    print('  - 独立细胞状态提供额外控制能力')
    print('  - 可能更适合很长的依赖')
    print('  - 控制更多：三个门，而非两个门')
    print('  - 应用经验充分')


def visualize_comparison(results, save_path='lstm_vs_gru.png'):
    """创建综合比较图。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    lstm_losses = results['LSTM']['losses']
    gru_losses = results['GRU']['losses']
    
    # 面板 1：损失曲线
    ax = axes[0, 0]
    window = 50
    lstm_smoothed = np.convolve(lstm_losses, np.ones(window)/window, mode='valid')
    gru_smoothed = np.convolve(gru_losses, np.ones(window)/window, mode='valid')
    
    ax.plot(lstm_smoothed, label='LSTM', color='blue', linewidth=2)
    ax.plot(gru_smoothed, label='GRU', color='red', linewidth=2)
    ax.set_xlabel('迭代', fontsize=12)
    ax.set_ylabel('损失（平滑后）', fontsize=12)
    ax.set_title('训练损失比较', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 面板 2：直接比较（散点图）
    ax = axes[0, 1]
    ax.scatter(lstm_losses[::10], gru_losses[::10], alpha=0.5, s=20)
    ax.plot([min(lstm_losses), max(lstm_losses)], 
           [min(lstm_losses), max(lstm_losses)],
           'k--', label='相同性能', linewidth=2)
    ax.set_xlabel('LSTM 损失', fontsize=12)
    ax.set_ylabel('GRU 损失', fontsize=12)
    ax.set_title('损失相关性', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 面板 3：参数量比较
    ax = axes[1, 0]
    # 占位值；实际应从模型实例统计
    categories = ['Update/Forget', 'Input/Reset', 'Cell/Hidden', '输出门']
    lstm_params = [25, 25, 25, 25]  # 简化值
    gru_params = [20, 20, 20, 20]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, lstm_params, width, label='LSTM', color='blue', alpha=0.7)
    ax.bar(x + width/2, gru_params, width, label='GRU', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('参数比例（%）', fontsize=12)
    ax.set_title('参数分布', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 面板 4：汇总表
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = [
        ['指标', 'LSTM', 'GRU', '较优模型'],
        ['', '', '', ''],
        ['Parameters', 'More', 'Less', 'GRU'],
        ['训练速度', 'Slower', 'Faster', 'GRU'],
        ['性能', 'Similar', 'Similar', 'Tie'],
        ['Complexity', 'Higher', 'Lower', 'GRU'],
        ['长序列', 'Better', 'Good', 'LSTM'],
        ['内存用量', 'Higher', 'Lower', 'GRU']
    ]
    
    table = ax.table(cellText=summary, cellLoc='left', loc='center',
                    colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
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
    print('LSTM 与 GRU：综合比较')
    print("="*70)
    
    print('\n本解答提供：')
    print('  1. 从零实现完整 GRU')
    print('  2. 并列训练比较')
    print('  3. 参数量分析')
    print('  4. 训练速度基准')
    print('  5. 性能比较')
    print('  6. 综合可视化')
    
    print("\n" + "-"*70)
    print('主要区别')
    print("-"*70)
    
    print('\nLSTM：')
    print('  • 三个门：遗忘门、输入门、输出门')
    print('  • 独立的细胞状态与隐藏状态')
    print('  • 参数较多')
    print('  • 显式控制记忆更新')
    
    print('\nGRU：')
    print('  • 两个门：更新门、重置门')
    print('  • 单个隐藏状态，没有独立细胞状态')
    print('  • 参数较少（少 25-30%）')
    print('  • 通过门控隐式管理记忆')
    
    print("\n" + "-"*70)
    print('如何选择模型')
    print("-"*70)
    
    print('\n✓ 可以考虑 LSTM 的情况：')
    print('  - 很长的序列（>100 步；此为原文经验建议）')
    print('  - 需要精细控制记忆')
    print('  - 训练数据较多')
    print('  - 计算成本不是主要限制')
    
    print('\n✓ 可以考虑 GRU 的情况：')
    print('  - 中等长度序列')
    print('  - 希望加快训练')
    print('  - 训练数据有限')
    print('  - 需要更简单的架构')
    print('  - 内存或算力受限')
    
    print("\n" + "="*70)
    print('实际应用中请比较两者，按结果选择。')
    print("="*70)
