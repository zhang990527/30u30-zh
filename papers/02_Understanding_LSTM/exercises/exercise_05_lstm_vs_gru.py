"""
练习 5：LSTM 与 GRU 对比
===================================

目标：实现 GRU，并与 LSTM 比较。

你的任务：
- 从零实现门控循环单元（Gated Recurrent Unit，GRU）
- 在相同任务上训练 LSTM 与 GRU
- 比较速度、性能、参数量
- 撰写比较报告

学习目标：
1. 理解 GRU 架构（三组变换，对应 LSTM 的四组变换）
2. 观察复杂度与性能之间的权衡
3. 学习何时选择 GRU 或 LSTM
4. 理解设计简化

用时：2-3 小时
难度：高
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from implementation import LSTM


class GRU:
    """
    门控循环单元（GRU）的实现。

    GRU 将 LSTM 的四组变换简化为三组，包含两个门和一个候选状态：
    - 重置门：r_t = σ(W_r·[h_{t-1}, x_t] + b_r)
    - 更新门：z_t = σ(W_z·[h_{t-1}, x_t] + b_z)
    - 候选状态：h̃_t = tanh(W·[r_t⊙h_{t-1}, x_t] + b)
    - 隐藏状态：h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t

    与 LSTM 的主要区别：
    - 没有独立细胞状态，只有隐藏状态
    - 三组权重矩阵，而不是四组
    - 更新门同时控制遗忘与写入
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化 GRU 权重。
        
        TODO 1：初始化三组权重矩阵：重置门、更新门、候选状态
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # TODO 2：重置门权重
        # self.Wr = ...
        # self.br = ...
        
        # TODO 3：更新门权重
        # self.Wz = ...
        # self.bz = ...
        
        # TODO 4：候选状态权重
        # self.Wh = ...
        # self.bh = ...
        
        # TODO 5：输出权重
        # self.Wy = ...
        # self.by = ...
        
        pass
    
    def sigmoid(self, x):
        """sigmoid 激活函数。"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, inputs, targets, h_prev):
        """
        通过 GRU 前向传播。
        
        TODO 6：实现 GRU 前向传播
        
        对每个时间步：
        1. 计算重置门：r_t = σ(W_r·[h_{t-1}, x_t] + b_r)
        2. 计算更新门：z_t = σ(W_z·[h_{t-1}, x_t] + b_z)
        3. 计算候选状态：h̃_t = tanh(W·[r_t⊙h_{t-1}, x_t] + b)
        4. 更新隐藏状态：h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t
        5. 计算输出与损失
        
        注意：GRU 没有独立细胞状态，只有隐藏状态。
        """
        # TODO：实现前向传播
        pass
    
    def backward(self):
        """
        通过 GRU 执行随时间反向传播（BPTT）。
        
        TODO 7：实现 GRU 反向传播
        
        提示：
        - 梯度仅通过隐藏状态传递，没有独立细胞状态
        - 需要穿过重置门与更新门反向传播
        - 不要忘记梯度裁剪
        """
        # TODO：实现反向传播
        pass
    
    def update_weights(self, learning_rate):
        """
        使用计算出的梯度更新权重。
        
        TODO 8：更新所有权重矩阵
        """
        # TODO：实现权重更新
        pass


def count_parameters(model):
    """
    统计模型参数总数。
    
    TODO 9：统计各权重矩阵中的参数数目
    """
    total = 0
    
    if isinstance(model, LSTM):
        # TODO：统计 LSTM 参数（Wf、Wi、Wc、Wo、Wy 及偏置）
        pass
    elif isinstance(model, GRU):
        # TODO：统计 GRU 参数（Wr、Wz、Wh、Wy 及偏置）
        pass
    
    return total


def train_model(model, text, char_to_idx, num_iterations=1000):
    """
    训练模型并返回耗时与损失信息。
    
    TODO 10：实现训练循环
    - 记录损失
    - 记录每次迭代耗时
    - 返回两者
    """
    losses = []
    times = []
    
    # TODO：实现训练
    
    return losses, times


def compare_models(text, char_to_idx, hidden_size=64):
    """
    并列比较 LSTM 与 GRU。
    
    TODO 11：训练两种模型并收集指标
    """
    vocab_size = len(char_to_idx)
    
    print("\n" + "=" * 60)
    print('模型比较')
    print("=" * 60)
    
    # TODO 12：创建 LSTM
    print('\n1. 正在训练 LSTM……')
    # lstm = LSTM(vocab_size, hidden_size, vocab_size)
    # lstm_params = count_parameters(lstm)
    # lstm_losses, lstm_times = train_model(lstm, text, char_to_idx)
    
    # TODO 13：创建 GRU
    print('\n2. 正在训练 GRU……')
    # gru = GRU(vocab_size, hidden_size, vocab_size)
    # gru_params = count_parameters(gru)
    # gru_losses, gru_times = train_model(gru, text, char_to_idx)
    
    results = {
        'lstm': {
            'params': 0,  # lstm_params,
            'losses': [],  # lstm_losses,
            'times': []  # lstm_times
        },
        'gru': {
            'params': 0,  # gru_params,
            'losses': [],  # gru_losses,
            'times': []  # gru_times
        }
    }
    
    return results


def plot_comparison(results):
    """
    创建比较图。
    
    TODO 14：绘制：
    - 两种模型的训练损失
    - 训练速度：每次迭代耗时
    - 参数量柱状图
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # TODO：图 1——损失曲线
    
    # TODO：图 2——速度比较
    
    # TODO：图 3——参数量
    
    plt.tight_layout()
    plt.show()


def write_comparison_report(results):
    """
    撰写详细比较报告。
    
    TODO 15：回答以下问题：
    
    1. 参数量：
       - 每种模型有多少参数？
       - 百分比差异是多少？
    
    2. 训练速度：
       - 哪种模型训练更快？
       - 快多少，百分比是多少？
    
    3. 最终性能：
       - 哪种模型的最终损失更低？
       - 差异是否显著？
    
    4. 收敛：
       - 哪种模型收敛更快？
       - 达到阈值需要多少次迭代？
    
    5. 建议：
       - 何时使用 LSTM？
       - 何时使用 GRU？
       - 有哪些权衡？
    """
    print("\n" + "=" * 60)
    print('比较报告')
    print("=" * 60)
    
    print('\n在此填写你的报告……')


if __name__ == "__main__":
    print(__doc__)
    
    # 加载数据
    with open('../data/input.txt', 'r') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    
    print(f"正在训练，文本长度为 {len(text)} 个字符")
    print(f"词表大小：{len(chars)}")
    
    # TODO 16：运行比较
    # results = compare_models(text, char_to_idx)
    
    # TODO 17：绘制结果
    # plot_comparison(results)
    
    # TODO 18：撰写报告
    # write_comparison_report(results)
    
    print('\n练习 5 结束。')
    print('与 solutions/solution_05_lstm_vs_gru.py 比较（原文路径；实际为 exercise_05_solution.py）')
    
    print('\n全部练习结束。可以继续学习第 3 天。')
