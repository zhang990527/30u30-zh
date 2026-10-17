"""
练习 3：消融研究
===========================

目标：每次移除一个门，并测量影响。

你的任务：
- 实现 4 种 LSTM 变体：无遗忘门、无输入门、无输出门及基线
- 在相同任务上训练 4 种模型
- 比较性能指标
- 解释哪些门最关键

学习目标：
1. 理解哪些门不可或缺
2. 观察各门如何互补
3. 学习复杂度与性能的权衡
4. 建立对 LSTM 设计选择的直观认识

用时：1-2 小时
难度：中
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from implementation import LSTM


class LSTMNoForget(LSTM):
    """将遗忘门固定为 1 的 LSTM（始终保留旧状态）。"""
    
    def forward(self, inputs, targets, h_prev, C_prev):
        """
        将前向传播修改为 f_t = 1。
        
        TODO 1：复制 LSTM 的 forward 方法，但令 f_t = np.ones(...)
        """
        # 在此填写代码
        pass


class LSTMNoInput(LSTM):
    """将输入门固定为 1 的 LSTM（始终写入候选值）。"""
    
    def forward(self, inputs, targets, h_prev, C_prev):
        """
        将前向传播修改为 i_t = 1。
        
        TODO 2：复制 LSTM 的 forward 方法，但令 i_t = np.ones(...)
        """
        # 在此填写代码
        pass


class LSTMNoOutput(LSTM):
    """将输出门固定为 1 的 LSTM（始终输出全部变换后的细胞状态）。"""
    
    def forward(self, inputs, targets, h_prev, C_prev):
        """
        将前向传播修改为 o_t = 1。
        
        TODO 3：复制 LSTM 的 forward 方法，但令 o_t = np.ones(...)
        """
        # 在此填写代码
        pass


def train_model(model, text, char_to_idx, num_iterations=1000):
    """
    训练模型，返回损失历史。
    
    参数：
        model：待训练的 LSTM 模型
        text：训练文本
        char_to_idx：字符到索引的映射
        num_iterations：训练迭代次数
    
    返回：
        losses：损失值列表
    """
    hidden_size = model.hidden_size
    seq_length = 25
    learning_rate = 0.001
    
    h_prev = np.zeros(hidden_size)
    C_prev = np.zeros(hidden_size)
    
    losses = []
    
    print("Training...")
    for iteration in range(num_iterations):
        # TODO 4：实现训练循环
        # - 随机抽取序列
        # - 前向传播
        # - 反向传播
        # - 更新权重
        # - 存储损失
        
        pass
    
    return losses


def compare_models(text, char_to_idx, hidden_size=64):
    """
    比较 4 种模型变体。
    
    TODO 5：创建并训练这 4 种模型：
    - 基线 LSTM
    - 无遗忘门
    - 无输入门
    - 无输出门
    
    返回：
        results：模型名称到损失历史的映射字典
    """
    vocab_size = len(char_to_idx)
    results = {}
    
    # TODO 6：创建基线 LSTM
    print('\n1. 正在训练基线 LSTM……')
    # baseline = LSTM(...)
    # results['Baseline'] = train_model(...)
    
    # TODO 7：创建并训练无遗忘门 LSTM
    print('\n2. 正在训练无遗忘门的 LSTM……')
    # no_forget = LSTMNoForget(...)
    # results['No Forget'] = train_model(...)
    
    # TODO 8：创建并训练无输入门 LSTM
    print('\n3. 正在训练无输入门的 LSTM……')
    # no_input = LSTMNoInput(...)
    # results['No Input'] = train_model(...)
    
    # TODO 9：创建并训练无输出门 LSTM
    print('\n4. 正在训练无输出门的 LSTM……')
    # no_output = LSTMNoOutput(...)
    # results['No Output'] = train_model(...)
    
    return results


def plot_comparison(results):
    """
    绘制所有模型的损失曲线。
    
    TODO 10：创建比较 4 种模型的图
    - 各模型使用不同颜色
    - 添加图例
    - 标注坐标轴
    - 添加标题
    """
    plt.figure(figsize=(12, 6))
    
    # 在此编写绘图代码
    
    plt.show()


def analyze_results(results):
    """
    分析并比较最终性能。
    
    TODO 11：计算并打印：
    - 各模型的最终损失
    - 收敛速度：达到阈值所需的迭代次数
    - 从优到劣的排名
    """
    print("\n" + "=" * 60)
    print('结果分析')
    print("=" * 60)
    
    # 在此进行分析


def write_conclusions():
    """
    撰写哪些门最重要的结论。
    
    TODO 12：回答以下问题：
    1. 移除哪个门对性能影响最大？
    2. 为什么你认为该门最重要？
    3. 移除任一单个门后，LSTM 还能工作吗？
    4. 这对理解 LSTM 的设计有何启发？
    """
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    
    print('\n在此填写你的结论……')


if __name__ == "__main__":
    print(__doc__)
    
    # 创建训练数据
    with open('../data/input.txt', 'r') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    
    print(f"正在训练，文本长度为 {len(text)} 个字符")
    print(f"词表大小：{len(chars)}")
    
    # 比较模型
    results = compare_models(text, char_to_idx)
    
    # 绘图
    plot_comparison(results)
    
    # 分析
    analyze_results(results)
    
    # 结论
    write_conclusions()
    
    print('\n练习 3 结束。')
    print('与 solutions/solution_03_ablation_study.py 比较（原文路径；实际为 exercise_03_solution.py）')
