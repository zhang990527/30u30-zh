"""
练习 2：门激活分析
=====================================

目标：训练 LSTM，分析各门学到的行为。

你的任务：
- 在简单重复模式上训练 LSTM
- 记录并可视化门激活值
- 分析各门何时开启或关闭
- 撰写观察报告

学习目标：
1. 理解遗忘门何时激活，以及遗忘什么
2. 观察输入门何时激活，以及写入什么
3. 观察输出门何时激活，以及输出什么
4. 学习各门如何协作完成任务

用时：1-2 小时
难度：中
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from implementation import LSTM
from visualization import plot_gate_activations, analyze_gate_patterns


def create_repeating_pattern(pattern="abc", repeats=10):
    """
    创建用于分析的简单重复模式。
    
    参数：
        pattern：待重复的字符串模式，例如 "abc"
        repeats：重复次数
    
    返回：
        text：重复后的模式字符串
        char_to_idx：字符到索引的映射
        idx_to_char：索引到字符的映射
    """
    # TODO 1：通过重复模式创建文本
    # 提示：text = pattern * repeats
    text = None  # 替换为你的代码
    
    # TODO 2：创建词表映射
    chars = sorted(list(set(text)))
    char_to_idx = {}  # 在此填写
    idx_to_char = {}  # 在此填写
    
    return text, char_to_idx, idx_to_char


def train_and_capture_gates(text, char_to_idx, hidden_size=32, num_iterations=500):
    """
    训练 LSTM 并记录门激活值。

    参数：
        text：训练文本
        char_to_idx：字符到索引的映射
        hidden_size：隐藏层维度
        num_iterations：训练迭代次数

    返回：
        lstm：训练好的 LSTM 模型
        gates_dict：门激活值字典
        cell_states：细胞状态列表
    """
    vocab_size = len(char_to_idx)
    
    # TODO 3：创建 LSTM
    lstm = None  # 替换为 LSTM 初始化代码
    
    # TODO 4：训练循环
    # 初始化状态
    h_prev = np.zeros(hidden_size)
    C_prev = np.zeros(hidden_size)
    
    seq_length = 10
    learning_rate = 0.01
    
    print('正在训练 LSTM……')
    for iteration in range(num_iterations):
        # TODO 5：从文本中随机抽取一段序列
        # 提示：start = np.random.randint(0, len(text) - seq_length - 1)
        pass
        
        # TODO 6：前向传播
        
        # TODO 7：反向传播
        
        # TODO 8：更新权重
        
        # 打印进度
        if iteration % 100 == 0:
            print(f"迭代 {iteration}")
    
    # TODO 9：在测试序列上记录门激活值
    # 运行前向传播，手动记录每一步的 f_t、i_t、o_t
    test_seq = text[:20]
    gates_dict = {
        'forget': [],
        'input': [],
        'output': []
    }
    cell_states = []
    
    # TODO 10：运行前向传播并填写 gates_dict 与 cell_states
    # 手动计算各门值（参见 implementation.py）
    
    return lstm, gates_dict, cell_states, test_seq


def analyze_gates(gates_dict, cell_states, sequence):
    """
    分析并可视化门模式。
    
    参数：
        gates_dict：含 'forget'、'input'、'output' 键的字典
        cell_states：细胞状态数组列表
        sequence：与各时间步对应的文本序列
    """
    print("\n" + "=" * 60)
    print('门激活分析')
    print("=" * 60)
    
    # TODO 11：使用可视化函数
    # plot_gate_activations(gates_dict, sequence)
    
    # TODO 12：计算统计量
    # 对每个门计算：
    # - 平均激活值
    # - 标准差
    # - 完全开启的比例（>0.9）
    # - 完全关闭的比例（<0.1）
    
    print('\n在此填写你的分析……')


def write_report(gates_dict, sequence):
    """
    撰写观察报告。
    
    TODO 13：在报告中回答以下问题：
    
    1. 哪些门平均最活跃？
    2. 是否有门出现特化行为，例如始终开启或关闭？
    3. 各门如何响应特定字符？
    4. 是否观察到与重复结构相关的模式？
    5. 什么最令你意外？
    """
    print("\n" + "=" * 60)
    print("REPORT")
    print("=" * 60)
    print('\n在此填写你的观察……')


if __name__ == "__main__":
    print(__doc__)
    
    # 创建简单模式
    text, char_to_idx, idx_to_char = create_repeating_pattern("abc", repeats=20)
    print(f"模式：{text[:30]}...")
    
    # 训练并记录门值
    lstm, gates_dict, cell_states, sequence = train_and_capture_gates(
        text, char_to_idx
    )
    
    # 分析
    analyze_gates(gates_dict, cell_states, sequence)
    
    # 撰写报告
    write_report(gates_dict, sequence)
    
    print('\n练习 2 结束。')
    print('与 solutions/solution_02_gate_analysis.py 比较（原文路径；实际为 exercise_02_solution.py）')
