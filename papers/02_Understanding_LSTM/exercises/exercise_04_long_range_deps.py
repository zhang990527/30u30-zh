"""
练习 4：长程依赖
====================================

目标：测试 LSTM 能记住多远之前的信息。

你的任务：
- 创建“记住第一个字符”的任务
- 测试不同距离：5、10、20、50、100 步
- 测量准确率与距离的关系
- 与基本 RNN 比较（可选）

学习目标：
1. 理解 LSTM 的记忆容量
2. 观察哪些情况下仍会出现梯度消失
3. 学习序列建模的实际限制
4. 设计测试记忆能力的任务

用时：1-2 小时
难度：中
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from implementation import LSTM


def generate_memory_task(distance, num_samples=1000, vocab_size=10):
    """
    生成“记住第一个字符”任务。
    
    格式：[A, 噪声, 噪声, ..., 噪声, ?] -> A
    
    参数：
        distance：起点与终点之间的噪声字符数
        num_samples：训练样本数
        vocab_size：词表大小
    
    返回：
        X：输入序列 (num_samples, distance+2)
        y：目标标签 (num_samples,)
    
    TODO 1：实现数据生成
    - 第一个字符：从 0 到 vocab_size-1 随机选择
    - 中间字符：随机噪声
    - 最后一个字符：特殊“查询”词元（vocab_size）
    - 目标：第一个字符
    """
    X = []
    y = []
    
    for _ in range(num_samples):
        # TODO：生成一个样本
        # first_char = np.random.randint(0, vocab_size)
        # noise = [np.random.randint(0, vocab_size) for _ in range(distance)]
        # query = vocab_size  # 特殊词元
        # sequence = [first_char] + noise + [query]
        pass
    
    return np.array(X), np.array(y)


def train_on_distance(distance, vocab_size=10, hidden_size=32, num_iterations=500):
    """
    在给定距离的记忆任务上训练 LSTM。

    参数：
        distance：待测试的距离
        vocab_size：词表大小
        hidden_size：隐藏层维度
        num_iterations：训练迭代次数

    返回：
        accuracy：最终测试准确率

    TODO 2：实现训练
    - 生成训练数据
    - 创建 LSTM
    - 训练 num_iterations 次
    - 在留出集上测试
    - 返回准确率
    """
    print(f"\n正在训练，距离为 {distance}...")
    
    # TODO 3：生成数据
    # X_train, y_train = generate_memory_task(distance, num_samples=1000)
    # X_test, y_test = generate_memory_task(distance, num_samples=200)
    
    # TODO 4：创建 LSTM
    # 注意：input_size = vocab_size + 1，包含查询词元
    # lstm = LSTM(...)
    
    # TODO 5：训练循环
    learning_rate = 0.01
    
    for iteration in range(num_iterations):
        # TODO：随机抽取一批数据
        # idx = np.random.randint(0, len(X_train))
        # inputs = X_train[idx]
        # target = y_train[idx]
        
        # TODO：前向传播、反向传播、更新
        pass
    
    # TODO 6：在测试集上评估
    correct = 0
    # for inputs, target in zip(X_test, y_test):
    #     prediction = predict(lstm, inputs)
    #     if prediction == target:
    #         correct += 1
    
    accuracy = 0.0  # 替换为：correct / len(X_test) * 100
    print(f"距离 {distance}: {accuracy:.1f}% 准确率")
    
    return accuracy


def test_all_distances(distances=[5, 10, 20, 50, 100]):
    """
    在多个距离上测试 LSTM。
    
    TODO 7：测试每个距离并返回结果
    """
    accuracies = []
    
    for distance in distances:
        # TODO 8：在当前距离上训练和评估
        acc = train_on_distance(distance)
        accuracies.append(acc)
    
    return distances, accuracies


def plot_results(distances, accuracies):
    """
    绘制准确率与距离的关系。
    
    TODO 9：创建可视化
    - x 轴为距离
    - y 轴为准确率
    - 为各点添加标记
    - 在随机猜测水平（1/vocab_size）处添加水平线
    """
    plt.figure(figsize=(10, 6))
    
    # TODO：在此编写绘图代码
    # plt.plot(distances, accuracies, marker='o')
    # plt.axhline(y=10, color='r', linestyle='--', label='随机猜测')
    # plt.xlabel('距离（步）')
    # plt.ylabel('准确率（%）')
    # plt.title('长程依赖测试')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    pass


def compare_with_rnn(distances):
    """
    可选：比较 LSTM 与基本 RNN。
    
    TODO 10（进阶）：实现基本 RNN 并比较
    - 原仓库预期：RNN 在超过约 10 步后表现较差，需验证
    - 原仓库预期：LSTM 可处理 50 步以上，需验证
    - 在同一张图上绘制两者
    """
    print('\n进阶：与基本 RNN 比较……')
    print('（尚未实现，请自行尝试）')


def analyze_findings(distances, accuracies):
    """
    分析结果并得出结论。
    
    TODO 11：回答以下问题：
    1. 性能从多大距离开始下降？
    2. LSTM 能处理的最大距离是多少？
    3. 与基本 RNN 相比如何（如果你实现了 RNN）？
    4. 哪些因素限制了最大距离？
    """
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    print('\n在此填写你的分析……')
    
    # TODO：打印统计量
    # - 表现最佳的距离
    # - 表现最差的距离
    # - 平均准确率
    # - 准确率低于阈值的位置


if __name__ == "__main__":
    print(__doc__)
    
    print('正在不同距离上测试 LSTM 记忆能力……')
    print("=" * 60)
    
    # TODO 12：运行实验
    distances = [5, 10, 20, 50, 100]
    
    # 测试所有距离
    # distances, accuracies = test_all_distances(distances)
    
    # 绘制结果
    # plot_results(distances, accuracies)
    
    # 分析
    # analyze_findings(distances, accuracies)
    
    # 可选：与 RNN 比较
    # compare_with_rnn(distances)
    
    print('\n练习 4 结束。')
    print('与 solutions/solution_04_long_range_deps.py 比较（原文路径；实际为 exercise_04_solution.py）')
