"""
练习 4：蒙特卡洛预测（集成的作用）
======================================================

目标：多次运行同一网络，并汇总预测。

任务：
- 多次从后验采样权重
- 用不同权重样本进行前向传播
- 汇总预测，得到均值和不确定性
- 与单次前向传播的预测比较

学习目标：
1. 贝叶斯推断的蒙特卡洛近似
2. 为什么多次采样能改善不确定性估计
3. 大数定律：更多样本使估计更精确
4. 计算代价与精度的权衡

用时：20-30 分钟
难度：中等
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 将父目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from implementation import BayesianLinearNetwork


def generate_sine_data(n_train=150):
    """
    生成用于演示的正弦波数据。
    
    参数：
        n_train：训练点数量
        
    返回：
        X_train：输入特征（Nx1）
        y_train：目标值
        X_test：密集采样的测试输入
        y_test：无噪测试目标
    """
    # TODO 1：随机采样训练点
    # X_train = np.random.uniform(0, 4*np.pi, n_train)
    # y_train = np.sin(X_train) + np.random.randn(n_train) * 0.1
    
    # TODO 2：生成密集测试点
    # X_test = np.linspace(0, 4*np.pi, 500)
    # y_test = np.sin(X_test)
    
    # TODO 3：重塑为列向量
    # X_train = X_train[:, np.newaxis]
    # X_test = X_test[:, np.newaxis]
    
    # TODO 4：按 X 排序训练数据以便绘图
    # idx = np.argsort(X_train.flatten())
    # X_train = X_train[idx]
    # y_train = y_train[idx]
    
    # TODO 5：返回数据
    # return X_train, y_train, X_test, y_test
    
    pass


def monte_carlo_forward(model, X, n_samples=100):
    """
    用不同权重样本多次进行前向传播。
    
    参数：
        model：训练好的 BayesianLinearNetwork（该类在当前实现中不存在，原有接口问题保留）
        X：输入数据（Nx1）
        n_samples：蒙特卡洛样本数（前向传播次数）
        
    返回：
        predictions：形状为 (N, n_samples) 的预测数组
        mean：每个点的预测均值
        std：预测标准差
        
    原理：
    - 每次调用 forward()，网络都会重新采样权重
    - 不同权重样本产生不同预测
    - 预测集合近似由权重后验诱导的预测分布，而非权重后验本身
    - 均值作为点估计，标准差表示不确定性
    """
    
    # TODO 6：初始化用于存储预测的数组
    # predictions = np.zeros((len(X), n_samples))
    
    # TODO 7：循环 n_samples 次
    # for i in range(n_samples):
    
    # TODO 8：进行一次前向传播（每次采样新权重）
    # predictions[:, i] = model.forward(X).flatten()
    
    # TODO 9：沿样本维度而非特征维度计算预测均值
    # mean = np.mean(predictions, axis=1)
    
    # TODO 10：计算预测标准差
    # std = np.std(predictions, axis=1)
    
    # TODO 11：返回结果
    # return predictions, mean, std
    
    pass


def analyze_mc_convergence():
    """
    研究需要多少样本才能得到良好的不确定性估计。
    """
    print('训练初始模型……')
    
    # TODO 12：创建并训练贝叶斯网络
    # X_train, y_train, X_test, y_test = generate_sine_data(n_train=100)
    # model = BayesianLinearNetwork(
    #     input_dim=1,
    #     hidden_dims=[32, 16],
    #     output_dim=1,
    #     kl_weight=0.01
    # )
    # losses = model.train(X_train, y_train, epochs=400, lr=0.01, verbose=False)
    
    print('使用不同样本数运行蒙特卡洛预测……')
    
    # TODO 13：测试不同蒙特卡洛样本数
    # sample_sizes = [1, 5, 10, 20, 50, 100, 200]
    
    results = {
        'n_samples': [],
        'std_estimates': [],
        'std_stability': []  # 不同运行之间的标准差变化程度
    }
    
    # TODO 14：对每个样本数：
    # for n_mc in sample_sizes:
    #     print(f"  n_samples = {n_mc:3d}", end="", flush=True)
    
    # TODO 15：进行蒙特卡洛前向传播
    # _, mean, std = monte_carlo_forward(model, X_test, n_samples=n_mc)
    
    # TODO 16：多次运行以检查稳定性
    # 再运行 3 次，观察标准差变化程度
    # stds_multiple_runs = []
    # for _ in range(3):
    #     _, _, std_run = monte_carlo_forward(model, X_test, n_samples=n_mc)
    #     stds_multiple_runs.append(std_run)
    
    # TODO 17：计算标准差估计值的波动
    # std_variation = np.std(stds_multiple_runs, axis=0)
    # mean_variation = np.mean(std_variation)
    
    # TODO 18：存储结果
    # results['n_samples'].append(n_mc)
    # results['std_estimates'].append(np.mean(std))
    # results['std_stability'].append(mean_variation)
    
    # TODO 19：打印解释
    # print(f" → Uncertainty={np.mean(std):.4f}, Stability={mean_variation:.4f}")
    
    # 可视化
    print('\n\n绘制蒙特卡洛收敛图……')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 图 1：不确定性估计如何收敛
    ax = axes[0]
    if 'results' in locals() and results['n_samples']:
        ax.semilogx(results['n_samples'], results['std_estimates'], 'o-', 
                    linewidth=2, markersize=8)
        ax.set_xlabel('蒙特卡洛样本数')
        ax.set_ylabel('平均预测不确定性')
        ax.set_title('不确定性估计的收敛')
        ax.grid(True, alpha=0.3)
    
    # 图 2：估计的稳定性
    ax = axes[1]
    if 'results' in locals() and results['n_samples']:
        ax.loglog(results['n_samples'], results['std_stability'], 's-', 
                  linewidth=2, markersize=8, color='purple')
        ax.set_xlabel('蒙特卡洛样本数')
        ax.set_ylabel('不确定性估计的波动')
        ax.set_title('稳定性：越低越可靠')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / 'mc_convergence.png', dpi=150, bbox_inches='tight')
    print('  已保存至：mc_convergence.png')
    plt.close()
    
    print('\n[ok] 蒙特卡洛收敛分析完成。')
    print('\n关键认识：')
    print('  - 1 个样本：只有一次点预测，无法估计贝叶斯不确定性')
    print('  - 5-10 个样本：初步的不确定性估计')
    print('  - 50+ 个样本：通常收敛更好（大数定律）')
    print('  - 100+ 个样本：估计通常较稳定')
    print('\n经验建议：推断时先使用 50-100 个蒙特卡洛样本，再检查收敛')


if __name__ == "__main__":
    analyze_mc_convergence()
