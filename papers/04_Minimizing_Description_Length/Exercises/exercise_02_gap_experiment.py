"""
练习 2：缺口实验（观察不确定性）
=========================================================

目标：在不完整数据上训练贝叶斯网络，并可视化不确定性。

任务：
- 创建带“缺口”（缺失区域）的数据集
- 在带缺口数据上训练贝叶斯网络
- 可视化网络在哪些区域不确定
- 与普通神经网络比较

学习目标：
1. 不确定性如何反映数据缺失导致的认知空白
2. 不确定性不等同于方差（理解其更深含义）
3. 偶然不确定性（aleatoric uncertainty）与认知不确定性（epistemic uncertainty）
4. 贝叶斯处理可能改善泛化的原因，并非无条件保证

用时：25-35 分钟
难度：困难
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 将父目录加入路径，以便从 implementation 导入
sys.path.insert(0, str(Path(__file__).parent.parent))
from implementation import BayesianLinearNetwork


def create_gappy_dataset(n_points=100, gap_regions=None):
    """
    创建带缺失区域的正弦波。
    
    参数：
        n_points：生成的数据点总数
        gap_regions：需要删除数据的 (start, end) 区间列表
        
    返回：
        X_train：排除缺口后的输入点
        y_train：排除缺口后的目标值
        X_dense：预测使用的完整输入范围
        y_true：完整范围内的真实正弦波
        
    为什么使用缺口？
    - 实际数据无法完美覆盖整个范围
    - 缺口要求预测未观测区域；原文称外推，但内部缺口通常属于插值区域
    - 预期在缺口处观察到不确定性增大！
    """
    
    # TODO 1：生成完整正弦波
    # X_dense = np.linspace(0, 4*np.pi, 500)
    # y_true = np.sin(X_dense)
    
    # TODO 2：创建不包含缺口的训练集
    # mask = np.ones(len(X_dense), dtype=bool)
    
    # TODO 3：遍历 gap_regions，并将对应掩码设为 False
    # for start, end in gap_regions:
    #     indices = (X_dense >= start) & (X_dense < end)
    #     mask[indices] = False
    
    # TODO 4：提取非缺口区域的训练数据
    # X_train = X_dense[mask]
    # y_train = y_true[mask]
    
    # TODO 5：返回全部组成部分
    # return X_train, y_train, X_dense, y_true, mask
    
    pass


def train_bayesian_model_on_gaps():
    """
    在带缺口数据上训练贝叶斯网络。
    """
    print('创建带缺口的数据集……')
    
    # TODO 6：定义缺口区域，例如避开 π/2 和 3π/2
    # gaps = [(np.pi/2 - 0.5, np.pi/2 + 0.5), 
    #         (3*np.pi/2 - 0.5, 3*np.pi/2 + 0.5)]
    # X_train, y_train, X_dense, y_true, mask = create_gappy_dataset(n_points=80, gap_regions=gaps)
    
    # print(f"  训练点数：{len(X_train)}")  # 实现 create_gappy_dataset 后即可运行
    
    print('初始化贝叶斯网络……')
    # TODO 7：创建具有 2 个隐藏层的 BayesianLinearNetwork（当前实现缺少该类）
    # model = BayesianLinearNetwork(
    #     input_dim=1,
    #     hidden_dims=[64, 32],
    #     output_dim=1,
    #     kl_weight=0.01
    # )
    
    print("Training...")
    # TODO 8：以学习率 0.01 训练 500 轮
    # losses = model.train(X_train[:, np.newaxis], y_train, 
    #                      epochs=500, lr=0.01, verbose=False)
    
    print('生成预测……')
    # TODO 9：多次运行网络，估计不确定性
    # n_mc = 100
    # predictions = np.zeros((len(X_dense), n_mc))
    # for i in range(n_mc):
    #     predictions[:, i] = model.forward(X_dense[:, np.newaxis]).flatten()
    
    # TODO 10：计算预测均值与标准差
    # mean_pred = np.mean(predictions, axis=1)
    # std_pred = np.std(predictions, axis=1)
    
    # 可视化
    print('绘制结果……')
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 图 1：数据与预测
    ax = axes[0]
    # TODO 10：完成上述 TODO 后，取消下列绘图语句的注释
    # if 'X_train' in locals():
    #     ax.scatter(X_train, y_train, color='red', s=30, label='训练数据', zorder=5)
    # if 'X_dense' in locals():
    #     ax.plot(X_dense, y_true, 'k--', linewidth=2, label='真实函数', zorder=2)
    #     ax.plot(X_dense, mean_pred, 'b-', linewidth=2, label='贝叶斯预测', zorder=3)
    #     ax.fill_between(X_dense, mean_pred - 2*std_pred, mean_pred + 2*std_pred, 
    #                      alpha=0.3, color='blue', label='2σ 不确定性')
    # TODO 11：用阴影标注缺口区域
    # for start, end in gaps:
    #     ax.axvspan(start, end, alpha=0.2, color='red', label='数据缺口' if start == gaps[0][0] else '')
    
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title('带缺口数据上的贝叶斯网络：缺失区域的不确定性')
    
    # 图 2：不确定性曲线
    ax = axes[1]
    # TODO 11：得到上面的 std_pred 后，取消注释：
    # if 'std_pred' in locals():
    #     ax.plot(X_dense, std_pred, 'purple', linewidth=2)
    #     ax.fill_between(X_dense, std_pred, alpha=0.3, color='purple')
    #     # TODO 12：此处也标注缺口区域
    #     for start, end in gaps:
    #         ax.axvspan(start, end, alpha=0.2, color='red')
    
    ax.set_xlabel('x')
    ax.set_ylabel('不确定性（σ）')
    ax.set_title('网络在哪些区域不确定？')
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / 'gap_experiment.png', dpi=150, bbox_inches='tight')
    print('  已保存至：gap_experiment.png')
    plt.close()
    
    print('\n[ok] 缺口实验完成。')
    print('\n关键观察：')
    print('  - 检查缺口区域的不确定性是否较高')
    print('  - 检查有数据处的不确定性是否较低')
    print('  - 这是认知不确定性：“这里没有数据”')


if __name__ == "__main__":
    train_bayesian_model_on_gaps()
