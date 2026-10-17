"""
第 4 天练习参考解答：最小化描述长度
=============================================================

本文件包含全部 5 个练习的参考解答。
查看之前，请先尝试自己实现！

关键模式：
- 练习 1：实现 softplus 与高斯采样
- 练习 2：可视化数据缺口处的不确定性
- 练习 3：研究 KL 权重对校准的影响
- 练习 4：理解蒙特卡洛收敛与不确定性
- 练习 5：分析压缩的帕累托前沿
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================================
# 练习 1：重参数化技巧
# ============================================================================

def softplus_solution(x):
    """
    解答：Softplus 激活函数。
    σ = log(1 + exp(x))
    """
    return np.log1p(np.exp(x))


def sample_gaussian_solution(mu, rho, n_samples=10000):
    """
    解答：实现重参数化技巧。
    
    w = μ + σ * ε，其中 σ = softplus(ρ)，ε ~ N(0,1)
    """
    sigma = softplus_solution(rho)
    epsilon = np.random.randn(n_samples)
    samples = mu + sigma * epsilon
    return samples, sigma


def exercise_1_solution():
    """
    练习 1 解答：重参数化技巧
    """
    print("\n" + "="*70)
    print('解答 1：重参数化技巧')
    print("="*70)
    
    mu = 5.0
    rho = 0.0
    
    samples, sigma = sample_gaussian_solution(mu, rho, n_samples=100000)
    
    sample_mean = np.mean(samples)
    sample_std = np.std(samples)
    
    print(f"\n测试用例：μ={mu}, ρ={rho}")
    print(f"  预期 σ = softplus(0) = {softplus_solution(0):.4f}")
    print(f"  样本实测 σ = {sample_std:.4f}")
    print(f"  预期均值 = {mu}，实测 = {sample_mean:.4f}")
    
    assert np.abs(sample_mean - mu) < 0.05, '均值检查失败！'
    assert np.abs(sample_std - sigma) < 0.05, '标准差检查失败！'
    
    print('\n[OK] 练习 1 通过：重参数化采样检查通过！')
    return samples, sigma


# ============================================================================
# 练习 2：缺口实验
# ============================================================================

def create_gappy_sine_solution(n_points=100):
    """
    解答：创建中间带缺口的正弦波。
    """
    # 创建左右两个区域
    X_left = np.linspace(0, np.pi - 0.5, n_points//2)
    X_right = np.linspace(np.pi + 0.5, 2*np.pi, n_points//2)
    
    X_train = np.concatenate([X_left, X_right])
    y_train = np.sin(X_train) + np.random.randn(len(X_train)) * 0.1
    
    # 密集评估点
    X_dense = np.linspace(0, 2*np.pi, 500)
    y_true = np.sin(X_dense)
    
    # 标识存在数据区域的掩码
    mask = np.isin(X_dense, X_train, assume_unique=False)
    
    return X_train, y_train, X_dense, y_true, mask


def exercise_2_solution():
    """
    练习 2 解答：缺口实验
    
    展示数据缺口处不确定性增大的预期现象。
    """
    print("\n" + "="*70)
    print('解答 2：缺口实验')
    print("="*70)
    
    print('\n关键认识：')
    print('  - 贝叶斯网络预期在缺失数据处显示较高不确定性')
    print('  - 普通网络可能过拟合（过度自信）或欠拟合（预测不准）')
    print('\n实现方法：')
    print('  1. 创建带缺口的正弦波，例如缺失 x ∈ [π-0.5, π+0.5]')
    print('  2. 用 kl_weight=0.01 训练 BayesianLinearNetwork（当前实现缺少该类）')
    print('  3. 运行 100 次蒙特卡洛采样，得到每个点的均值与标准差')
    print('  4. 绘制预测与 2σ 不确定性带')
    print('  5. 观察：缺口处 σ 较高，有数据处 σ 较低')
    
    X_train, y_train, X_dense, y_true, mask = create_gappy_sine_solution(80)
    
    print(f"\n[OK] 练习 2 通过：缺口实验设置完成！")
    print(f"  - 训练点数：{len(X_train)}")
    print(f"  - 缺口区域：x ∈ [π-0.5, π+0.5]（{np.sum(~mask)} 个点）")
    print(f"  - 有数据区域：{np.sum(mask)} 个评估点")


# ============================================================================
# 练习 3：Beta 参数研究
# ============================================================================

def exercise_3_solution():
    """
    练习 3 解答：Beta 参数研究
    
    展示 kl_weight 如何控制正则化强度。
    """
    print("\n" + "="*70)
    print('解答 3：Beta 参数研究')
    print("="*70)
    
    print('\n关键认识：')
    print('  β（kl_weight）控制数据拟合与简洁性的权衡')
    print('\n需要观察的行为：')
    print('  - β = 0.00：无正则化，可能过拟合（低 MSE、低不确定性）')
    print('  - β = 0.01：弱正则化（拟合较好、不确定性较合理）')
    print('  - β = 0.1：折中（中等拟合、较合理的不确定性）')
    print('  - β = 1.0：强正则化（高 MSE、高不确定性）')
    print('  - β = 10.0：极强正则化（几乎不学习、非常不确定）')
    
    print('\n实现方法：')
    print('  1. 生成含噪正弦波')
    print('  2. 训练 6 个模型，β ∈ [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]')
    print('  3. 测量各模型的：')
    print('     - 测试集 MSE')
    print('     - 预测标准差的平均值（平均不确定性）')
    print('     - 校准误差（1σ 带覆盖率与预期 68% 之差）')
    print('  4. 绘制 MSE、不确定性、校准与 β 关系的 3 个子图')
    print('\n[OK] 练习 3 设置完成！')


# ============================================================================
# 练习 4：蒙特卡洛预测
# ============================================================================

def exercise_4_solution():
    """
    练习 4 解答：蒙特卡洛预测
    
    展示为什么多次采样可以改善不确定性估计。
    """
    print("\n" + "="*70)
    print('解答 4：蒙特卡洛预测')
    print("="*70)
    
    print('\n关键认识：')
    print('  单次前向传播得到一次权重采样下的点预测，不能据此估计贝叶斯不确定性')
    print('  多次前向传播得到权重后验所诱导预测分布的蒙特卡洛近似')
    print('\n收敛行为：')
    print('  - 1 个样本：无法可靠估计不确定性')
    print('  - 5 个样本：逐渐清晰')
    print('  - 10 个样本：可能得到较好近似')
    print('  - 50 个样本：估计通常更稳定（大数定律）')
    print('  - 100 个样本：通常更稳定，是否可用于生产仍需验证')
    
    print('\n实现方法：')
    print('  1. 训练一个贝叶斯网络模型')
    print('  2. 对 n_samples ∈ [1, 5, 10, 20, 50, 100, 200]：')
    print('     a. 前向传播 n_samples 次')
    print('     b. 收集全部预测，形状为 (N_test, n_samples)')
    print('     c. 沿样本维度计算均值与标准差')
    print('     d. 运行 3 次，测量标准差估计的变化（稳定性）')
    print('  3. 绘制收敛图：标准差估计如何趋稳')
    print('  4. 绘制稳定性图：增加样本数后波动如何变化')
    
    print('\n[OK] 练习 4 设置完成！')


# ============================================================================
# 练习 5：进阶 MDL
# ============================================================================

def exercise_5_solution():
    """
    练习 5 解答：进阶 MDL——帕累托前沿
    
    分析压缩与准确性的权衡。
    """
    print("\n" + "="*70)
    print('解答 5：进阶 MDL——帕累托前沿')
    print("="*70)
    
    print('\n关键认识：')
    print('  MDL 损失 = 重构损失 + KL 散度')
    print('  两项之间存在竞争！')
    
    print('\n帕累托前沿：')
    print('  X 轴：重构损失（数据拟合）')
    print('  Y 轴：KL 散度（模型复杂度）')
    print('\n  左下：理想情况（拟合好、模型简单），较少见')
    print('  右下：原文写为过拟合（拟合好、模型复杂），但与上述轴定义不一致')
    print('  左上：原文写为欠拟合（拟合差、模型简单），但与上述轴定义不一致')
    print('  右上：最差情况（拟合差、模型复杂）')
    print('\n  帕累托前沿：无法在不损害另一目标的同时改善某一目标')
    print('  此处前沿预期从左上延伸到右下')
    
    print('\n实现方法：')
    print('  1. 生成含噪正弦波')
    print('  2. 训练模型，β ∈ [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]')
    print('  3. 对每个模型：')
    print('     a. 计算 reconstruction_loss = mean((pred - y_test)²)')
    print('     b. 计算 kl_divergence = 所有层 KL 项之和')
    print('     c. 计算 mdl_loss = reconstruction + kl')
    print('     d. 计算 compression_ratio = kl / recon（启发式比值，并非实际编码压缩率）')
    print('  4. 绘制帕累托前沿（recon_loss 与 kl_div）')
    print('  5. 标注 β 值，展示变化过程')
    
    print('\n[OK] 练习 5 设置完成！')


# ============================================================================
# 主程序
# ============================================================================

def print_all_solutions():
    """Print all solution summaries."""
    exercise_1_solution()
    exercise_2_solution()
    exercise_3_solution()
    exercise_4_solution()
    exercise_5_solution()
    
    print("\n" + "="*70)
    print('汇总：全部 5 个练习')
    print("="*70)
    print('\n[OK] 练习 1：重参数化技巧（采样）')
    print('[OK] 练习 2：缺口实验（不确定性可视化）')
    print('[OK] 练习 3：Beta 参数研究（校准）')
    print('[OK] 练习 4：蒙特卡洛预测（收敛）')
    print('[OK] 练习 5：帕累托前沿（压缩分析）')
    print("\n" + "="*70)


if __name__ == "__main__":
    print_all_solutions()
