"""
练习 1：重参数化技巧（数学 → 代码）
=========================================================

目标：实现贝叶斯神经网络的核心机制。

任务：
- 实现高斯采样方法
- 验证样本与预期统计量一致
- 理解 Softplus 技巧

学习目标：
1. 如何让随机采样可微
2. 重参数化技巧：w = μ + σ * ε
3. 为什么对 σ 使用 Softplus 而非 Sigmoid
4. 蒙特卡洛近似

用时：20-30 分钟
难度：中等
"""

import numpy as np
import matplotlib.pyplot as plt


def softplus(x):
    """
    Softplus 激活函数：log(1 + exp(x))。
    将 (-∞, ∞) 映射到 (0, ∞)。
    用于保证 σ > 0。
    """
    # TODO 1：实现 softplus
    # 提示：使用 np.log1p(np.exp(x))；原文称其数值稳定，但大正数下 exp 仍可能溢出
    pass


def sample_gaussian(mu, rho, n_samples=10000):
    """
    通过重参数化技巧采样权重。
    
    参数：
        mu：分布均值（“学习到的值”）
        rho：无约束的不确定性参数
        n_samples：要抽取的样本数
        
    返回：
        samples：权重样本数组
        sigma：计算得到的标准差
        
    公式：
        σ = Softplus(ρ)
        ε ~ N(0, 1)
        w = μ + σ * ε
        
    为什么使用此技巧？
    - 普通反向传播不能直接对随机抽样操作求导
    - 重参数化将参数无关的随机性移出待求导路径
    - 这样就能对 μ 和 ρ 反向传播！
    """
    # TODO 2：使用 softplus 计算 sigma
    # sigma = softplus(rho)
    
    # TODO 3：从 N(0, 1) 采样 epsilon
    # epsilon = np.random.randn(n_samples)
    
    # TODO 4：计算 w = mu + sigma * epsilon
    # samples = mu + sigma * epsilon
    
    # TODO 5：返回 samples 与 sigma
    # return samples, sigma
    
    pass


def test_gaussian_sampling():
    """
    测试采样是否产生正确统计量。
    """
    print('测试高斯采样……\n')
    
    # 测试 1：基本采样
    mu = 5.0
    rho = 0.0  # 应得到 sigma ≈ log(2) ≈ 0.693
    samples, sigma = sample_gaussian(mu, rho, n_samples=100000)
    
    # TODO 6：验证均值接近 mu
    # sample_mean = np.mean(samples)
    # assert abs(sample_mean - mu) < 0.05, f"均值不匹配：{sample_mean} 与 {mu}"
    
    # TODO 7：验证标准差接近 sigma
    # sample_std = np.std(samples)
    # assert abs(sample_std - sigma) < 0.05, f"标准差不匹配：{sample_std} 与 {sigma}"
    
    print('[ok] 基本采样测试通过')
    
    # 测试 2：不同 rho 值
    print('\n测试不同 rho 值：')
    for rho_val in [-2, 0, 2, 4]:
        # TODO 8：采样并计算预期 sigma
        # samples, sigma = sample_gaussian(5.0, rho_val, 50000)
        
        # TODO 9：绘制直方图
        # print(f"  ρ={rho_val:2d} → σ={sigma:.3f}")
        
        pass
    
    # 测试 3：可视化
    # TODO 10：创建图形展示不同分布
    print('\n[ok] 全部测试通过。')


if __name__ == "__main__":
    test_gaussian_sampling()
