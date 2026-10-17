"""
第 4 天练习参考解答
=============================

含有解答！请先尝试自己实现。
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 将父目录加入路径，以导入 implementation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from implementation import MDLNetwork

def exercise_1_reparameterization():
    print('\n--- 练习 1：重参数化技巧 ---')
    
    def sample_gaussian(mu, rho, n_samples=10000):
        # 1. Softplus 变换
        sigma = np.log1p(np.exp(rho))
        
        # 2. 采样 epsilon
        epsilon = np.random.randn(n_samples)
        
        # 3. 缩放与平移
        w = mu + sigma * epsilon
        return w, sigma

    # 测试用例
    mu_target = 5.0
    rho_target = -2.0 # sigma 应约为 0.126
    
    samples, expected_sigma = sample_gaussian(mu_target, rho_target)
    
    print(f"目标均值：{mu_target}，实测均值：{np.mean(samples):.4f}")
    print(f"目标标准差：{expected_sigma:.4f}，实测标准差：{np.std(samples):.4f}")
    
    if np.abs(np.mean(samples) - mu_target) < 0.1:
        print('[OK] 均值检查通过')
    if np.abs(np.std(samples) - expected_sigma) < 0.1:
        print('[OK] Sigma 检查通过')

def exercise_2_gap_experiment():
    print('\n--- 练习 2：缺口实验 ---')
    
    # 1. 生成带缺口的三次函数数据
    X_left = np.linspace(-4, -2, 50).reshape(-1, 1)
    X_right = np.linspace(2, 4, 50).reshape(-1, 1)
    X_train = np.concatenate([X_left, X_right])
    
    # 归一化 y，使梯度保持稳定
    y_train = (X_train**3) / 10.0 + np.random.randn(100, 1) * 0.1
    
    # 2. 训练
    net = MDLNetwork(1, 20, 1)
    lr = 0.01
    kl_weight = 0.1
    
    print('在带缺口的三次函数上训练（需要一些时间）……')
    for i in range(2000):
        preds = net.forward(X_train)
        mse = np.mean((preds - y_train)**2)
        kl = net.total_kl() / 100.0
        
        loss = mse + kl_weight * kl
        
        net.backward(2*(preds - y_train)/100.0)
        net.update_weights(lr, kl_weight/100.0)
    
    # 3. 可视化
    X_test = np.linspace(-5, 5, 200).reshape(-1, 1)
    preds = np.array([net.forward(X_test) for _ in range(100)]).squeeze()
    
    mu = np.mean(preds, axis=0)
    sigma = np.std(preds, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, c='red', label='Data')
    plt.plot(X_test, mu, label='预测均值')
    plt.fill_between(X_test.flatten(), mu-2*sigma, mu+2*sigma, alpha=0.3, label='Uncertainty')
    plt.title('练习 2：带缺口的三次函数')
    plt.legend()
    plt.show()
    print('[OK] 检查图形：预期在 [-2, 2] 中不确定性增大')

def exercise_3_beta_sensitivity():
    print('\n--- 练习 3：Beta（KL 权重）敏感性 ---')
    print('这是一个思考实验代码段。')
    
    betas = [0.0, 0.1, 100.0]
    descriptions = [
        '过拟合（普通神经网络式行为）',
        '折中（较合理的不确定性）',
        '欠拟合（趋向先验，预测接近平线）'
    ]
    
    for b, desc in zip(betas, descriptions):
        print(f"Beta = {b}: {desc}")
        # 运行本练习时，可用不同 kl_weight 值循环执行练习 2

def exercise_4_monte_carlo():
    print('\n--- 练习 4：蒙特卡洛平滑 ---')
    
    # 设置测试网络
    net = MDLNetwork(1, 10, 1)
    x = np.array([[0.5]])
    
    print('预测 10 次……')
    preds = [net.forward(x).item() for _ in range(10)]
    print(f"各次预测：{[round(p, 3) for p in preds]}")
    print(f"方差：{np.var(preds):.5f}")
    print('[OK] 注意到单次预测会波动了吗？这就是取平均的原因。')

def exercise_5_pruning():
    print('\n--- 练习 5：免费午餐（剪枝）---')
    
    net = MDLNetwork(1, 20, 1)
    # 将部分权重设为高不确定性，模拟训练结果
    net.layer1.w_rho += 5.0 # 让这些权重具有很高不确定性
    
    # 计算 SNR
    mu = net.layer1.w_mu
    # sigma 约为 log(1 + exp(rho))
    sigma = np.log1p(np.exp(net.layer1.w_rho))
    
    snr = np.abs(mu) / sigma
    
    # 剪枝
    threshold = 0.1
    mask = snr < threshold
    n_pruned = np.sum(mask)
    total = mu.size
    
    print(f"权重总数：{total}")
    print(f"剪枝权重数（SNR < {threshold}): {n_pruned}")
    print(f"剪枝比例：{n_pruned/total*100:.1f}%")
    print('[OK] 原练习预期剪枝对输出均值影响不大，仍应实际验证。')

if __name__ == "__main__":
    # 取消注释以运行
    exercise_1_reparameterization()
    # exercise_2_gap_experiment() # 需要 matplotlib
    exercise_4_monte_carlo()
    exercise_5_pruning()