"""
最简 MDL／贝叶斯网络训练脚本
==============================================

训练含噪权重神经网络，求解回归任务。
使用带缺口的正弦波数据集，展示 MDL 的作用。

挑战：
-------------
向模型提供 x=[-3, -1] 和 x=[1, 3] 区间的数据。
删除中间区间（x=[-1, 1]）的数据。

- 普通神经网络可能自信地用直线穿过缺口。
- MDL 网络的预期现象是在缺口处表现出更高不确定性；这不是普遍保证。

用法：
    # 使用默认设置训练
    python train_minimal.py
    
    # 使用较强的“简洁性压力”（较大的 KL 权重）训练
    python train_minimal.py --kl-weight 0.5
    
    # 训练大网络，观察是否过拟合
    python train_minimal.py --hidden-size 100 --epochs 5000
"""

import numpy as np
import argparse
import os
import sys

# 确保能导入本项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from implementation import MDLNetwork
from visualization import create_comprehensive_report

def generate_gappy_data(n_samples=100, noise_std=0.1):
    """
    生成中间缺失一段数据的正弦波。
    
    参数：
        n_samples：数据点数量
        noise_std：加入 y 的随机噪声标准差
        
    返回：
        X：输入，形状为 (n, 1)
        y：目标，形状为 (n, 1)
    """
    # 在 [-3, 3] 中生成随机点
    X = np.random.uniform(-3, 3, size=(n_samples, 1))
    
    # 构造缺口：删除 |x| < 1 的点
    # 让模型预测中间未观测区域（原文称为外推，几何上位于观测区间内部）
    mask = np.abs(X) > 1.0
    X = X[mask].reshape(-1, 1)
    
    # 计算 y = sin(x) + 噪声
    y = np.sin(X) + np.random.normal(0, noise_std, size=X.shape)
    
    print(f"数据生成：")
    print(f"  - 范围：[-3, 3]，缺口为 [-1, 1]")
    print(f"  - 样本数：{len(X)}")
    print(f"  - 噪声水平：{noise_std}")
    
    return X, y

def train(args):
    """
    训练主循环。
    """
    # 1. 准备数据
    print(f"\n{'='*60}")
    print('1. 生成合成数据')
    print(f"{'='*60}")
    X_train, y_train = generate_gappy_data(n_samples=args.samples)
    
    # 2. 初始化网络
    print(f"\n{'='*60}")
    print('2. 初始化 MDL 网络')
    print(f"{'='*60}")
    
    net = MDLNetwork(
        input_size=1,
        hidden_size=args.hidden_size,
        output_size=1
    )
    
    print(f"  - 结构：[1] -> 贝叶斯层[{args.hidden_size}] -> ReLU -> 贝叶斯层[1]")
    print(f"  - 复杂度代价权重（Beta）：{args.kl_weight}")
    print(f"  - 学习率：{args.lr}")
    
    # 3. 训练循环
    print(f"\n{'='*60}")
    print('3. 开始优化')
    print(f"{'='*60}")
    
    history = {'total': [], 'nll': [], 'kl': []}
    
    # 预先计算批次大小归一化因子
    # 让 KL 与按批次平均的 NLL 处于可比较尺度
    # 因此将 KL 除以训练批次数（或总样本数）；这里实际使用总样本数
    # 提示：数据损失与 KL 的缩放必须一致；原文将其归于论文，尚待核验
    complexity_scale = 1.0 / len(X_train)
    
    for epoch in range(args.epochs):
        # --- 前向传播 ---
        # 注意：每次调用 forward() 都会采样新噪声！
        preds = net.forward(X_train)
        
        # --- 计算损失 ---
        # 1. 误差代价（负对数似然，NLL）
        # 假设高斯噪声时，NLL 与 MSE 成比例
        mse = np.mean((preds - y_train) ** 2)
        nll = mse 
        
        # 2. 复杂度代价（KL 散度）
        # 权重与先验 N(0,1) 相差多远？
        raw_kl = net.total_kl()
        kl_term = raw_kl * complexity_scale
        
        # 3. 总目标（自由能）
        # 最小化：误差 + beta * 复杂度
        loss = nll + args.kl_weight * kl_term
        
        # --- 反向传播 ---
        # MSE 对预测值的梯度
        d_nll = 2 * (preds - y_train) / len(X_train)
        
        # 反向传播
        net.backward(d_nll)
        
        # --- 更新权重 ---
        net.update_weights(args.lr, args.kl_weight * complexity_scale)
        
        # --- 记录日志 ---
        history['total'].append(loss)
        history['nll'].append(nll)
        history['kl'].append(kl_term)
        
        if epoch % (args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(f"轮次 {epoch:4d}/{args.epochs} | "
                  f"总损失：{loss:.4f} | "
                  f"误差（NLL）：{nll:.4f} | "
                  f"复杂度（KL）：{kl_term:.4f}")

    print(f"\n训练完成。")
    print(f"最终误差：{history['nll'][-1]:.4f}")
    
    # 4. 保存模型
    net.save(args.checkpoint)
    
    # 5. 生成分析报告
    print(f"\n{'='*60}")
    print('4. 生成可视化报告')
    print(f"{'='*60}")
    
    # 创建密集测试集，使绘图平滑
    X_test = np.linspace(-4, 4, 200).reshape(-1, 1)
    
    create_comprehensive_report(
        model=net,
        history=history,
        X=X_train,
        y=y_train,
        save_dir=args.output_dir
    )
    
    print(f"\n完成！请查看 '{args.output_dir}' 目录中的图形。")

def main():
    parser = argparse.ArgumentParser(description='训练 MDL／贝叶斯神经网络')
    
    # 实验设置
    parser.add_argument('--samples', type=int, default=150,
                       help='训练点数量（默认：150）')
    parser.add_argument('--hidden-size', type=int, default=20,
                       help='隐藏层大小（默认：20）')
    parser.add_argument('--epochs', type=int, default=3000,
                       help='训练轮数（默认：3000）')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='学习率（默认：0.01）')
    
    # 最重要的参数：Beta（KL 权重）
    parser.add_argument('--kl-weight', type=float, default=0.1,
                       help='简洁性压力（Beta）：越大则权重描述更简单、分布更分散（默认：0.1）')
    
    # 输入／输出
    parser.add_argument('--checkpoint', type=str, default='mdl_model.pkl',
                       help='模型保存路径（默认：mdl_model.pkl）')
    parser.add_argument('--output-dir', type=str, default='mdl_analysis',
                       help='分析图保存目录（默认：mdl_analysis）')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print('MDL“含噪权重”实验')
    print("=" * 60)
    print('目标：拟合中间缺失一段数据的正弦波。')
    
    train(args)

if __name__ == "__main__":
    main()