"""
练习 3：Beta 参数研究（不确定性校准）
=================================================================

目标：探索 KL 权重如何控制不确定性校准。

任务：
- 使用不同 kl_weight 值训练网络
- 测量不确定性如何变化
- 寻找校准的合适取值
- 绘制不确定性与准确性的权衡

学习目标：
1. kl_weight 的作用是什么？（控制先验约束强度）
2. 较小 kl_weight：模型较自信，但可能过拟合
3. 较大 kl_weight：模型较不确定，但可能欠拟合
4. 如何寻找最佳平衡

用时：30-40 分钟
难度：中等
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 将父目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from implementation import BayesianLinearNetwork


def create_sine_dataset(n_points=200, noise_std=0.1):
    """
    创建含噪正弦波数据集。
    
    参数：
        n_points：训练点数量
        noise_std：高斯噪声标准差
        
    返回：
        X_train：训练输入
        y_train：训练目标（含噪正弦波）
        X_test：测试输入
        y_test：测试目标（无噪正弦波）
    """
    # TODO 1：在 [0, 4π] 上均匀生成训练点
    # X_train = np.random.uniform(0, 4*np.pi, n_points)
    
    # TODO 2：生成含噪正弦波
    # y_train = np.sin(X_train) + np.random.randn(n_points) * noise_std
    
    # TODO 3：生成更密集的无噪测试点
    # X_test = np.linspace(0, 4*np.pi, 500)
    # y_test = np.sin(X_test)
    
    # TODO 4：按 X 排序训练数据以便绘图
    # idx = np.argsort(X_train)
    # X_train = X_train[idx]
    # y_train = y_train[idx]
    
    # TODO 5：以列向量形式返回（重塑为 Nx1）
    # return X_train[:, np.newaxis], y_train, X_test[:, np.newaxis], y_test
    
    pass


def evaluate_model(model, X_test, y_test, n_mc=50):
    """
    评估训练好的模型并计算指标。
    
    参数：
        model：训练好的 BayesianLinearNetwork（该类在当前实现中不存在，原有接口问题保留）
        X_test：测试输入
        y_test：无噪测试目标
        n_mc：用于不确定性估计的蒙特卡洛样本数
        
    返回：
        mse：均方误差
        mean_uncertainty：平均预测不确定性
        calibration_error：不确定性校准程度
    """
    # TODO 6：多次运行网络
    # predictions = np.zeros((len(X_test), n_mc))
    # for i in range(n_mc):
    #     predictions[:, i] = model.forward(X_test).flatten()
    
    # TODO 7：计算均值与标准差
    # mean_pred = np.mean(predictions, axis=1)
    # std_pred = np.std(predictions, axis=1)
    
    # TODO 8：计算 MSE
    # mse = np.mean((mean_pred - y_test)**2)
    
    # TODO 9：计算平均不确定性（标准差的平均值）
    # mean_uncertainty = np.mean(std_pred)
    
    # TODO 10：计算校准误差
    # 计算落入 1σ 带的点的比例（高斯且校准良好时约为 68%）
    # in_band = np.abs(mean_pred - y_test) <= std_pred
    # coverage_1sigma = np.mean(in_band)
    # calibration_error = np.abs(coverage_1sigma - 0.68)
    
    # TODO 11：返回指标
    # return mse, mean_uncertainty, calibration_error
    
    pass


def study_kl_weights():
    """
    使用不同 kl_weight 值训练模型并比较。
    """
    print('准备数据集……')
    X_train, y_train, X_test, y_test = create_sine_dataset(n_points=150, noise_std=0.15)
    
    # TODO 12：定义要测试的 kl_weights
    kl_weights = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
    # 覆盖从“无正则化”到“极强正则化”的范围
    
    results = {
        'kl_weights': [],
        'mse': [],
        'uncertainty': [],
        'calibration_error': []
    }
    
    print('\n用不同 KL 权重训练模型……\n')
    
    # TODO 13：对每个 kl_weight：
    for kl_w in kl_weights:
        print(f"  kl_weight = {kl_w:8.4f}", end="", flush=True)
        
        # TODO 14：创建并训练模型
        # model = BayesianLinearNetwork(
        #     input_dim=1,
        #     hidden_dims=[32, 16],
        #     output_dim=1,
        #     kl_weight=kl_w
        # )
        # losses = model.train(X_train, y_train, epochs=300, lr=0.01, verbose=False)
        
        # TODO 15：在测试集上评估
        # mse, unc, calib = evaluate_model(model, X_test, y_test)
        
        # TODO 16：存储结果
        # results['kl_weights'].append(kl_w)
        # results['mse'].append(mse)
        # results['uncertainty'].append(unc)
        # results['calibration_error'].append(calib)
        
        # TODO 17：打印结果，并用表情符号辅助解释
        # print(f" → MSE={mse:.4f}, Uncertainty={unc:.4f}")
        
        pass
    
    # 可视化
    print('\n\n绘制结果……')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 图 1：MSE 与 kl_weight 的关系
    ax = axes[0]
    if 'results' in locals() and results['kl_weights']:
        ax.semilogx(results['kl_weights'], results['mse'], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('KL 权重（β）')
        ax.set_ylabel('测试 MSE')
        ax.set_title('拟合质量：较高 MSE 可能表示欠拟合')
        ax.grid(True, alpha=0.3)
    
    # 图 2：不确定性与 kl_weight 的关系
    ax = axes[1]
    if 'results' in locals() and results['kl_weights']:
        ax.semilogx(results['kl_weights'], results['uncertainty'], 's-', 
                    linewidth=2, markersize=8, color='purple')
        ax.set_xlabel('KL 权重（β）')
        ax.set_ylabel('平均不确定性（σ）')
        ax.set_title('不确定性：越大表示网络越不确定')
        ax.grid(True, alpha=0.3)
    
    # 图 3：校准与 kl_weight 的关系
    ax = axes[2]
    if 'results' in locals() and results['kl_weights']:
        ax.semilogx(results['kl_weights'], results['calibration_error'], '^-', 
                    linewidth=2, markersize=8, color='green')
        ax.axhline(0, color='red', linestyle='--', label='理想校准')
        ax.set_xlabel('KL 权重（β）')
        ax.set_ylabel('校准误差')
        ax.set_title('校准：越低表示不确定性估计越接近目标覆盖率')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / 'beta_study.png', dpi=150, bbox_inches='tight')
    print('  已保存至：beta_study.png')
    plt.close()
    
    print('\n[ok] Beta 研究完成。')
    print('\n关键认识：')
    print('  - 低 β：可能过拟合（训练拟合好，但不确定性可能不佳）')
    print('  - 高 β：可能欠拟合（MSE 较高；校准改善需实际验证）')
    print('  - 平衡点：兼顾 MSE 与校准误差的位置')


if __name__ == "__main__":
    study_kl_weights()
