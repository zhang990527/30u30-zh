"""
MDL／贝叶斯网络可视化工具
===============================================

用于分析贝叶斯神经网络的绘图函数。

包括：
1. 不确定性包络：整个输入空间上的预测不确定性
2. 权重分布：学习得到的 mu 与 sigma 直方图
3. 信噪比分析：哪些权重承载信息
4. 先验到后验的偏移：权重偏离先验的程度
5. 损失动态：训练中误差与复杂度的权衡
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import os

# 设置绘图样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def plot_uncertainty_envelope(model, X_train, y_train, X_test=None, n_samples=100, save_path=None):
    """
    可视化贝叶斯网络的不确定性包络。
    
    此图展示 MDL 的预期作用：
    - 数据附近：模型较确定（方差低，带状区域窄）。
    - 远离数据：模型较不确定（方差高，带状区域宽）；这取决于模型和训练。
    
    参数：
        model：训练好的 MDLNetwork 实例
        X_train, y_train：训练数据点
        X_test：测试范围，默认自动确定
        n_samples：蒙特卡洛前向传播次数
    """
    print(f"  [可视化] 生成不确定性包络（{n_samples} 个样本）……")
    
    if X_test is None:
        # 自动范围：比训练数据范围向外扩大 50%
        span = X_train.max() - X_train.min()
        X_test = np.linspace(X_train.min() - span*0.5, 
                             X_train.max() + span*0.5, 
                             300).reshape(-1, 1)
    
    # 蒙特卡洛采样
    preds = []
    for _ in range(n_samples):
        # 每次前向传播都使用不同随机噪声！
        preds.append(model.forward(X_test))
    
    preds = np.array(preds).squeeze() # (n_samples, n_points)
    
    # 统计量
    mu = np.mean(preds, axis=0)
    sigma = np.std(preds, axis=0)
    
    # 设置图形
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 1. 多条采样预测曲线（集成）
    # 使用浅色线展示各个“假设”
    for i in range(min(20, n_samples)):
        ax.plot(X_test, preds[i], color='teal', alpha=0.15, linewidth=1)
        
    # 2. 预测均值
    ax.plot(X_test, mu, color='navy', linewidth=3, label='预测均值')
    
    # 3. 不确定性带（原文称置信区间；此处是预测均值周围的标准差带）
    # 1 个标准差（高斯情形约 68%）
    ax.fill_between(X_test.flatten(), mu - sigma, mu + sigma, 
                    color='teal', alpha=0.3, label='±1σ 不确定性带（高斯情形约 68%）')
    # 2 个标准差（高斯情形约 95%）
    ax.fill_between(X_test.flatten(), mu - 2*sigma, mu + 2*sigma, 
                    color='teal', alpha=0.1, label='±2σ 不确定性带（高斯情形约 95%）')
    
    # 4. 训练数据
    ax.scatter(X_train, y_train, color='crimson', s=50, edgecolors='white', 
               linewidth=1.5, zorder=10, label='训练数据')
    
    # 设置格式
    ax.set_title('贝叶斯不确定性包络', fontsize=18, fontweight='bold')
    ax.set_xlabel('输入空间（$x$）', fontsize=14)
    ax.set_ylabel('模型输出（$y$）', fontsize=14)
    ax.legend(frameon=True, fontsize=12, loc='upper left')
    
    # 添加教学说明
    low_uncertainty_idx = np.argmin(sigma)
    high_uncertainty_idx = np.argmax(sigma)
    
    ax.annotate('低不确定性\n（模型“知道”）',
                xy=(X_test[low_uncertainty_idx], mu[low_uncertainty_idx]), 
                xytext=(X_test[low_uncertainty_idx], mu[low_uncertainty_idx]-1.5),
                arrowprops=dict(facecolor='black', shrink=0.05))
                
    ax.annotate('高不确定性\n（模型“猜测”）',
                xy=(X_test[high_uncertainty_idx], mu[high_uncertainty_idx] + 2*sigma[high_uncertainty_idx]), 
                xytext=(X_test[high_uncertainty_idx]-1, mu[high_uncertainty_idx] + 3),
                arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"  [已保存] {save_path}")
    else:
        plt.show()

def plot_weight_distributions(model, layer_name='layer1', save_path=None):
    """
    可视化权重分布的构成（Mu 与 Sigma）。
    
    这用来直观解释“描述长度”。
    - 尖峰（较小 sigma）：信息代价高。
    - 平缓分布（较大 sigma）：精度代价低，但过大的 sigma 仍受 KL 惩罚。
    """
    print(f"  [可视化] 分析以下层的权重分布：{layer_name}...")
    layer = getattr(model, layer_name)
    
    # 提取参数
    mu = layer.w_mu.flatten()
    # 由 rho 计算 sigma（Softplus）
    sigma = np.log1p(np.exp(-np.abs(layer.w_rho))) + np.maximum(layer.w_rho, 0)
    sigma = sigma.flatten()
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 图 1：均值直方图（取值是什么？）
    sns.histplot(mu, kde=True, ax=axes[0], color='indigo', bins=30)
    axes[0].set_title(f'权重均值（$\mu$）\n“点估计”', fontweight='bold')
    axes[0].set_xlabel('权重值')
    axes[0].axvline(0, color='black', linestyle='--', alpha=0.5)
    
    # 图 2：不确定性直方图（有多确定？）
    sns.histplot(sigma, kde=True, ax=axes[1], color='darkorange', bins=30)
    axes[1].set_title(f'权重不确定性（$\sigma$）\n“分散程度”', fontweight='bold')
    axes[1].set_xlabel('标准差')
    
    # 图 3：火山图（显著性）
    # 大权重通常需要较低不确定性才有用。
    # 小权重可有较高不确定性，通常影响较小。
    sns.scatterplot(x=mu, y=sigma, ax=axes[2], alpha=0.6, color='seagreen')
    axes[2].set_title('权重分布图（$\\mu$ 与 $\\sigma$）', fontweight='bold')
    axes[2].set_xlabel('权重均值（$\\mu$）')
    axes[2].set_ylabel('不确定性（$\\sigma$）')
    axes[2].axvline(0, color='black', linestyle='--', alpha=0.3)
    
    # 添加“剪枝区域”标注
    axes[2].add_patch(plt.Rectangle((-0.5, 0.2), 1.0, 1.0, 
                                    color='red', alpha=0.1, label='候选剪枝区域'))
    axes[2].text(0, 0.5, '高噪声、\n低数值\n（可考虑剪枝）',
                 ha='center', va='center', color='darkred', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"  [已保存] {save_path}")
    else:
        plt.show()

def plot_snr_analysis(model, threshold=1.0, save_path=None):
    """
    信噪比（Signal-to-Noise Ratio，SNR）分析。
    
    SNR = |mu| / sigma
    
    用作判断“网络是否学会了该权重”的启发式指标。
    - 高 SNR（>1）：均值的绝对值超过标准差，可能较重要，并非保证符号确定。
    - 低 SNR（<1）：噪声尺度大于均值绝对值，权重更受随机性影响。
    """
    print('  [可视化] 计算信噪比……')
    
    layers = [('第 1 层', model.layer1), ('第 2 层', model.layer2)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, (name, layer) in zip(axes, layers):
        mu = layer.w_mu.flatten()
        sigma = np.log1p(np.exp(-np.abs(layer.w_rho))) + np.maximum(layer.w_rho, 0)
        sigma = sigma.flatten()
        
        # 计算 SNR
        snr = np.abs(mu) / (sigma + 1e-9)
        
        # 排序以便可视化
        sorted_snr = np.sort(snr)[::-1]
        
        # 绘图
        ax.plot(sorted_snr, linewidth=2, color='crimson')
        ax.axhline(threshold, color='black', linestyle='--', label=f'阈值（{threshold})')
        
        # 阴影
        ax.fill_between(range(len(snr)), 0, sorted_snr, where=(sorted_snr > threshold),
                        color='green', alpha=0.3, label='有效权重')
        ax.fill_between(range(len(snr)), 0, sorted_snr, where=(sorted_snr <= threshold),
                        color='gray', alpha=0.3, label='噪声权重')
        
        # 统计量
        effective_count = np.sum(snr > threshold)
        total_count = len(snr)
        
        ax.set_title(f'{name} SNR 谱\n（{effective_count}/{total_count} 个有效）',
                     fontweight='bold')
        ax.set_xlabel('权重索引（已排序）')
        ax.set_ylabel('信噪比（|$\\mu$| / $\\sigma$）')
        ax.set_yscale('log')
        ax.legend()
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"  [已保存] {save_path}")
    else:
        plt.show()

def plot_prior_posterior_shift(model, save_path=None):
    """
    将学习可视化为“偏离先验”的过程。
    
    先验为 N(0, 1)（或类似分布）。
    训练后近似后验为 N(mu, sigma)。
    两者差异用于解释获得的信息。
    """
    print('  [可视化] 绘制先验与后验的偏移……')
    
    # 收集所有层的全部权重
    all_mu = []
    all_sigma = []
    
    for layer in [model.layer1, model.layer2]:
        all_mu.append(layer.w_mu.flatten())
        s = np.log1p(np.exp(-np.abs(layer.w_rho))) + np.maximum(layer.w_rho, 0)
        all_sigma.append(s.flatten())
        
    all_mu = np.concatenate(all_mu)
    all_sigma = np.concatenate(all_sigma)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 1. 绘制先验（标准正态分布）
    x_axis = np.linspace(-4, 4, 1000)
    ax.plot(x_axis, norm.pdf(x_axis, 0, 1), 'k--', linewidth=2, label='先验 $N(0,1)$')
    
    # 2. 绘制后验汇总分布
    # 用样本的核密度估计（KDE）近似后验密度
    sns.kdeplot(all_mu, fill=True, color='blue', alpha=0.3, 
                label='后验均值（学习结果）', ax=ax)
    
    ax.set_title('模型学到了多少？\n（与先验的差异）', fontsize=16, fontweight='bold')
    ax.set_xlabel('权重空间')
    ax.set_ylabel('Density')
    ax.legend(fontsize=12)
    
    # 添加解释文字
    ax.text(2.5, 0.3, '远离 0 的峰\n可能表示较强特征',
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"  [已保存] {save_path}")
    else:
        plt.show()

def plot_loss_dynamics(history, save_path=None):
    """
    绘制训练中的损失动态：复杂度与误差。
    展示优化轨迹。
    """
    print('  [可视化] 绘制损失动态……')
    epochs = range(len(history['total']))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 图 1：主要损失
    ax1.plot(epochs, history['total'], 'k-', linewidth=2, label='总损失（目标）')
    ax1.plot(epochs, history['nll'], 'b--', linewidth=2, label='误差（NLL）')
    ax1.set_ylabel('损失值')
    ax1.set_title('训练轨迹：最小化自由能', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图 2：复杂度代价（KL）
    # 模型学习结构时该项通常上升，随后趋于稳定
    ax2.plot(epochs, history['kl'], 'r-', linewidth=2, label='复杂度代价（KL）')
    ax2.set_ylabel('信息代价（奈特）')
    ax2.set_xlabel('Epochs')
    ax2.set_title('描述长度代价（模型的“文件大小”）', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"  [已保存] {save_path}")
    else:
        plt.show()

def analyze_compression_stats(model, threshold_snr=0.5):
    """
    生成压缩效率文本报告。
    基于 SNR 计算启发式有效参数数量。
    """
    print("\n" + "="*60)
    print('MDL 压缩报告')
    print("="*60)
    
    total_params = 0
    effective_params = 0
    total_bits_theor = 0
    
    for i, layer in enumerate([model.layer1, model.layer2]):
        name = f"层 {i+1}"
        mu = layer.w_mu.flatten()
        sigma = np.log1p(np.exp(-np.abs(layer.w_rho))) + np.maximum(layer.w_rho, 0)
        sigma = sigma.flatten()
        
        snr = np.abs(mu) / (sigma + 1e-9)
        n_layer = len(mu)
        n_eff = np.sum(snr > threshold_snr)
        
        # 所需比特的启发式近似：H(w) ≈ -log2(sigma) + const；不是高斯熵公式
        # sigma 较大（1.0）时估计约 0 比特；sigma 较小（0.01）时约 7 比特。
        bits = np.sum(np.maximum(0, -np.log2(sigma + 1e-9)))
        
        print(f"\n{name}:")
        print(f"  权重总数：      {n_layer}")
        print(f"  有效权重数：    {n_eff} (SNR > {threshold_snr})")
        print(f"  稀疏率：        {100*(1 - n_eff/n_layer):.1f}%")
        print(f"  平均不确定性：  {np.mean(sigma):.4f}")
        
        total_params += n_layer
        effective_params += n_eff
        total_bits_theor += bits
        
    print("-" * 60)
    print(f"整体系统：")
    print(f"  原始大小：      {total_params * 32} 比特（以 float32 为参照）")
    print(f"  MDL 编码大小估计：{int(total_bits_theor)} 比特（启发式估计，并非实际编码结果）")
    print(f"  压缩率估计：{total_params * 32 / (total_bits_theor + 1):.1f}x")
    print("=" * 60 + "\n")

def create_comprehensive_report(model, history, X, y, save_dir='mdl_analysis'):
    """
    组织并运行完整分析流程。
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"开始完整 MDL 分析 -> {save_dir}/")
    
    # 1. 不确定性
    plot_uncertainty_envelope(model, X, y, save_path=f"{save_dir}/1_uncertainty.png")
    
    # 2. 权重
    plot_weight_distributions(model, 'layer1', save_path=f"{save_dir}/2_weights_layer1.png")
    
    # 3. SNR
    plot_snr_analysis(model, save_path=f"{save_dir}/3_snr_analysis.png")
    
    # 4. 先验偏移
    plot_prior_posterior_shift(model, save_path=f"{save_dir}/4_prior_shift.png")
    
    # 5. 动态
    plot_loss_dynamics(history, save_path=f"{save_dir}/5_training_dynamics.png")
    
    # 6. 文本报告
    with open(f"{save_dir}/compression_report.txt", 'w') as f:
        import sys
        original = sys.stdout
        sys.stdout = f
        analyze_compression_stats(model)
        sys.stdout = original
        
    print('分析完成，所有产物已保存。')

if __name__ == "__main__":
    print('MDL 可视化库已加载。')
    print('在 train_mdl.py 中调用或在 Notebook 中导入（原提示脚本名待核对）。')