"""
练习 5：进阶 MDL（压缩比分析）
======================================================

目标：分析完整 MDL 目标，理解压缩前沿。

任务：
- 分别计算数据拟合损失与 KL 散度
- 计算有效压缩比
- 绘制两项之间的关系
- 寻找压缩的帕累托前沿

学习目标：
1. MDL = -log(data fit) - log(prior knowledge)（原文示意式，严格目标需明确概率定义）
2. 权衡：愿意牺牲多少准确性来换取压缩？
3. 帕累托效率：无法同时改善两个目标
4. 实际应用（边缘设备、移动端）

用时：30-40 分钟
难度：困难
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 将父目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from implementation import BayesianLinearNetwork


def analyze_loss_components(model, X_test, y_test):
    """
    将损失分解为数据拟合项和正则化项。
    
    参数：
        model：训练好的 BayesianLinearNetwork（该类在当前实现中不存在，原有接口问题保留）
        X_test：测试输入
        y_test：测试目标
        
    返回：
        reconstruction_loss：数据拟合程度
        kl_divergence：与先验的差异程度
        mdl_loss：组合损失（越低表示该目标越优）
        
    MDL 原理：
        L_MDL = L_data + KL_divergence
        
    平衡两个目标：
        1. L_data：充分解释数据（残差小）
        2. KL：靠近先验（权重简单）
    
    好模型兼顾两者！
    """
    
    # TODO 1：前向传播
    # predictions = model.forward(X_test)
    
    # TODO 2：计算重构损失（MSE）
    # reconstruction_loss = np.mean((predictions - y_test)**2)
    
    # TODO 3：由模型参数计算 KL 散度
    # 需要访问网络的权重分布
    # kl_divergence = model.compute_kl_divergence()
    
    # TODO 4：组合为 MDL 损失
    # mdl_loss = reconstruction_loss + kl_divergence
    
    # TODO 5：返回各组成项
    # return reconstruction_loss, kl_divergence, mdl_loss
    
    pass


def study_pareto_frontier():
    """
    使用不同 kl_weights 训练网络并绘制权衡关系。
    """
    print('生成数据集……')
    # TODO 6：创建数据集（含噪正弦波）
    X_train = np.random.uniform(0, 4*np.pi, 200)[:, np.newaxis]
    y_train = np.sin(X_train.flatten()) + np.random.randn(200) * 0.15
    
    X_test = np.linspace(0, 4*np.pi, 500)[:, np.newaxis]
    y_test = np.sin(X_test.flatten())
    
    # TODO 7：定义覆盖压缩前沿的 kl_weights
    kl_weights = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    
    pareto_data = {
        'kl_weights': [],
        'reconstruction_loss': [],
        'kl_divergence': [],
        'mdl_loss': [],
        'compression_ratio': []
    }
    
    print('\n跨压缩前沿训练模型……\n')
    
    # TODO 8：对每个 kl_weight：
    for kl_w in kl_weights:
        print(f"  β = {kl_w:8.5f}", end="", flush=True)
        
        # TODO 9：训练模型
        # model = BayesianLinearNetwork(
        #     input_dim=1,
        #     hidden_dims=[64, 32],
        #     output_dim=1,
        #     kl_weight=kl_w
        # )
        # losses = model.train(X_train, y_train, epochs=500, lr=0.01, verbose=False)
        
        # TODO 10：分析损失组成项
        # recon_loss, kl_div, mdl_loss = analyze_loss_components(model, X_test, y_test)
        
        # TODO 11：计算压缩比（原文设定越低越好；此比值不是实际编码压缩率）
        # compression_ratio = kl_div / (recon_loss + 1e-8)
        
        # TODO 12：存储结果
        # pareto_data['kl_weights'].append(kl_w)
        # pareto_data['reconstruction_loss'].append(recon_loss)
        # pareto_data['kl_divergence'].append(kl_div)
        # pareto_data['mdl_loss'].append(mdl_loss)
        # pareto_data['compression_ratio'].append(compression_ratio)
        
        # TODO 13：打印并解释结果
        # print(f" → Recon={recon_loss:.4f}, KL={kl_div:.4f}, Ratio={compression_ratio:.2f}")
        
        pass
    
    # 可视化
    print('\n\n绘制帕累托前沿……')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 图 1：帕累托前沿（重构损失与 KL）
    ax = axes[0, 0]
    if 'pareto_data' in locals() and pareto_data['reconstruction_loss']:
        ax.scatter(pareto_data['reconstruction_loss'], 
                   pareto_data['kl_divergence'], 
                   s=100, c=range(len(pareto_data['kl_weights'])), 
                   cmap='viridis', alpha=0.7)
        for i, (x, y, kl_w) in enumerate(zip(pareto_data['reconstruction_loss'],
                                             pareto_data['kl_divergence'],
                                             pareto_data['kl_weights'])):
            if i % 2 == 0:  # 每隔一个点添加标签
                ax.annotate(f'β={kl_w:.2e}', (x, y), fontsize=8)
        ax.set_xlabel('重构损失（数据拟合）')
        ax.set_ylabel('KL 散度（正则化）')
        ax.set_title('帕累托前沿：压缩权衡')
        ax.grid(True, alpha=0.3)
    
    # 图 2：不同权重下的 MDL 损失
    ax = axes[0, 1]
    if 'pareto_data' in locals() and pareto_data['mdl_loss']:
        ax.semilogx(pareto_data['kl_weights'], pareto_data['mdl_loss'], 'o-', linewidth=2)
        ax.set_xlabel('KL 权重（β）')
        ax.set_ylabel('MDL 损失')
        ax.set_title('总压缩代价（越低越好）')
        ax.grid(True, alpha=0.3)
    
    # 图 3：压缩比
    ax = axes[1, 0]
    if 'pareto_data' in locals() and pareto_data['compression_ratio']:
        ax.loglog(pareto_data['kl_weights'], pareto_data['compression_ratio'], 's-', 
                  linewidth=2, color='purple')
        ax.set_xlabel('KL 权重（β）')
        ax.set_ylabel('压缩比指标（KL/Recon）')
        ax.set_title('模型简洁性与数据拟合')
        ax.grid(True, alpha=0.3)
    
    # 图 4：组成项分解
    ax = axes[1, 1]
    if 'pareto_data' in locals() and pareto_data['kl_weights']:
        width = 0.35
        x = np.arange(len(pareto_data['kl_weights']))
        ax.bar(x - width/2, pareto_data['reconstruction_loss'], width, label='数据拟合')
        ax.bar(x + width/2, pareto_data['kl_divergence'], width, label='Regularization')
        ax.set_xlabel('KL 权重（β）')
        ax.set_ylabel('Loss')
        ax.set_title('损失组成项分解')
        # TODO 14：设置 x 轴刻度以显示 kl_weights
        # ax.set_xticks(x)
        # ax.set_xticklabels([f'{kl_w:.2e}' for kl_w in pareto_data['kl_weights']], 
        #                    rotation=45, fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / 'pareto_frontier.png', dpi=150, bbox_inches='tight')
    print('  已保存至：pareto_frontier.png')
    plt.close()
    
    print('\n[ok] 帕累托前沿分析完成。')
    print('\n关键认识：')
    print('  - 左侧：原文标为欠拟合（KL 过高），与横轴定义不一致，需结合图核对')
    print('  - 右侧：原文标为过拟合（重构损失过高），与横轴定义不一致，需结合图核对')
    print('  - 中间：压缩与拟合的折中点')
    print('\n帕累托最优：无法同时改善两个目标！')


if __name__ == "__main__":
    study_pareto_frontier()
