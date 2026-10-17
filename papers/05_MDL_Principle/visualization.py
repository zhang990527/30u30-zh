"""
第 5 天：最小描述长度原理——可视化
===================================================================

用于理解 MDL 模型选择的图形：
- 模型复杂度与数据拟合的权衡
- MDL 得分分解（模型代价 + 数据代价）
- MDL、AIC、BIC 比较
- 压缩比估计可视化
- 预序贯编码可视化

运行脚本观察 MDL 的表现！
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 从本项目实现中导入
from implementation import (
    two_part_mdl_polynomial,
    select_polynomial_degree,
    prequential_mdl_polynomial,
    prequential_model_selection,
    compare_mdl_aic_bic,
    aic_score,
    bic_score,
    compression_ratio,
    generate_test_data,
    mdl_model_probability
)


# =============================================================================
# 样式配置
# =============================================================================

# 配色方案
COLORS = {
    'mdl': '#2E86AB',       # 蓝色：MDL
    'aic': '#A23B72',       # 品红：AIC
    'bic': '#F18F01',       # 橙色：BIC
    'model_cost': '#E94F37',  # 红色：模型代价
    'data_cost': '#44AF69',   # 绿色：数据代价
    'total': '#3D348B',       # 紫色：总和
    'true': '#2E7D32',        # 深绿：真实模型
    'fit': '#1E88E5',         # 蓝色：拟合结果
    'data': '#424242',        # 灰色：数据点
    'background': '#F5F5F5',  # 浅灰背景
}

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': COLORS['background'],
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})


# =============================================================================
# 图 1：MDL 得分分解
# =============================================================================

def plot_mdl_breakdown(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
    true_degree: int = None,
    save_path: str = None
):
    """
    可视化 MDL 得分分解：模型代价与数据代价。
    
    这是理解 MDL 的关键可视化！
    
    图中展示：
    - 蓝色柱：模型代价 L(H)，随复杂度增加
    - 绿色柱：数据代价 L(D|H)，通常随拟合改善而降低
    - 紫色线：总 MDL，寻找其最小值
    
    参数：
        x, y：数据点
        max_degree：要展示的最高多项式次数
        true_degree：若已知，则标注真实次数
        save_path：图形保存路径
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    degrees = list(range(max_degree + 1))
    model_costs = []
    data_costs = []
    totals = []
    
    for deg in degrees:
        total, model, data = two_part_mdl_polynomial(x, y, deg)
        model_costs.append(model)
        data_costs.append(data)
        totals.append(total)
    
    best_degree = np.argmin(totals)
    
    # 柱宽
    width = 0.35
    x_pos = np.arange(len(degrees))
    
    # 堆叠柱状图
    bars1 = ax.bar(x_pos, model_costs, width, label='Model Cost L(H)',
                   color=COLORS['model_cost'], alpha=0.8)
    bars2 = ax.bar(x_pos, data_costs, width, bottom=model_costs,
                   label='Data Cost L(D|H)', color=COLORS['data_cost'], alpha=0.8)
    
    # 总和曲线
    ax.plot(x_pos, totals, 'o-', color=COLORS['total'], linewidth=2.5,
            markersize=8, label='Total MDL', zorder=5)
    
    # 标记最佳次数
    ax.axvline(x=best_degree, color=COLORS['total'], linestyle='--',
               linewidth=2, alpha=0.7, label=f'Best (Degree {best_degree})')
    
    # 若已知则标记真实次数
    if true_degree is not None:
        ax.axvline(x=true_degree, color=COLORS['true'], linestyle=':',
                   linewidth=2, alpha=0.7, label=f'True (Degree {true_degree})')
    
    ax.set_xlabel('Polynomial Degree', fontsize=12)
    ax.set_ylabel('Code Length (bits)', fontsize=12)
    ax.set_title('MDL Score Breakdown: The Trade-off\n'
                 '"Simple model + many exceptions" vs "Complex model + few exceptions"',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(degrees)
    ax.legend(loc='upper right')
    
    # 添加说明
    ax.annotate(f'Optimal:\nDegree {best_degree}',
                xy=(best_degree, totals[best_degree]),
                xytext=(best_degree + 1.5, totals[best_degree] + 50),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# 图 2：多项式拟合
# =============================================================================

def plot_polynomial_fits(
    x: np.ndarray,
    y: np.ndarray,
    degrees_to_show: List[int] = [1, 2, 5, 9],
    true_degree: int = None,
    save_path: str = None
):
    """
    并排展示不同的多项式拟合。
    
    直观展示欠拟合、合适拟合与过拟合。
    """
    n_plots = len(degrees_to_show)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    
    if n_plots == 1:
        axes = [axes]
    
    # 获取 MDL 得分
    result = select_polynomial_degree(x, y, max_degree=max(degrees_to_show))
    best_degree = result['best_degree']
    
    x_fine = np.linspace(x.min(), x.max(), 200)
    
    for ax, deg in zip(axes, degrees_to_show):
        # 绘制数据
        ax.scatter(x, y, c=COLORS['data'], alpha=0.6, s=30, label='Data')
        
        # 拟合并绘制多项式
        coeffs = np.polyfit(x, y, deg)
        poly = np.poly1d(coeffs)
        y_fit = poly(x_fine)
        
        # 根据是否最优设置颜色
        if deg == best_degree:
            color = COLORS['true']
            title_extra = "\n[ok] OPTIMAL (MDL)"
        elif deg < best_degree:
            color = COLORS['aic']
            title_extra = "\n[FAIL] Underfit"
        else:
            color = COLORS['model_cost']
            title_extra = "\n[FAIL] Overfit"
        
        ax.plot(x_fine, y_fit, color=color, linewidth=2.5, label=f'Degree {deg}')
        
        # MDL 得分
        mdl_score = result['scores'][deg]
        
        ax.set_title(f'Degree {deg}{title_extra}\nMDL: {mdl_score:.0f} bits',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='best', fontsize=8)
    
    fig.suptitle('Polynomial Fitting: Underfit → Optimal → Overfit',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# 图 3：MDL、AIC 与 BIC 比较
# =============================================================================

def plot_mdl_vs_aic_bic(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
    true_degree: int = None,
    save_path: str = None
):
    """
    比较 MDL、AIC 与 BIC 模型选择准则。
    
    展示三种方法何时一致、何时不同。
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    comparison = compare_mdl_aic_bic(x, y, max_degree)
    
    degrees = list(range(max_degree + 1))
    
    # 为比较曲线形状进行最小—最大归一化，不表示准则原始数值可直接比较
    def normalize(scores):
        vals = list(scores.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return {k: 0.5 for k in scores}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}
    
    methods = [
        ('MDL', comparison['mdl_scores'], comparison['mdl_best'], COLORS['mdl']),
        ('AIC', comparison['aic_scores'], comparison['aic_best'], COLORS['aic']),
        ('BIC', comparison['bic_scores'], comparison['bic_best'], COLORS['bic'])
    ]
    
    for ax, (name, scores, best, color) in zip(axes, methods):
        vals = [scores[d] for d in degrees]
        
        ax.bar(degrees, vals, color=color, alpha=0.7, edgecolor=color)
        ax.axvline(x=best, color='black', linestyle='--', linewidth=2,
                   label=f'Selected: {best}')
        
        if true_degree is not None:
            ax.axvline(x=true_degree, color=COLORS['true'], linestyle=':',
                       linewidth=2, label=f'True: {true_degree}')
        
        ax.set_xlabel('Polynomial Degree')
        ax.set_ylabel('Score (lower is better)')
        ax.set_title(f'{name}\nSelects: Degree {best}',
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.set_xticks(degrees)
    
    fig.suptitle('Model Selection Showdown: MDL vs AIC vs BIC',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# 图 4：预序贯 MDL 可视化
# =============================================================================

def plot_prequential_coding(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 2,
    save_path: str = None
):
    """
    可视化预序贯（顺序预测）编码过程。
    
    展示模型如何逐步学习并预测。
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    n = len(x)
    start_idx = degree + 2
    
    # 收集随时间变化的预测误差
    errors = []
    predictions = []
    actuals = []
    sigmas = []
    
    for i in range(start_idx, n):
        x_train = x[:i]
        y_train = y[:i]
        
        coeffs = np.polyfit(x_train, y_train, degree)
        poly = np.poly1d(coeffs)
        
        y_pred = poly(x[i])
        predictions.append(y_pred)
        actuals.append(y[i])
        errors.append(y[i] - y_pred)
        
        if len(y_train) > 1:
            sigmas.append(np.std(y_train - poly(x_train)))
        else:
            sigmas.append(1.0)
    
    time_steps = list(range(start_idx, n))
    
    # 图 1：预测与实际值
    ax1 = axes[0, 0]
    ax1.scatter(x, y, c=COLORS['data'], alpha=0.5, s=30, label='All data')
    ax1.scatter(x[start_idx:], actuals, c=COLORS['true'], s=50,
                marker='s', label='Test points', zorder=5)
    ax1.scatter(x[start_idx:], predictions, c=COLORS['mdl'], s=50,
                marker='^', label='Predictions', zorder=5)
    
    for i, (actual, pred) in enumerate(zip(actuals, predictions)):
        ax1.plot([x[start_idx + i], x[start_idx + i]], [actual, pred],
                 'r-', alpha=0.3, linewidth=1)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Prequential: Predict → Observe → Update',
                  fontsize=11, fontweight='bold')
    ax1.legend()
    
    # 图 2：预测误差
    ax2 = axes[0, 1]
    colors = ['red' if e > 0 else 'blue' for e in errors]
    ax2.bar(time_steps, errors, color=colors, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.fill_between(time_steps, [-s for s in sigmas], sigmas,
                     color='gray', alpha=0.2, label='±σ')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Prediction Error')
    ax2.set_title('Prediction Errors Over Time',
                  fontsize=11, fontweight='bold')
    
    # 图 3：累积码长
    ax3 = axes[1, 0]
    code_lengths = []
    cumulative = 0
    for i, (err, sigma) in enumerate(zip(errors, sigmas)):
        sigma = max(sigma, 1e-10)
        bits = 0.5 * np.log2(2 * np.pi * sigma**2) + err**2 / (2 * sigma**2 * np.log(2))
        cumulative += bits
        code_lengths.append(cumulative)
    
    ax3.plot(time_steps, code_lengths, color=COLORS['total'], linewidth=2)
    ax3.fill_between(time_steps, 0, code_lengths, color=COLORS['total'], alpha=0.3)
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Cumulative Code Length (bits)')
    ax3.set_title('Prequential MDL Score Accumulation',
                  fontsize=11, fontweight='bold')
    
    # 图 4：不同次数比较
    ax4 = axes[1, 1]
    preq_result = prequential_model_selection(x, y, max_degree=8)
    
    degrees = list(preq_result['scores'].keys())[:9]
    scores = [preq_result['scores'][d] for d in degrees]
    
    bars = ax4.bar(degrees, scores, color=COLORS['mdl'], alpha=0.7)
    bars[preq_result['best_degree']].set_color(COLORS['true'])
    
    ax4.set_xlabel('Polynomial Degree')
    ax4.set_ylabel('Prequential MDL (bits)')
    ax4.set_title(f'Best Degree: {preq_result["best_degree"]} (Prequential)',
                  fontsize=11, fontweight='bold')
    ax4.set_xticks(degrees)
    
    fig.suptitle('Prequential MDL: Sequential Prediction-Based Model Selection',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# 图 5：压缩比估计可视化
# =============================================================================

def plot_compression_analysis(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
    save_path: str = None
):
    """
    可视化不同模型的压缩效果估计。
    
    展示合适模型可能压缩数据，而过拟合模型可能得不偿失。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    raw_bits = len(y) * 32  # 假设使用 32 比特浮点数
    
    degrees = list(range(max_degree + 1))
    mdl_scores = []
    compression_ratios = []
    
    for deg in degrees:
        total, _, _ = two_part_mdl_polynomial(x, y, deg)
        mdl_scores.append(total)
        compression_ratios.append(raw_bits / total if total > 0 else 0)
    
    best_degree = np.argmax(compression_ratios)
    
    # 图 1：原始大小与压缩估计
    ax1 = axes[0]
    
    ax1.axhline(y=raw_bits, color='gray', linestyle='--', linewidth=2,
                label=f'Raw data: {raw_bits} bits')
    ax1.plot(degrees, mdl_scores, 'o-', color=COLORS['total'], linewidth=2,
             markersize=8, label='MDL compressed')
    
    ax1.fill_between(degrees, mdl_scores, raw_bits, 
                     where=[s < raw_bits for s in mdl_scores],
                     color=COLORS['data_cost'], alpha=0.3, label='Compression savings')
    
    ax1.axvline(x=best_degree, color=COLORS['true'], linestyle=':',
                linewidth=2, label=f'Best compression: Degree {best_degree}')
    
    ax1.set_xlabel('Polynomial Degree')
    ax1.set_ylabel('Size (bits)')
    ax1.set_title('Raw Data vs MDL Compressed',
                  fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.set_xticks(degrees)
    
    # 图 2：压缩比估计
    ax2 = axes[1]
    
    colors = [COLORS['true'] if r == max(compression_ratios) else COLORS['mdl']
              for r in compression_ratios]
    
    bars = ax2.bar(degrees, compression_ratios, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=0.5)
    
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                label='No compression (ratio = 1)')
    
    ax2.set_xlabel('Polynomial Degree')
    ax2.set_ylabel('Compression Ratio')
    ax2.set_title('Compression Ratio by Model Complexity\n(Higher = Better Compression)',
                  fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.set_xticks(degrees)
    
    # 在柱上添加数值标签
    for bar, ratio in zip(bars, compression_ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{ratio:.1f}x', ha='center', va='bottom', fontsize=8)
    
    fig.suptitle('Compression Reveals Regularity: MDL Score Breakdown',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# 图 6：模型概率权重可视化
# =============================================================================

def plot_model_probabilities(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
    true_degree: int = None,
    save_path: str = None
):
    """
    可视化由码长权重归一化得到的近似模型概率，未必是真实贝叶斯后验。
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    result = select_polynomial_degree(x, y, max_degree=max_degree)
    probs = mdl_model_probability(result['scores'])
    
    degrees = list(range(max_degree + 1))
    prob_values = [probs[d] for d in degrees]
    
    # 创建柱状图
    colors = [COLORS['true'] if probs[d] == max(prob_values) else COLORS['mdl']
              for d in degrees]
    
    bars = ax.bar(degrees, prob_values, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    
    if true_degree is not None:
        ax.axvline(x=true_degree, color=COLORS['model_cost'], linestyle='--',
                   linewidth=2, label=f'True degree: {true_degree}')
    
    ax.set_xlabel('Polynomial Degree', fontsize=12)
    ax.set_ylabel('Posterior Probability P(model | data)', fontsize=12)
    ax.set_title('Model Posterior Probabilities from MDL\n'
                 '(Derived from 2^(-MDL score))',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(degrees)
    ax.legend()
    
    # 添加概率标签
    for bar, prob in zip(bars, prob_values):
        if prob > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.2%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# 图 7：编码类比可视化
# =============================================================================

def plot_spy_analogy(save_path: str = None):
    """
    可视化论文两段式编码思想的类比（原文实际对应 §2.4，而非原仓库所称第 3 节）。
    
    展示逐项编码与基于模型编码的对比。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 生成示例“温度”数据
    np.random.seed(42)
    hours = np.arange(24)
    base_temp = 15 + 10 * np.sin((hours - 6) * np.pi / 12)  # 中午达到峰值
    actual_temp = base_temp + np.random.randn(24) * 2  # 添加噪声
    
    # 左侧：逐项编码
    ax1 = axes[0]
    ax1.bar(hours, actual_temp, color=COLORS['data'], alpha=0.7, edgecolor='black')
    
    for h, t in zip(hours, actual_temp):
        ax1.annotate(f'{t:.1f}', (h, t + 0.5), fontsize=7, ha='center', rotation=90)
    
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('NAIVE ENCODING: Send Every Value\n'
                  f'Cost: 24 values × 32 bits = 768 bits',
                  fontsize=12, fontweight='bold', color=COLORS['model_cost'])
    ax1.set_xticks(hours)
    
    # 右侧：基于模型的编码
    ax2 = axes[1]
    
    # 绘制实际值
    ax2.scatter(hours, actual_temp, c=COLORS['data'], s=50, label='Actual', zorder=5)
    
    # 绘制拟合结果
    hours_fine = np.linspace(0, 23, 100)
    model_pred = 15 + 10 * np.sin((hours_fine - 6) * np.pi / 12)
    ax2.plot(hours_fine, model_pred, color=COLORS['mdl'], linewidth=2.5,
             label='Model: T = 15 + 10·sin((h-6)π/12)')
    
    # 展示残差
    model_at_hours = 15 + 10 * np.sin((hours - 6) * np.pi / 12)
    for h, actual, pred in zip(hours, actual_temp, model_at_hours):
        ax2.plot([h, h], [actual, pred], 'r-', alpha=0.4, linewidth=1)
    
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('MODEL-BASED ENCODING: Pattern + Residuals\n'
                  'Cost: ~50 bits (model) + ~150 bits (residuals) = ~200 bits',
                  fontsize=12, fontweight='bold', color=COLORS['true'])
    ax2.set_xticks(hours[::2])
    ax2.legend()
    
    # 添加文本框
    ax1.text(0.5, 0.02, '[FAIL] Expensive!\nNo structure found',
             transform=ax1.transAxes, fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.text(0.5, 0.02, '[ok] Efficient!\nPattern + small corrections',
             transform=ax2.transAxes, fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('The Coding Analogy: How MDL Finds Patterns',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# 主程序：生成所有可视化
# =============================================================================

def main():
    """Generate all MDL visualizations."""
    
    print("=" * 60)
    print("MDL Principle - Visualization Gallery")
    print("=" * 60)
    
    # 生成测试数据
    np.random.seed(42)
    true_degree = 2
    x, y, _ = generate_test_data(
        true_degree=true_degree,
        n_points=50,
        noise_std=2.0,
        seed=42
    )
    
    print(f"\nGenerated test data: n={len(x)}, true degree={true_degree}")
    print("\nGenerating visualizations...\n")
    
    # 1. MDL 分解
    print("1/7: MDL Score Breakdown...")
    fig1 = plot_mdl_breakdown(x, y, max_degree=8, true_degree=true_degree)
    plt.figure(fig1.number)
    plt.show(block=False)
    
    # 2. 多项式拟合
    print("2/7: Polynomial Fits Comparison...")
    fig2 = plot_polynomial_fits(x, y, degrees_to_show=[1, 2, 5, 8],
                                 true_degree=true_degree)
    plt.figure(fig2.number)
    plt.show(block=False)
    
    # 3. MDL、AIC 与 BIC
    print("3/7: MDL vs AIC vs BIC...")
    fig3 = plot_mdl_vs_aic_bic(x, y, max_degree=8, true_degree=true_degree)
    plt.figure(fig3.number)
    plt.show(block=False)
    
    # 4. 预序贯编码
    print("4/7: Prequential MDL...")
    fig4 = plot_prequential_coding(x, y, degree=true_degree)
    plt.figure(fig4.number)
    plt.show(block=False)
    
    # 5. 压缩分析
    print("5/7: Compression Analysis...")
    fig5 = plot_compression_analysis(x, y, max_degree=8)
    plt.figure(fig5.number)
    plt.show(block=False)
    
    # 6. 模型概率权重
    print("6/7: Model Probabilities...")
    fig6 = plot_model_probabilities(x, y, max_degree=8, true_degree=true_degree)
    plt.figure(fig6.number)
    plt.show(block=False)
    
    # 7. 编码类比
    print("7/7: Coding Analogy...")
    fig7 = plot_spy_analogy()
    plt.figure(fig7.number)
    plt.show(block=False)
    
    print("\n" + "=" * 60)
    print("All visualizations generated!")
    print("Close the windows to exit, or they will remain open.")
    print("=" * 60)
    
    plt.show()  # 阻塞，直到所有窗口关闭


if __name__ == "__main__":
    main()
