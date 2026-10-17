"""visualization.py——咖啡元胞自动机实验的绘图工具。

生成与论文对应的图：
  - 熵与复杂度随时间的变化（图 2、10）
  - 关键时刻的网格状态可视化（图 3、4、11、12）
  - 标度分析（图 6-8）
  - 压缩器比较（图 5）

论文：Aaronson、Carroll、Ouellette（2014），arXiv:1405.6903"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from implementation import (
    CoffeeAutomaton, run_simulation, coarse_grain,
    threshold_array, row_majority_adjust, gzip_size, grid_to_bytes
)


def plot_entropy_complexity(results: Dict, title: str = '咖啡元胞自动机',
                            ax: Optional[plt.Axes] = None):
    """在同一时间轴上绘制熵与复杂度。
复现图 2 和图 10 的左／右面板。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(results['times'], results['entropy'], 'b-', label='熵（细粒度状态的 gzip 大小）')
    ax.set_xlabel('时间步')
    ax.set_ylabel('压缩大小（字节）', color='b')
    ax.tick_params(axis='y', labelcolor='b')

    ax2 = ax.twinx()
    ax2.plot(results['times'], results['complexity'], 'r-', label='复杂度（粗粒化状态的 gzip 大小）')
    ax2.set_ylabel('复杂度（字节）', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # 标出复杂度峰值
    peak_idx = np.argmax(results['complexity'])
    ax2.axvline(results['times'][peak_idx], color='gray', linestyle='--', alpha=0.4)

    ax.set_title(title)

    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)

    return ax


def plot_grid_snapshots(results: Dict, num_snapshots: int = 5,
                        grain_size: int = 10, num_buckets: int = 7,
                        title: str = '网格状态随时间的变化'):
    """可视化关键时刻的自动机状态。
复现图 3、4、11、12。

上排显示细粒度状态，下排显示粗粒化状态。"""
    if 'grids' not in results or len(results['grids']) == 0:
        print("没有保存网格快照（是否 grid_size > 100？）")
        return

    grids = results['grids']
    times = results['times']
    complexities = results['complexity']

    # 选择快照：初始、峰值，以及均匀间隔的位置
    peak_idx = np.argmax(complexities)
    total = len(grids)
    indices = [0, peak_idx]

    # 添加均匀间隔的索引
    step = total // (num_snapshots - 1)
    for i in range(1, num_snapshots - 1):
        idx = i * step
        if idx not in indices:
            indices.append(idx)
    indices.append(total - 1)
    indices = sorted(set(indices))[:num_snapshots]

    fig, axes = plt.subplots(2, len(indices), figsize=(3 * len(indices), 6))

    for col, idx in enumerate(indices):
        grid = grids[idx]
        t = times[idx]

        # 细粒度状态（上排）
        axes[0, col].imshow(grid, cmap='gray_r', vmin=0, vmax=1)
        label = f't={t}'
        if idx == peak_idx:
            label += '（峰值）'
        axes[0, col].set_title(label, fontsize=9)
        axes[0, col].axis('off')

        # 粗粒化状态（下排）
        coarse = coarse_grain(grid, grain_size)
        discrete = threshold_array(coarse, num_buckets)
        discrete = row_majority_adjust(discrete)
        axes[1, col].imshow(discrete, cmap='gray_r', vmin=0, vmax=num_buckets - 1)
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('细粒度', fontsize=10)
    axes[1, 0].set_ylabel('粗粒化', fontsize=10)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def plot_model_comparison(interacting: Dict, non_interacting: Dict):
    """并排比较有相互作用与无相互作用模型。
复现图 2 和图 10 的布局。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 有相互作用
    axes[0, 0].plot(interacting['times'], interacting['entropy'], 'b-')
    axes[0, 0].set_title('有相互作用：熵')
    axes[0, 0].set_ylabel('gzip 大小（字节）')

    axes[0, 1].plot(interacting['times'], interacting['complexity'], 'r-')
    peak = np.argmax(interacting['complexity'])
    axes[0, 1].axvline(interacting['times'][peak], color='gray', ls='--', alpha=0.4)
    axes[0, 1].set_title('有相互作用：复杂度')
    axes[0, 1].set_ylabel('gzip 大小（字节）')

    # 无相互作用
    axes[1, 0].plot(non_interacting['times'], non_interacting['entropy'], 'b-')
    axes[1, 0].set_title('无相互作用：熵')
    axes[1, 0].set_xlabel('时间步')
    axes[1, 0].set_ylabel('gzip 大小（字节）')

    axes[1, 1].plot(non_interacting['times'], non_interacting['complexity'], 'r-')
    axes[1, 1].set_title('无相互作用：复杂度')
    axes[1, 1].set_xlabel('时间步')
    axes[1, 1].set_ylabel('gzip 大小（字节）')

    plt.suptitle('咖啡元胞自动机：有相互作用与无相互作用\n'
                 '(cf. Figures 2 and 10, Aaronson et al. 2014)')
    plt.tight_layout()
    return fig


def plot_scaling_analysis(scaling_results: Dict):
    """绘制标度行为，复现图 6-8。
最大熵 ~ n^2，最大复杂度 ~ n，达到峰值的时间 ~ n^2。"""
    sizes = scaling_results['sizes']
    x_fit = np.linspace(sizes[0], sizes[-1], 50)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 图 6：最大熵与 n 的关系
    axes[0].plot(sizes, scaling_results['max_entropy'], 'bo-', markersize=6)
    c = np.polyfit(sizes, scaling_results['max_entropy'], 2)
    r2 = 1 - np.sum((scaling_results['max_entropy'] - np.polyval(c, sizes))**2) / \
         np.sum((scaling_results['max_entropy'] - np.mean(scaling_results['max_entropy']))**2)
    axes[0].plot(x_fit, np.polyval(c, x_fit), 'b--', alpha=0.5,
                 label=f'二次拟合（r^2={r2:.4f}）')
    axes[0].set_xlabel('网格边长 n')
    axes[0].set_ylabel('最大熵（字节）')
    axes[0].set_title('最大熵 ~ n^2（图 6）')
    axes[0].legend(fontsize=8)

    # 图 7：最大复杂度与 n 的关系
    axes[1].plot(sizes, scaling_results['max_complexity'], 'ro-', markersize=6)
    c = np.polyfit(sizes, scaling_results['max_complexity'], 1)
    r2 = 1 - np.sum((scaling_results['max_complexity'] - np.polyval(c, sizes))**2) / \
         np.sum((scaling_results['max_complexity'] - np.mean(scaling_results['max_complexity']))**2)
    axes[1].plot(x_fit, np.polyval(c, x_fit), 'r--', alpha=0.5,
                 label=f'线性拟合（r^2={r2:.4f}）')
    axes[1].set_xlabel('网格边长 n')
    axes[1].set_ylabel('最大复杂度（字节）')
    axes[1].set_title('最大复杂度 ~ n（图 7）')
    axes[1].legend(fontsize=8)

    # 图 8：达到峰值的时间与 n 的关系
    axes[2].plot(sizes, scaling_results['peak_time'], 'go-', markersize=6)
    c = np.polyfit(sizes, scaling_results['peak_time'], 2)
    r2 = 1 - np.sum((scaling_results['peak_time'] - np.polyval(c, sizes))**2) / \
         np.sum((scaling_results['peak_time'] - np.mean(scaling_results['peak_time']))**2)
    axes[2].plot(x_fit, np.polyval(c, x_fit), 'g--', alpha=0.5,
                 label=f'二次拟合（r^2={r2:.4f}）')
    axes[2].set_xlabel('网格边长 n')
    axes[2].set_ylabel('达到峰值的时间')
    axes[2].set_title('达到峰值的时间 ~ n^2（图 8）')
    axes[2].legend(fontsize=8)

    plt.suptitle('标度分析——Aaronson 等（2014）的图 6-8')
    plt.tight_layout()
    return fig


def plot_compressor_comparison(grid_size: int = 50, total_steps: int = 100000,
                               num_snapshots: int = 80, seed: int = 42):
    """比较不同压缩算法，复现图 5。

论文表明，gzip、bzip2 等压缩器产生的复杂度曲线在定性上相似。"""
    import bz2
    import lzma

    # 只运行一次模拟，收集网格
    ca = CoffeeAutomaton(grid_size, 'interacting', seed)
    grain_size = max(2, grid_size // 10)
    steps_per = max(1, total_steps // num_snapshots)

    times = [0]
    gzip_cx = [0]
    bz2_cx = [0]
    lzma_cx = [0]

    # 初始测量
    grid = ca.get_binary_grid()
    coarse = coarse_grain(grid, grain_size)
    discrete = threshold_array(coarse, 7)
    discrete = row_majority_adjust(discrete)
    data = grid_to_bytes(discrete)

    gzip_cx[0] = len(gzip.compress(data, compresslevel=9))
    bz2_cx[0] = len(bz2.compress(data, compresslevel=9))
    lzma_cx[0] = len(lzma.compress(data))

    for i in range(num_snapshots):
        ca.step(steps_per)
        times.append(ca.time)

        grid = ca.get_binary_grid()
        coarse = coarse_grain(grid, grain_size)
        discrete = threshold_array(coarse, 7)
        discrete = row_majority_adjust(discrete)
        data = grid_to_bytes(discrete)

        gzip_cx.append(len(gzip.compress(data, compresslevel=9)))
        bz2_cx.append(len(bz2.compress(data, compresslevel=9)))
        lzma_cx.append(len(lzma.compress(data)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, gzip_cx, 'b-', label='gzip')
    ax.plot(times, bz2_cx, 'r-', label='bzip2')
    ax.plot(times, lzma_cx, 'g-', label='lzma')

    ax.set_xlabel('时间步')
    ax.set_ylabel('粗粒化状态的压缩大小（字节）')
    ax.set_title('使用不同压缩器的复杂度估计\n'
                 '（参见 Aaronson 等，2014，图 5）')
    ax.legend()

    plt.tight_layout()
    return fig


def plot_complexity_curve_annotated(results: Dict, model: str = 'interacting'):
    """带注释的单条复杂度曲线，展示先升后降。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    times = results['times']
    cx = results['complexity']

    ax.plot(times, cx, 'r-', linewidth=2)
    ax.fill_between(times, cx, alpha=0.1, color='r')

    peak_idx = np.argmax(cx)
    peak_t = times[peak_idx]
    peak_v = cx[peak_idx]

    ax.annotate(f'峰值：{peak_v} 字节\nt = {peak_t}',
                xy=(peak_t, peak_v),
                xytext=(peak_t + (times[-1] - times[0]) * 0.1, peak_v * 0.9),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10)

    # 标注各阶段
    third = len(times) // 3
    ax.axvspan(times[0], times[third], alpha=0.05, color='blue', label='分离')
    ax.axvspan(times[third], times[2 * third], alpha=0.05, color='red', label='复杂结构')
    ax.axvspan(times[2 * third], times[-1], alpha=0.05, color='green', label='充分混合')

    ax.set_xlabel('时间步')
    ax.set_ylabel('表观复杂度（字节）')
    ax.set_title(f'复杂度的兴衰——{model.title()} 模型')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 主程序：生成所有关键图
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("正在为咖啡元胞自动机论文生成可视化……")
    print("=" * 55)

    # 运行快速模拟
    print("\n1. 正在运行有相互作用模拟（50x50，50000 步）……")
    results = run_simulation(grid_size=50, total_steps=50000, num_snapshots=80,
                             model='interacting', seed=42)

    print("2. 正在绘制熵与复杂度……")
    plot_entropy_complexity(results, 'Interacting Model (50x50)')

    print("3. 正在绘制网格快照……")
    plot_grid_snapshots(results, num_snapshots=5, grain_size=5)

    print("4. 正在绘制带注释的复杂度曲线……")
    plot_complexity_curve_annotated(results)

    print("\n5. 正在比较压缩器……")
    plot_compressor_comparison(grid_size=50, total_steps=50000, num_snapshots=60)

    plt.show()
    print("\n完成。")
