"""visualization.py —— 咖啡混合复杂性可视化工具

生成图表，探索“复杂动力学第一定律”所猜想的熵与复杂性差异：熵趋于增加，复杂性可能在中间时刻达到峰值。
核验说明：gzip 代理量未必出现峰值，局部熵也不保证逐步单调，图表不能证明猜想。
生成的图表：
1. 复杂性隆起（gzip 大小随时间变化）
2. 熵与复杂性的并排比较
3. 关键时刻网格快照
4. 度量比较（gzip、粗粒化、两段式编码）
5. 多尺度热图
6. 精致复杂度（第一部分）轨迹
7. 混合动画帧
参考：Scott Aaronson（2011）—— https://scottaaronson.blog/?p=762
用法：
    python visualization.py
    python visualization.py --grid-size 64 --steps 30000"""

import argparse
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
except ImportError:
    print('可视化需要 matplotlib。')
    print('安装命令：pip install matplotlib')
    sys.exit(1)

from implementation import (
    run_simulation,
    create_initial_grid,
    batch_swap,
    gzip_complexity,
    coarse_grained_kc,
    two_part_code,
    grid_entropy,
    mean_local_entropy,
    boundary_fraction,
    multiscale_kc,
    sophistication_proxy,
    compute_normalization_bounds,
    normalize_complexity,
)


def plot_complexity_hump(results, output_dir='.'):
    """图 1：绘制复杂性（gzip 大小）随时间变化，检验是否出现第一定律猜想启发的隆起；gzip 大小本身不是复杂熵。"""
    times = results['times']
    gzip_c = results['gzip_complexity']

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(times, gzip_c, 'r-', linewidth=2, label='gzip 复杂度')

    # 标记峰值
    peak_idx = np.argmax(gzip_c)
    ax.plot(times[peak_idx], gzip_c[peak_idx], 'ko', markersize=10, zorder=5)
    ax.annotate(
        f'峰值：{gzip_c[peak_idx]} 字节\n交换尝试次数：{times[peak_idx]:,}',
        xy=(times[peak_idx], gzip_c[peak_idx]),
        xytext=(times[peak_idx] * 1.1, gzip_c[peak_idx] * 0.95),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='black'),
    )

    # 标注区域
    ax.axvspan(0, times[peak_idx] * 0.3, alpha=0.1, color='blue',
               label='简单（分离）')
    ax.axvspan(times[peak_idx] * 0.6, times[peak_idx] * 1.4, alpha=0.1,
               color='red', label='复杂（卷须纹理）')
    ax.axvspan(times[-1] * 0.7, times[-1], alpha=0.1, color='green',
               label='粗粒度下简单（混合）')

    ax.set_xlabel('交换尝试总次数', fontsize=12)
    ax.set_ylabel('压缩大小（字节）', fontsize=12)
    ax.set_title(
        '复杂性隆起检验\n'
        'Aaronson《复杂动力学第一定律》（2011）',
        fontsize=14
    )
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = f"{output_dir}/plot1_complexity_hump.png"
    plt.savefig(outpath, dpi=150)
    print(f"  已保存：{outpath}")
    plt.close()


def plot_entropy_vs_complexity(results, output_dir='.'):
    """图 2：并排比较熵（原说明预期单调）与复杂性（预期隆起）。这是博文讨论的核心差别，但代理量不保证满足预期。"""
    times = results['times']
    entropy = results['entropy']
    gzip_c = results['gzip_complexity']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 熵
    axes[0].plot(times, entropy, 'b-', linewidth=2)
    axes[0].set_title('全局熵（本模拟中守恒）', fontsize=13)
    axes[0].set_xlabel('交换尝试总次数')
    axes[0].set_ylabel('香农熵（比特）')
    axes[0].grid(True, alpha=0.3)

    # 复杂性
    axes[1].plot(times, gzip_c, 'r-', linewidth=2)
    peak_idx = np.argmax(gzip_c)
    axes[1].axvline(x=times[peak_idx], color='gray', linestyle='--', alpha=0.5)
    axes[1].plot(times[peak_idx], gzip_c[peak_idx], 'ko', markersize=8)
    axes[1].set_title('复杂性（检验是否先升后降）', fontsize=13)
    axes[1].set_xlabel('交换尝试总次数')
    axes[1].set_ylabel('gzip 压缩大小（字节）')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        '熵与复杂性：核心问题\n'
        '为什么有趣程度会达到峰值，而熵趋于增加？',
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    outpath = f"{output_dir}/plot2_entropy_vs_complexity.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  已保存：{outpath}")
    plt.close()


def plot_grid_snapshots(results, output_dir='.'):
    """图 3：关键时刻的网格快照，直观展示分离 -> 卷须纹理 -> 混合。"""
    grids = results['grids']
    times = results['times']
    n = len(grids)

    # 选取 6 张等间隔快照
    indices = [0] + [int(n * f) for f in [0.1, 0.25, 0.5, 0.75]] + [n - 1]
    indices = [min(i, n - 1) for i in indices]

    fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3.5))

    for ax, idx in zip(axes, indices):
        ax.imshow(grids[idx], cmap='gray', interpolation='nearest')
        ax.set_title(f"t={times[idx]:,}", fontsize=10)
        ax.axis('off')

    fig.suptitle(
        '咖啡随时间混合\n'
        '分离（简单）--> 卷须纹理（复杂）--> 粗粒度混合（简单）',
        fontsize=13
    )
    plt.tight_layout()
    outpath = f"{output_dir}/plot3_grid_snapshots.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  已保存：{outpath}")
    plt.close()


def plot_measures_comparison(results, output_dir='.'):
    """图 4：对同一模拟比较全部复杂性度量：gzip、粗粒化 KC、精致复杂度代理量（两段式编码第一部分）。"""
    times = results['times']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # gzip 复杂度
    axes[0, 0].plot(times, results['gzip_complexity'], 'r-', linewidth=1.5)
    axes[0, 0].set_title('gzip 复杂度（原始 KC 代理量）')
    axes[0, 0].set_ylabel('压缩后字节数')
    axes[0, 0].grid(True, alpha=0.3)

    # 粗粒化 KC
    for bs, vals in results['coarse_kc'].items():
        if len(vals) > 0 and max(vals) > 0:
            axes[0, 1].plot(times, vals, linewidth=1.5, label=f"块大小={bs}")
    axes[0, 1].set_title('粗粒化 KC（Carroll，评论 #6）')
    axes[0, 1].set_ylabel('压缩后字节数')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 两段式编码
    part1 = [p[0] for p in results['two_part']]
    part2 = [p[1] for p in results['two_part']]
    axes[1, 0].plot(times, part1, 'g-', linewidth=1.5,
                    label='第一部分（模型 = 精致复杂度代理量）')
    axes[1, 0].plot(times, part2, 'm--', linewidth=1.5,
                    label='第二部分（残差 = 随机性代理量）')
    axes[1, 0].set_title('两段式编码（精致复杂度代理量）')
    axes[1, 0].set_xlabel('交换尝试总次数')
    axes[1, 0].set_ylabel('压缩后字节数')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 边界比例
    axes[1, 1].plot(times, results['fraction_mixed'], 'b-', linewidth=1.5)
    axes[1, 1].set_title('边界比例（界面长度）')
    axes[1, 1].set_xlabel('交换尝试总次数')
    axes[1, 1].set_ylabel('比例')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        '复杂性度量比较\n'
        '检验各代理量是否在中间时刻出现隆起',
        fontsize=14
    )
    plt.tight_layout()
    outpath = f"{output_dir}/plot4_measures_comparison.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  已保存：{outpath}")
    plt.close()


def plot_multiscale_heatmap(results, output_dir='.'):
    """图 5：多尺度复杂性热图（Trevisan 的想法，评论 #17），展示尺度空间中的复杂性隆起位置。"""
    times = results['times']
    grid_size = results['grid_size']

    # 为存储的网格重新计算多尺度 KC
    max_power = int(np.log2(grid_size)) - 1
    scales = [2**k for k in range(1, max_power + 1)]

    matrix = np.zeros((len(scales), len(results['grids'])))
    for t_idx, grid in enumerate(results['grids']):
        ms = multiscale_kc(grid, scales)
        for s_idx, s in enumerate(scales):
            matrix[s_idx, t_idx] = ms.get(s, 0)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        matrix, aspect='auto', cmap='hot', interpolation='bilinear',
        extent=[times[0], times[-1], len(scales) - 0.5, -0.5]
    )
    ax.set_yticks(range(len(scales)))
    ax.set_yticklabels([str(s) for s in scales])
    ax.set_xlabel('交换尝试总次数', fontsize=12)
    ax.set_ylabel('粗粒化尺度', fontsize=12)
    ax.set_title(
        '多尺度复杂性热图（Trevisan，评论 #17）\n'
        '越亮复杂性越高，检验中间尺度和时刻的隆起。',
        fontsize=13
    )
    plt.colorbar(im, label='压缩后字节数')

    plt.tight_layout()
    outpath = f"{output_dir}/plot5_multiscale_heatmap.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  已保存：{outpath}")
    plt.close()


def plot_sophistication_trajectory(results, output_dir='.'):
    """图 6：精致复杂度代理量（两段式编码第一部分）的轨迹。原说明称它最接近复杂熵；此教学代理量未经严格等价性验证。"""
    times = results['times']
    part1 = [p[0] for p in results['two_part']]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(times, part1, 'g-', linewidth=2)
    peak_idx = np.argmax(part1)
    ax.plot(times[peak_idx], part1[peak_idx], 'ko', markersize=10)
    ax.annotate(
        f'精致复杂度代理量峰值\n交换尝试次数：{times[peak_idx]:,}',
        xy=(times[peak_idx], part1[peak_idx]),
        xytext=(times[peak_idx] * 1.1, part1[peak_idx] * 0.95),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='black'),
    )

    ax.set_xlabel('交换尝试总次数', fontsize=12)
    ax.set_ylabel('第一部分大小（字节）= 精致复杂度代理量', fontsize=12)
    ax.set_title(
        '精致复杂度代理量随时间变化\n'
        '两段式编码第一部分：模型复杂度（不等同于复杂熵）',
        fontsize=13
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = f"{output_dir}/plot6_sophistication.png"
    plt.savefig(outpath, dpi=150)
    print(f"  已保存：{outpath}")
    plt.close()


def plot_animation_frames(results, output_dir='.'):
    """图 7：在一张图中展示多个时刻的网格，呈现完整混合过程。"""
    grids = results['grids']
    times = results['times']
    n = len(grids)

    # 选取约 20 帧
    n_frames = min(20, n)
    indices = np.linspace(0, n - 1, n_frames, dtype=int)

    cols = 5
    rows = (n_frames + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    axes_flat = axes.flatten()

    for i, idx in enumerate(indices):
        axes_flat[i].imshow(grids[idx], cmap='gray', interpolation='nearest')
        axes_flat[i].set_title(f"t={times[idx]:,}", fontsize=8)
        axes_flat[i].axis('off')

    # 隐藏未使用的坐标轴
    for i in range(len(indices), len(axes_flat)):
        axes_flat[i].axis('off')

    fig.suptitle(
        '咖啡混合过程\n'
        '从分离到复杂卷须纹理，再到粗粒度下均匀混合',
        fontsize=13
    )
    plt.tight_layout()
    outpath = f"{output_dir}/plot7_progression.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  已保存：{outpath}")
    plt.close()


def generate_all_plots(args):
    """生成全部 7 张图。"""
    print("=" * 60)
    print('生成可视化图表')
    print('Aaronson《复杂动力学第一定律》（2011）')
    print("=" * 60)
    print()

    print('运行模拟……')
    results = run_simulation(
        grid_size=args.grid_size,
        n_steps=args.steps,
        swaps_per_step=args.swaps_per_step,
        measure_interval=args.measure_interval,
        seed=args.seed,
    )
    print(f"  网格：{args.grid_size}x{args.grid_size}")
    print(f"  交换尝试总次数：{args.steps * args.swaps_per_step:,}")
    print(f"  测量次数：{len(results['times'])}")
    print()

    print('生成图表……')
    plot_complexity_hump(results, args.output_dir)
    plot_entropy_vs_complexity(results, args.output_dir)
    plot_grid_snapshots(results, args.output_dir)
    plot_measures_comparison(results, args.output_dir)
    plot_multiscale_heatmap(results, args.output_dir)
    plot_sophistication_trajectory(results, args.output_dir)
    plot_animation_frames(results, args.output_dir)

    print()
    print('全部 7 张图已生成。')
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='咖啡混合复杂性可视化工具')
    parser.add_argument('--grid-size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--swaps-per-step', type=int, default=50)
    parser.add_argument('--measure-interval', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='.')
    args = parser.parse_args()

    generate_all_plots(args)
