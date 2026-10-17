"""train_minimal.py —— 咖啡混合模拟命令行界面

运行本章根据 Aaronson《复杂动力学第一定律》博文编写的咖啡—牛奶混合模拟，绘制预期的复杂性隆起；是否出现峰值需实测。
用法：
    # 基本运行
    python train_minimal.py --grid-size 64 --steps 50000
    # 比较复杂性度量
    python train_minimal.py --grid-size 64 --steps 50000 --compare-measures
    # 多尺度分析（Trevisan 的想法）
    python train_minimal.py --grid-size 64 --steps 50000 --multiscale
    # 快速测试
    python train_minimal.py --grid-size 32 --steps 5000
参考：https://scottaaronson.blog/?p=762"""

import argparse
import sys
import time
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print('[NOTE] 未安装 matplotlib，将跳过绘图。')
    print('       安装命令：pip install matplotlib')

from implementation import (
    create_initial_grid,
    batch_swap,
    gzip_complexity,
    coarse_grained_kc,
    two_part_code,
    grid_entropy,
    mean_local_entropy,
    boundary_fraction,
    multiscale_kc,
    robust_gzip_complexity,
    sophistication_proxy,
    summarize_results,
    run_simulation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='咖啡混合模拟：复杂动力学第一定律'
    )
    parser.add_argument(
        '--grid-size', type=int, default=64,
        help='方形网格边长（默认：64）'
    )
    parser.add_argument(
        '--steps', type=int, default=50000,
        help='批量交换步数（默认：50000）'
    )
    parser.add_argument(
        '--swaps-per-step', type=int, default=50,
        help='每个批量步内的交换尝试次数（默认：50）'
    )
    parser.add_argument(
        '--measure-interval', type=int, default=500,
        help='复杂性测量间隔步数（默认：500）'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子（默认：42）'
    )
    parser.add_argument(
        '--compare-measures', action='store_true',
        help='在同一次模拟中比较全部复杂性度量'
    )
    parser.add_argument(
        '--multiscale', action='store_true',
        help='运行多尺度分析（Trevisan，博文评论 #17）'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='跳过绘图（仅输出结果）'
    )
    parser.add_argument(
        '--output-dir', type=str, default='.',
        help='输出文件目录（默认：当前目录）'
    )
    return parser.parse_args()


def run_basic(args):
    """运行基本模拟并绘制复杂性曲线。"""
    print(f"运行咖啡混合模拟……")
    print(f"  网格：{args.grid_size}x{args.grid_size}")
    print(f"  步数：{args.steps:,} (x{args.swaps_per_step} 次交换尝试/步）")
    print(f"  交换尝试总次数：{args.steps * args.swaps_per_step:,}")
    print()

    start_time = time.time()

    results = run_simulation(
        grid_size=args.grid_size,
        n_steps=args.steps,
        swaps_per_step=args.swaps_per_step,
        measure_interval=args.measure_interval,
        seed=args.seed,
    )

    elapsed = time.time() - start_time
    print(f"模拟完成，用时 {elapsed:.1f}s")
    print()
    print(summarize_results(results))

    if HAS_MATPLOTLIB and not args.no_plot:
        plot_basic_results(results, args.output_dir)

    return results


def plot_basic_results(results, output_dir='.'):
    """绘制熵与复杂性的比较（核心演示）。"""
    times = results['times']
    entropy = results['entropy']
    gzip_c = results['gzip_complexity']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. 熵（通过局部熵度量跟踪增长趋势，不保证逐步单调）
    axes[0].plot(times, entropy, 'b-', linewidth=1.5)
    axes[0].set_title('全局熵（交换中守恒）')
    axes[0].set_xlabel('交换尝试总次数')
    axes[0].set_ylabel('香农熵（比特）')
    axes[0].grid(True, alpha=0.3)

    # 2. 复杂性（预期隆起）
    axes[1].plot(times, gzip_c, 'r-', linewidth=1.5)
    peak_idx = np.argmax(gzip_c)
    axes[1].axvline(x=times[peak_idx], color='gray', linestyle='--', alpha=0.5)
    axes[1].plot(times[peak_idx], gzip_c[peak_idx], 'ko', markersize=8)
    axes[1].set_title('复杂性（gzip 大小）——检验隆起')
    axes[1].set_xlabel('交换尝试总次数')
    axes[1].set_ylabel('压缩大小（字节）')
    axes[1].grid(True, alpha=0.3)

    # 3. 网格快照
    n_grids = len(results['grids'])
    snapshot_indices = [0, n_grids // 4, n_grids // 2, -1]
    snapshot_labels = ['t=0\n（分离）', '早期\n（形成卷须纹理）',
                       '中期\n（复杂）', '后期\n（混合）']
    for i, (idx, label) in enumerate(zip(snapshot_indices, snapshot_labels)):
        ax_inset = fig.add_axes([0.68 + i * 0.08, 0.55, 0.07, 0.35])
        ax_inset.imshow(results['grids'][idx], cmap='gray', interpolation='nearest')
        ax_inset.set_title(label, fontsize=7)
        ax_inset.axis('off')

    axes[2].axis('off')
    axes[2].set_title('网格快照（见插图）')

    plt.tight_layout()
    outpath = f"{output_dir}/complexity_hump.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\n图已保存至 {outpath}")
    plt.close()


def run_compare_measures(args):
    """在同一次模拟中比较全部复杂性度量。"""
    print('运行模拟并测量全部复杂性指标……')
    print()

    results = run_simulation(
        grid_size=args.grid_size,
        n_steps=args.steps,
        swaps_per_step=args.swaps_per_step,
        measure_interval=args.measure_interval,
        seed=args.seed,
    )

    print(summarize_results(results))

    if HAS_MATPLOTLIB and not args.no_plot:
        plot_compare_measures(results, args.output_dir)

    return results


def plot_compare_measures(results, output_dir='.'):
    """并排绘制全部复杂性度量。"""
    times = results['times']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. gzip 复杂度
    axes[0, 0].plot(times, results['gzip_complexity'], 'r-', linewidth=1.5)
    axes[0, 0].set_title('gzip 复杂度（原始 KC 代理量）')
    axes[0, 0].set_xlabel('交换尝试总次数')
    axes[0, 0].set_ylabel('压缩后字节数')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 各尺度粗粒化 KC
    for bs, vals in results['coarse_kc'].items():
        if len(vals) > 0 and max(vals) > 0:
            axes[0, 1].plot(times, vals, linewidth=1.5, label=f"块大小={bs}")
    axes[0, 1].set_title('粗粒化 KC（Carroll 的度量）')
    axes[0, 1].set_xlabel('交换尝试总次数')
    axes[0, 1].set_ylabel('压缩后字节数')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 两段式编码
    part1 = [p[0] for p in results['two_part']]
    part2 = [p[1] for p in results['two_part']]
    axes[1, 0].plot(times, part1, 'g-', linewidth=1.5, label='第一部分（模型/精致复杂度）')
    axes[1, 0].plot(times, part2, 'm-', linewidth=1.5, label='第二部分（残差）')
    axes[1, 0].plot(times, [p1 + p2 for p1, p2 in zip(part1, part2)],
                    'k--', linewidth=1, label='总计')
    axes[1, 0].set_title('两段式编码（精致复杂度代理量）')
    axes[1, 0].set_xlabel('交换尝试总次数')
    axes[1, 0].set_ylabel('压缩后字节数')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 边界比例
    axes[1, 1].plot(times, results['fraction_mixed'], 'b-', linewidth=1.5)
    axes[1, 1].set_title('边界比例（界面长度）')
    axes[1, 1].set_xlabel('交换尝试总次数')
    axes[1, 1].set_ylabel('异色相邻像素对比例')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        '复杂性度量比较\n'
        'Aaronson《复杂动力学第一定律》（2011）',
        fontsize=14
    )
    plt.tight_layout()
    outpath = f"{output_dir}/measures_comparison.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\n图已保存至 {outpath}")
    plt.close()


def run_multiscale(args):
    """按 Trevisan 的思路运行多尺度分析（博文评论 #17）。"""
    print('运行多尺度复杂性分析（Trevisan 的想法）……')
    print()

    np.random.seed(args.seed)
    grid = create_initial_grid(args.grid_size)

    # 待分析尺度
    max_power = int(np.log2(args.grid_size)) - 1
    scales = [2**k for k in range(1, max_power + 1)]

    # 收集数据
    n_measurements = min(100, args.steps // max(args.measure_interval, 1))
    step_interval = args.steps // n_measurements

    time_points = []
    scale_time_kc = {s: [] for s in scales}

    # 初始测量
    time_points.append(0)
    ms = multiscale_kc(grid, scales)
    for s in scales:
        scale_time_kc[s].append(ms.get(s, 0))

    for i in range(1, n_measurements + 1):
        for _ in range(step_interval):
            batch_swap(grid, n_swaps=args.swaps_per_step)

        time_points.append(i * step_interval * args.swaps_per_step)
        ms = multiscale_kc(grid, scales)
        for s in scales:
            scale_time_kc[s].append(ms.get(s, 0))

        if i % 10 == 0:
            print(f"  进度：{i}/{n_measurements}")

    print(f"\n多尺度分析完成。")

    if HAS_MATPLOTLIB and not args.no_plot:
        plot_multiscale(time_points, scale_time_kc, scales, args.output_dir)


def plot_multiscale(time_points, scale_time_kc, scales, output_dir='.'):
    """绘制多尺度复杂性热图。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 折线：各尺度 KC 随时间的变化
    for s in scales:
        axes[0].plot(time_points, scale_time_kc[s], linewidth=1.5, label=f"尺度={s}")
    axes[0].set_title('各粗粒化尺度的 KC')
    axes[0].set_xlabel('交换尝试总次数')
    axes[0].set_ylabel('压缩后字节数')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 热图：尺度 x 时间
    matrix = np.array([scale_time_kc[s] for s in scales])
    im = axes[1].imshow(
        matrix, aspect='auto', cmap='hot', interpolation='nearest',
        extent=[time_points[0], time_points[-1], len(scales) - 0.5, -0.5]
    )
    axes[1].set_yticks(range(len(scales)))
    axes[1].set_yticklabels([str(s) for s in scales])
    axes[1].set_title('多尺度复杂性热图（Trevisan）')
    axes[1].set_xlabel('交换尝试总次数')
    axes[1].set_ylabel('粗粒化尺度')
    plt.colorbar(im, ax=axes[1], label='压缩后字节数')

    fig.suptitle(
        '咖啡混合复杂性的多尺度分析\n'
        'Trevisan 的建议（博文评论 #17）',
        fontsize=13
    )
    plt.tight_layout()
    outpath = f"{output_dir}/multiscale_analysis.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\n图已保存至 {outpath}")
    plt.close()


if __name__ == "__main__":
    args = parse_args()

    if args.compare_measures:
        run_compare_measures(args)
    elif args.multiscale:
        run_multiscale(args)
    else:
        run_basic(args)
