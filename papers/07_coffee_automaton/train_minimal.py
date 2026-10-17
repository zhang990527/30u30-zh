"""train_minimal.py——从命令行运行咖啡元胞自动机实验。

用法：
    # 基本运行：有相互作用模型，展示熵与复杂度随时间的变化
    python train_minimal.py --grid-size 50 --steps 100000

    # 比较有相互作用与无相互作用模型（复现图 2 和图 10）
    python train_minimal.py --grid-size 50 --steps 100000 --compare-models

    # 调整后的粗粒化（第 6 节）与基本方法（第 5 节）比较
    python train_minimal.py --grid-size 50 --steps 100000 --compare-methods

    # 标度实验（复现图 6-8）
    python train_minimal.py --scaling --sizes 10 20 30 50

    # 仅运行无相互作用模型
    python train_minimal.py --grid-size 50 --steps 50000 --model non_interacting

论文：Aaronson、Carroll、Ouellette（2014），arXiv:1405.6903"""

import argparse
import sys
import os
import numpy as np

# 将上级目录加入路径，以便导入 implementation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from implementation import (
    CoffeeAutomaton, run_simulation, compare_models,
    scaling_experiment, measure_entropy, measure_complexity
)


def plot_single_run(results, model_name='interacting', save_path=None):
    """绘制单次模拟中熵与复杂度随时间变化的曲线。"""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 在同一时间轴上绘制熵与复杂度
    ax1.plot(results['times'], results['entropy'], 'b-', label='熵（细粒度状态的 gzip 大小）')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('压缩大小（字节）', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title(f'{model_name.replace("_", "-").title()} 模型')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(results['times'], results['complexity'], 'r-', label='复杂度（粗粒化状态的 gzip 大小）')
    ax1_twin.set_ylabel('复杂度（字节）', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    # 单独绘制复杂度并标出峰值
    ax2.plot(results['times'], results['complexity'], 'r-', linewidth=2)
    peak_idx = np.argmax(results['complexity'])
    ax2.axvline(results['times'][peak_idx], color='gray', linestyle='--', alpha=0.5)
    ax2.annotate(f'峰值位于 t={results["times"][peak_idx]}',
                 xy=(results['times'][peak_idx], results['complexity'][peak_idx]),
                 xytext=(10, 10), textcoords='offset points', fontsize=9)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('表观复杂度（字节）')
    ax2.set_title('复杂度的兴衰')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图已保存至 {save_path}")
    plt.show()


def plot_model_comparison(interacting, non_interacting, save_path=None):
    """比较有相互作用与无相互作用模型，复现图 2／图 10。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 有相互作用模型的熵
    axes[0, 0].plot(interacting['times'], interacting['entropy'], 'b-')
    axes[0, 0].set_title('有相互作用：熵')
    axes[0, 0].set_xlabel('时间步')
    axes[0, 0].set_ylabel('gzip 大小（字节）')

    # 有相互作用模型的复杂度
    axes[0, 1].plot(interacting['times'], interacting['complexity'], 'r-')
    axes[0, 1].set_title('有相互作用：复杂度')
    axes[0, 1].set_xlabel('时间步')
    axes[0, 1].set_ylabel('gzip 大小（字节）')

    # 无相互作用模型的熵
    axes[1, 0].plot(non_interacting['times'], non_interacting['entropy'], 'b-')
    axes[1, 0].set_title('无相互作用：熵')
    axes[1, 0].set_xlabel('时间步')
    axes[1, 0].set_ylabel('gzip 大小（字节）')

    # 无相互作用模型的复杂度
    axes[1, 1].plot(non_interacting['times'], non_interacting['complexity'], 'r-')
    axes[1, 1].set_title('无相互作用：复杂度')
    axes[1, 1].set_xlabel('时间步')
    axes[1, 1].set_ylabel('gzip 大小（字节）')

    plt.suptitle('咖啡元胞自动机：有相互作用与无相互作用\n'
                 '(Aaronson, Carroll, Ouellette 2014 — Figures 2/10)',
                 fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图已保存至 {save_path}")
    plt.show()


def plot_method_comparison(results_3bucket, results_7bucket, save_path=None):
    """比较第 5 节（3 分箱）与第 6 节（调整后的 7 分箱）。"""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(results_3bucket['times'], results_3bucket['complexity'],
             'r-', label='3 分箱（第 5 节）')
    ax1.plot(results_7bucket['times'], results_7bucket['complexity'],
             'b-', label='调整后的 7 分箱（第 6 节）')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('表观复杂度（字节）')
    ax1.set_title('有相互作用模型')
    ax1.legend()

    # 若有无相互作用模型的数据……
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('表观复杂度（字节）')
    ax2.set_title('无相互作用模型')
    ax2.text(0.5, 0.5, '运行 --compare-models\n获取无相互作用模型数据',
             ha='center', va='center', transform=ax2.transAxes, fontsize=10, color='gray')

    plt.suptitle('第 5 节与第 6 节粗粒化方法的比较')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_scaling(scaling_results, save_path=None):
    """绘制标度结果，复现图 6-8。"""
    import matplotlib.pyplot as plt

    sizes = scaling_results['sizes']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 最大熵与 n 的关系（应为二次关系 ~ n^2）
    axes[0].plot(sizes, scaling_results['max_entropy'], 'bo-')
    # 二次拟合
    coeffs = np.polyfit(sizes, scaling_results['max_entropy'], 2)
    x_fit = np.linspace(sizes[0], sizes[-1], 50)
    axes[0].plot(x_fit, np.polyval(coeffs, x_fit), 'b--', alpha=0.5,
                 label=f'二次拟合')
    axes[0].set_xlabel('网格边长（n）')
    axes[0].set_ylabel('最大熵（字节）')
    axes[0].set_title('最大熵与网格大小\n（预期：~ n^2）')
    axes[0].legend()

    # 最大复杂度与 n 的关系（应为线性关系 ~ n）
    axes[1].plot(sizes, scaling_results['max_complexity'], 'ro-')
    # 线性拟合
    coeffs = np.polyfit(sizes, scaling_results['max_complexity'], 1)
    axes[1].plot(x_fit, np.polyval(coeffs, x_fit), 'r--', alpha=0.5,
                 label=f'线性拟合')
    axes[1].set_xlabel('网格边长（n）')
    axes[1].set_ylabel('最大复杂度（字节）')
    axes[1].set_title('最大复杂度与网格大小\n（预期：~ n）')
    axes[1].legend()

    # 达到最大复杂度的时间与 n 的关系（应为二次关系 ~ n^2）
    axes[2].plot(sizes, scaling_results['peak_time'], 'go-')
    coeffs = np.polyfit(sizes, scaling_results['peak_time'], 2)
    axes[2].plot(x_fit, np.polyval(coeffs, x_fit), 'g--', alpha=0.5,
                 label=f'二次拟合')
    axes[2].set_xlabel('网格边长（n）')
    axes[2].set_ylabel('达到复杂度峰值的时间')
    axes[2].set_title('达到峰值的时间与网格大小\n（预期：~ n^2）')
    axes[2].legend()

    plt.suptitle('标度分析（论文图 6-8）', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图已保存至 {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='咖啡元胞自动机——Aaronson、Carroll、Ouellette（2014）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python train_minimal.py --grid-size 50 --steps 100000
  python train_minimal.py --compare-models --grid-size 50 --steps 100000
  python train_minimal.py --scaling --sizes 10 20 30 50
        """
    )

    parser.add_argument('--grid-size', type=int, default=50,
                        help='网格边长 N（默认：50）')
    parser.add_argument('--steps', type=int, default=100000,
                        help='模拟总步数（默认：100000）')
    parser.add_argument('--snapshots', type=int, default=100,
                        help='测量点数量（默认：100）')
    parser.add_argument('--model', type=str, default='interacting',
                        choices=['interacting', 'non_interacting'],
                        help='要运行的模型（默认：interacting）')
    parser.add_argument('--buckets', type=int, default=7,
                        choices=[3, 7],
                        help='阈值分箱数：3（第 5 节）或 7（第 6 节）')
    parser.add_argument('--no-adjust', action='store_true',
                        help='禁用行众数调整，使用原始阈值化')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认：42）')

    # 运行模式选项
    parser.add_argument('--compare-models', action='store_true',
                        help='比较有相互作用与无相互作用模型')
    parser.add_argument('--compare-methods', action='store_true',
                        help='比较第 5 节（3 分箱）与第 6 节（7 分箱）')
    parser.add_argument('--scaling', action='store_true',
                        help='运行标度实验（图 6-8）')
    parser.add_argument('--sizes', type=int, nargs='+', default=None,
                        help='标度实验中的网格边长')

    parser.add_argument('--save', type=str, default=None,
                        help='将图保存到文件而不显示')
    parser.add_argument('--no-plot', action='store_true',
                        help='跳过绘图，仅打印结果')

    args = parser.parse_args()

    if args.scaling:
        # 标度实验（图 6-8）
        print("标度实验（论文图 6-8）")
        print("=" * 50)
        sizes = args.sizes or [10, 20, 30, 50]
        results = scaling_experiment(sizes=sizes, seed=args.seed)

        print("\n结果：")
        print(f"  {'边长':>6} {'最大熵':>12} {'最大复杂度':>14} {'峰值时间':>10}")
        print(f"  {'----':>6} {'-----------':>12} {'--------------':>14} {'---------':>10}")
        for i, n in enumerate(results['sizes']):
            print(f"  {n:>6} {results['max_entropy'][i]:>12} "
                  f"{results['max_complexity'][i]:>14} {results['peak_time'][i]:>10}")

        if not args.no_plot:
            plot_scaling(results, save_path=args.save)

    elif args.compare_models:
        # 有相互作用与无相互作用（图 2／10）
        print("比较有相互作用与无相互作用模型")
        print("=" * 50)
        inter, non_inter = compare_models(
            args.grid_size, args.steps, args.snapshots, args.seed)

        inter_peak = np.argmax(inter['complexity'])
        non_peak = np.argmax(non_inter['complexity'])
        print(f"\n有相互作用：峰值复杂度 = {inter['complexity'][inter_peak]} "
              f"位于 t = {inter['times'][inter_peak]}")
        print(f"无相互作用：峰值复杂度 = {non_inter['complexity'][non_peak]} "
              f"位于 t = {non_inter['times'][non_peak]}")

        if not args.no_plot:
            plot_model_comparison(inter, non_inter, save_path=args.save)

    elif args.compare_methods:
        # 第 5 节与第 6 节的方法
        print("比较粗粒化方法")
        print("=" * 50)

        print("正在运行 3 分箱阈值化（第 5 节）……")
        results_3 = run_simulation(
            args.grid_size, args.steps, args.snapshots,
            args.model, num_buckets=3, adjust=False, seed=args.seed)

        print("正在运行调整后的 7 分箱（第 6 节）……")
        results_7 = run_simulation(
            args.grid_size, args.steps, args.snapshots,
            args.model, num_buckets=7, adjust=True, seed=args.seed)

        peak_3 = np.argmax(results_3['complexity'])
        peak_7 = np.argmax(results_7['complexity'])
        print(f"\n3 分箱：峰值 = {results_3['complexity'][peak_3]}，位于 t={results_3['times'][peak_3]}")
        print(f"调整后的 7 分箱：峰值 = {results_7['complexity'][peak_7]}，位于 t={results_7['times'][peak_7]}")

        if not args.no_plot:
            plot_method_comparison(results_3, results_7, save_path=args.save)

    else:
        # 运行单个模型
        print(f"正在运行 {args.model} 模型（{args.grid_size}x{args.grid_size}，"
              f"{args.steps} 步，{args.buckets} 个分箱，"
              f"adjust={'off' if args.no_adjust else 'on'})")
        print("=" * 50)

        results = run_simulation(
            args.grid_size, args.steps, args.snapshots,
            args.model, num_buckets=args.buckets,
            adjust=not args.no_adjust, seed=args.seed)

        peak_idx = np.argmax(results['complexity'])
        print(f"\n结果：")
        print(f"  峰值复杂度：{results['complexity'][peak_idx]} 字节，"
              f"位于 t = {results['times'][peak_idx]}")
        print(f"  最终熵：{results['entropy'][-1]} 字节")
        print(f"  最终复杂度：{results['complexity'][-1]} 字节")
        print(f"  混合比例：{results['fraction_mixed'][-1]:.3f}")

        if not args.no_plot:
            plot_single_run(results, args.model, save_path=args.save)


if __name__ == '__main__':
    main()
