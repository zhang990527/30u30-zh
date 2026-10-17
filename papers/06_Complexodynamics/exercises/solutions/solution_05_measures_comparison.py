"""解答 5：比较复杂性度量"""

import numpy as np
import gzip
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from implementation import (
    create_initial_grid, batch_swap, gzip_complexity,
    coarse_grained_kc, two_part_code, grid_entropy, boundary_fraction,
)


def normalize_to_01(values):
    arr = np.array(values, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def run_comparison(grid_size=64, n_steps=30000, swaps_per_step=50,
                   measure_interval=300):
    """运行模拟并收集全部复杂性度量。"""
    np.random.seed(42)
    grid = create_initial_grid(grid_size)

    data = {
        'times': [],
        'gzip': [],
        'coarse_kc_4': [],
        'coarse_kc_8': [],
        'part1': [],
        'boundary': [],
        'entropy': [],
    }

    def measure(step):
        data['times'].append(step * swaps_per_step)
        data['gzip'].append(gzip_complexity(grid))
        data['coarse_kc_4'].append(coarse_grained_kc(grid, 4))
        data['coarse_kc_8'].append(coarse_grained_kc(grid, 8))
        p1, _ = two_part_code(grid, block_size=4)
        data['part1'].append(p1)
        data['boundary'].append(boundary_fraction(grid))
        data['entropy'].append(grid_entropy(grid))

    measure(0)
    for step in range(1, n_steps + 1):
        batch_swap(grid, n_swaps=swaps_per_step)
        if step % measure_interval == 0:
            measure(step)
            if step % (measure_interval * 10) == 0:
                print(f"  步数 {step}/{n_steps}")

    return data


def analyze_results(data):
    """输出各度量的分析。"""
    measures = {
        'gzip（KC 代理量）': data['gzip'],
        '粗粒化 KC（bs=4）': data['coarse_kc_4'],
        '粗粒化 KC（bs=8）': data['coarse_kc_8'],
        '精致复杂度（第一部分）': data['part1'],
        '边界比例': data['boundary'],
        'entropy': data['entropy'],
    }

    times = data['times']

    for name, values in measures.items():
        arr = np.array(values)
        peak_idx = np.argmax(arr)
        is_monotone = all(arr[i] <= arr[i+1] + 0.01 for i in range(len(arr)-1))

        initial = arr[0]
        peak = arr[peak_idx]
        final = arr[-1]

        print(f"  {name}:")
        print(f"    初值={initial:.1f}，峰值={peak:.1f}（t={times[peak_idx]:,}), "
              f"Final={final:.1f}")
        if is_monotone:
            print(f"    --> 单调或端点最大（无内部隆起；并非严格单调性检验）")
        elif peak > initial and peak > final:
            ratio = peak / max(initial, 0.001)
            print(f"    --> 检测到隆起（峰值/初值 = {ratio:.2f}x)")
        else:
            print(f"    --> 模式不明确")
        print()


def plot_comparison(data):
    """将全部归一化度量绘制在同一张图上。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('（matplotlib 不可用）')
        return

    times = data['times']

    measures = {
        'gzip (KC)': ('r', data['gzip']),
        '粗粒化 KC（bs=4）': ('orange', data['coarse_kc_4']),
        'sophistication': ('g', data['part1']),
        'boundary': ('b', data['boundary']),
        'entropy': ('purple', data['entropy']),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 原始值
    for name, (color, values) in measures.items():
        axes[0].plot(times, values, color=color, linewidth=1.5, label=name)
    axes[0].set_title('原始值')
    axes[0].set_xlabel('交换尝试总次数')
    axes[0].set_ylabel('数值')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 归一化
    for name, (color, values) in measures.items():
        normed = normalize_to_01(values)
        axes[1].plot(times, normed, color=color, linewidth=1.5, label=name)
    axes[1].set_title('归一化 [0, 1]')
    axes[1].set_xlabel('交换尝试总次数')
    axes[1].set_ylabel('归一化数值')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        '全部复杂性度量比较\n'
        '全局熵守恒；其他代理量是否隆起需实测',
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig("measures_comparison.png", dpi=150)
    print('已保存：measures_comparison.png')
    plt.close()


if __name__ == "__main__":
    print('解答 5：比较复杂性度量')
    print("=" * 40)
    print()

    print('运行模拟并计算全部度量……')
    data = run_comparison()
    print()

    print('分析：')
    analyze_results(data)

    print('绘制度量比较图……')
    plot_comparison(data)
    print()

    print('原脚本预设结论（请结合实际结果核对）：')
    print('- 全局熵守恒，无隆起；局部熵不保证逐步单调')
    print('- 原说明称 gzip、粗粒化 KC、精致复杂度代理量均呈隆起；不保证成立')
    print('- 原说明认为精致复杂度代理量（第一部分）最接近复杂熵；未经严格验证')
    print('- 原说明称边界比例也有峰值；它是几何度量，')
    print('  不是信息论度量，后期也未必下降')
