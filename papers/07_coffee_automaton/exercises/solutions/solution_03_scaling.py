"""解答 3：标度分析（图 6-8）"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import run_simulation


def scaling_experiment(sizes=None, steps_multiplier=40):
    """在多个网格大小下模拟，收集标度数据。"""
    if sizes is None:
        sizes = [10, 20, 30, 50]

    max_entropy = []
    max_complexity = []
    peak_time = []

    for n in sizes:
        total_steps = steps_multiplier * n * n
        print(f"  网格 {n}x{n}，{total_steps} 步……")

        results = run_simulation(n, total_steps, num_snapshots=100,
                                 model='interacting', seed=42)

        max_entropy.append(np.max(results['entropy']))
        max_complexity.append(np.max(results['complexity']))
        peak_idx = np.argmax(results['complexity'])
        peak_time.append(results['times'][peak_idx])

    return (np.array(sizes), np.array(max_entropy),
            np.array(max_complexity), np.array(peak_time))


def fit_and_plot(sizes, max_entropy, max_complexity, peak_time):
    """拟合曲线并绘制标度结果。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x_fit = np.linspace(sizes[0], sizes[-1], 50)

    # 最大熵 ~ n^2
    c = np.polyfit(sizes, max_entropy, 2)
    ss_res = np.sum((max_entropy - np.polyval(c, sizes))**2)
    ss_tot = np.sum((max_entropy - np.mean(max_entropy))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    axes[0].plot(sizes, max_entropy, 'bo-')
    axes[0].plot(x_fit, np.polyval(c, x_fit), 'b--', alpha=0.5)
    axes[0].set_xlabel('网格边长 n')
    axes[0].set_ylabel('最大熵（字节）')
    axes[0].set_title(f'最大熵 ~ n^2（r^2={r2:.4f}）')

    # 最大复杂度 ~ n
    c = np.polyfit(sizes, max_complexity, 1)
    ss_res = np.sum((max_complexity - np.polyval(c, sizes))**2)
    ss_tot = np.sum((max_complexity - np.mean(max_complexity))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    axes[1].plot(sizes, max_complexity, 'ro-')
    axes[1].plot(x_fit, np.polyval(c, x_fit), 'r--', alpha=0.5)
    axes[1].set_xlabel('网格边长 n')
    axes[1].set_ylabel('最大复杂度（字节）')
    axes[1].set_title(f'最大复杂度 ~ n（r^2={r2:.4f}）')

    # 达到峰值的时间 ~ n^2
    c = np.polyfit(sizes, peak_time, 2)
    ss_res = np.sum((peak_time - np.polyval(c, sizes))**2)
    ss_tot = np.sum((peak_time - np.mean(peak_time))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    axes[2].plot(sizes, peak_time, 'go-')
    axes[2].plot(x_fit, np.polyval(c, x_fit), 'g--', alpha=0.5)
    axes[2].set_xlabel('网格边长 n')
    axes[2].set_ylabel('达到峰值的时间')
    axes[2].set_title(f'达到峰值的时间 ~ n^2（r^2={r2:.4f}）')

    plt.suptitle('标度分析（图 6-8）')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("标度实验（图 6-8）")
    print("=" * 50)

    sizes, max_e, max_c, peak_t = scaling_experiment(sizes=[10, 20, 30, 50])

    print(f"\n{'边长':>6} {'最大熵':>12} {'最大复杂度':>14} {'峰值时间':>10}")
    for i in range(len(sizes)):
        print(f"{sizes[i]:>6} {max_e[i]:>12} {max_c[i]:>14} {peak_t[i]:>10}")

    fit_and_plot(sizes, max_e, max_c, peak_t)
