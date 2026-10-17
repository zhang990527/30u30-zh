"""解答 5：比较压缩器（图 5）"""

import sys
import os
import gzip
import bz2
import lzma
import zlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import (
    CoffeeAutomaton, coarse_grain, threshold_array,
    row_majority_adjust, grid_to_bytes
)


def compress_with_all(data: bytes) -> dict:
    """使用多种算法压缩数据。"""
    return {
        'gzip': len(gzip.compress(data, compresslevel=9)),
        'bz2': len(bz2.compress(data, compresslevel=9)),
        'lzma': len(lzma.compress(data)),
        'zlib': len(zlib.compress(data, level=9)),
    }


def run_compressor_comparison(grid_size=50, total_steps=50000, num_snapshots=60):
    """运行自动机，收集各压缩器的复杂度估计值。"""
    ca = CoffeeAutomaton(grid_size, 'interacting', seed=42)
    grain_size = max(2, grid_size // 10)
    steps_per = max(1, total_steps // num_snapshots)

    times = []
    results = {'gzip': [], 'bz2': [], 'lzma': [], 'zlib': []}

    # 初始测量
    grid = ca.get_binary_grid()
    coarse = coarse_grain(grid, grain_size)
    discrete = threshold_array(coarse, 7)
    discrete = row_majority_adjust(discrete)
    data = grid_to_bytes(discrete)

    times.append(0)
    for name, size in compress_with_all(data).items():
        results[name].append(size)

    for i in range(num_snapshots):
        ca.step(steps_per)
        times.append(ca.time)

        grid = ca.get_binary_grid()
        coarse = coarse_grain(grid, grain_size)
        discrete = threshold_array(coarse, 7)
        discrete = row_majority_adjust(discrete)
        data = grid_to_bytes(discrete)

        for name, size in compress_with_all(data).items():
            results[name].append(size)

    return times, results


def plot_comparison(times, results):
    """为所有压缩器绘制复杂度曲线。"""
    import matplotlib.pyplot as plt

    colors = {'gzip': 'blue', 'bz2': 'red', 'lzma': 'green', 'zlib': 'orange'}

    plt.figure(figsize=(10, 6))
    for name, sizes in results.items():
        plt.plot(times, sizes, color=colors.get(name, 'black'),
                 label=name, linewidth=1.5)

    plt.xlabel('时间步')
    plt.ylabel('压缩大小（字节）')
    plt.title('不同压缩器的复杂度（参见图 5）')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("压缩器比较（图 5）")
    print("=" * 50)

    times, compressor_results = run_compressor_comparison()

    print("\n各压缩器的复杂度峰值：")
    for name, sizes in compressor_results.items():
        peak = max(sizes)
        peak_t = times[sizes.index(peak)]
        print(f"  {name:>6}：{peak:>6} 字节，位于 t={peak_t}")

    print("\n所有压缩器都应呈现定性一致的先升后降模式。")

    plot_comparison(times, compressor_results)
