"""解答 3：多尺度复杂性分析"""

import numpy as np
import gzip
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from implementation import create_initial_grid, batch_swap


def gzip_size(data: bytes) -> int:
    return len(gzip.compress(data, compresslevel=9))


def coarse_grain(grid: np.ndarray, block_size: int) -> np.ndarray:
    """对 block_size x block_size 块内的网格值求平均。"""
    h, w = grid.shape
    h_new = h // block_size
    w_new = w // block_size
    result = np.zeros((h_new, w_new))
    for i in range(h_new):
        for j in range(w_new):
            block = grid[i * block_size:(i + 1) * block_size,
                        j * block_size:(j + 1) * block_size]
            result[i, j] = block.mean()
    return result


def coarse_grained_kc(grid: np.ndarray, block_size: int) -> int:
    """粗粒化网格的 KC 近似。"""
    coarse = coarse_grain(grid, block_size)
    quantized = (coarse * 255).astype(np.uint8)
    return gzip_size(quantized.tobytes())


def multiscale_analysis(grid: np.ndarray) -> dict:
    """计算多个尺度的 KC 近似。"""
    grid_size = min(grid.shape)
    scales = [2**k for k in range(1, int(np.log2(grid_size)))]
    result = {}
    for s in scales:
        if grid_size // s >= 2:
            result[s] = coarse_grained_kc(grid, s)
    return result


def run_multiscale_experiment(grid_size=64, n_steps=20000, swaps_per_step=50):
    """运行咖啡模拟并跟踪多尺度 KC。"""
    np.random.seed(42)
    grid = create_initial_grid(grid_size)

    max_power = int(np.log2(grid_size)) - 1
    scales = [2**k for k in range(1, max_power + 1)]
    scale_data = {s: [] for s in scales}
    times = []

    measure_interval = max(1, n_steps // 50)

    # 初始测量
    times.append(0)
    ms = multiscale_analysis(grid)
    for s in scales:
        scale_data[s].append(ms.get(s, 0))

    for step in range(1, n_steps + 1):
        batch_swap(grid, n_swaps=swaps_per_step)
        if step % measure_interval == 0:
            times.append(step * swaps_per_step)
            ms = multiscale_analysis(grid)
            for s in scales:
                scale_data[s].append(ms.get(s, 0))

    return times, scale_data


if __name__ == "__main__":
    print('解答 3：多尺度复杂性分析')
    print("=" * 40)
    print()

    print('运行实验……')
    times, scale_data = run_multiscale_experiment()
    print(f"  时间点数量：{len(times)}")
    print(f"  尺度：{sorted(scale_data.keys())}")

    for scale, values in sorted(scale_data.items()):
        peak_idx = np.argmax(values)
        print(f"  尺度 {scale}：峰值位于 t={times[peak_idx]:,}, "
              f"val={values[peak_idx]}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for scale, values in sorted(scale_data.items()):
            axes[0].plot(times, values, label=f"尺度={scale}")
        axes[0].set_xlabel('交换尝试总次数')
        axes[0].set_ylabel('压缩后字节数')
        axes[0].set_title('不同尺度的 KC')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        scales_sorted = sorted(scale_data.keys())
        matrix = np.array([scale_data[s] for s in scales_sorted])
        im = axes[1].imshow(matrix, aspect='auto', cmap='hot',
                           interpolation='bilinear')
        axes[1].set_yticks(range(len(scales_sorted)))
        axes[1].set_yticklabels([str(s) for s in scales_sorted])
        axes[1].set_title('多尺度热图（Trevisan）')
        axes[1].set_xlabel('时间索引')
        axes[1].set_ylabel('尺度')
        plt.colorbar(im, ax=axes[1])

        plt.tight_layout()
        plt.savefig("multiscale_analysis.png", dpi=100)
        print('已保存：multiscale_analysis.png')
    except ImportError:
        print('（matplotlib 不可用）')
