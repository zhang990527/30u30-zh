"""解答 4：两段式编码（精致复杂度代理量）"""

import numpy as np
import gzip
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from implementation import create_initial_grid, batch_swap, gzip_complexity


def gzip_size(data: bytes) -> int:
    return len(gzip.compress(data, compresslevel=9))


def two_part_code(grid: np.ndarray, block_size: int = 4):
    """计算两段式编码大小。"""
    h, w = grid.shape
    grid_f = grid.astype(np.float64)

    # 粗粒化
    h_new = h // block_size
    w_new = w // block_size
    coarse = np.zeros((h_new, w_new))
    for i in range(h_new):
        for j in range(w_new):
            block = grid_f[i * block_size:(i + 1) * block_size,
                          j * block_size:(j + 1) * block_size]
            coarse[i, j] = block.mean()

    # 第一部分：模型
    model_q = (coarse * 255).astype(np.uint8)
    part1 = gzip_size(model_q.tobytes())

    # 重建
    reconstructed = np.repeat(
        np.repeat(coarse, block_size, axis=0),
        block_size, axis=1
    )[:h, :w]

    # 第二部分：残差
    residual = grid_f - reconstructed
    residual_uint8 = ((residual + 1.0) * 127.5).astype(np.uint8)
    part2 = gzip_size(residual_uint8.tobytes())

    return part1, part2


def verify_on_simple_grids():
    """用已知网格类型验证精致复杂度代理量。"""
    size = 64

    grids = {
        'all-zeros': np.zeros((size, size), dtype=np.uint8),
        'all-ones': np.ones((size, size), dtype=np.uint8),
        'random': np.random.randint(0, 2, (size, size), dtype=np.uint8),
        'checkerboard': np.indices((size, size)).sum(axis=0) % 2,
    }

    for name, grid in grids.items():
        p1, p2 = two_part_code(grid.astype(np.uint8))
        print(f"  {name:15s}：第一部分={p1:4d}，第二部分={p2:4d}，总计={p1+p2:4d}")

    # 检查：简单网格的第一部分较小
    p1_zeros, _ = two_part_code(grids['all-zeros'])
    p1_random, _ = two_part_code(grids['random'])
    print()
    print(f"  [OK] 全零网格第一部分（{p1_zeros}）和随机网格第一部分（{p1_random})")
    print('       预期两者都较小（简单模型）')


def sophistication_trajectory(grid_size=64, n_steps=20000, swaps_per_step=50):
    """随时间跟踪咖啡混合中第一部分的大小。"""
    np.random.seed(42)
    grid = create_initial_grid(grid_size)

    times = [0]
    part1_vals = []
    gzip_vals = []

    p1, _ = two_part_code(grid)
    part1_vals.append(p1)
    gzip_vals.append(gzip_complexity(grid))

    measure_interval = max(1, n_steps // 50)

    for step in range(1, n_steps + 1):
        batch_swap(grid, n_swaps=swaps_per_step)
        if step % measure_interval == 0:
            times.append(step * swaps_per_step)
            p1, _ = two_part_code(grid)
            part1_vals.append(p1)
            gzip_vals.append(gzip_complexity(grid))

    return times, part1_vals, gzip_vals


if __name__ == "__main__":
    print('解答 4：两段式编码（精致复杂度代理量）')
    print("=" * 40)
    print()

    print('用简单网格验证……')
    verify_on_simple_grids()
    print()

    print('计算精致复杂度代理量轨迹……')
    times, part1_vals, gzip_vals = sophistication_trajectory()

    peak_soph = np.argmax(part1_vals)
    peak_gzip = np.argmax(gzip_vals)
    print(f"  精致复杂度代理量峰值位于 t={times[peak_soph]:,}")
    print(f"  gzip KC 峰值位于 t={times[peak_gzip]:,}")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(times, gzip_vals, 'r-', label='gzip (KC)')
        axes[0].set_title('gzip 复杂度')
        axes[0].set_xlabel('交换尝试总次数')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(times, part1_vals, 'g-', label='第一部分（精致复杂度）')
        axes[1].set_title('精致复杂度代理量')
        axes[1].set_xlabel('交换尝试总次数')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("sophistication_trajectory.png", dpi=100)
        print('已保存：sophistication_trajectory.png')
    except ImportError:
        print('（matplotlib 不可用）')
