"""解答 4：调整后的粗粒化（第 6 节）"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import (
    CoffeeAutomaton, coarse_grain, threshold_array, gzip_size, grid_to_bytes
)


def row_majority_adjust(thresholded: np.ndarray) -> np.ndarray:
    """第 6.1 节的行众数调整。
将与该行众数相差不超过 1 个分箱的格点归并到众数。"""
    adjusted = thresholded.copy()
    for i in range(adjusted.shape[0]):
        row = adjusted[i]
        values, counts = np.unique(row, return_counts=True)
        majority = values[np.argmax(counts)]
        mask = np.abs(row.astype(np.int16) - int(majority)) <= 1
        adjusted[i, mask] = majority
    return adjusted


def measure_with_method(grid, grain_size, num_buckets, use_adjustment):
    """使用指定的粗粒化方法测量复杂度。"""
    coarse = coarse_grain(grid, grain_size)
    discrete = threshold_array(coarse, num_buckets)
    if use_adjustment:
        discrete = row_majority_adjust(discrete)
    return gzip_size(grid_to_bytes(discrete))


if __name__ == '__main__':
    print("调整后的粗粒化实验（第 6 节）")
    print("=" * 50)

    grid_size = 50
    total_steps = 50000
    num_snapshots = 50

    for model in ['interacting', 'non_interacting']:
        print(f"\n{model.replace('_', '-').title()} 模型：")

        ca = CoffeeAutomaton(grid_size, model, seed=42)
        steps_per = total_steps // num_snapshots

        cx_3bucket = []
        cx_7adjusted = []

        for i in range(num_snapshots):
            ca.step(steps_per)
            grid = ca.get_binary_grid()

            c3 = measure_with_method(grid, grain_size=5, num_buckets=3, use_adjustment=False)
            c7 = measure_with_method(grid, grain_size=5, num_buckets=7, use_adjustment=True)

            cx_3bucket.append(c3)
            cx_7adjusted.append(c7)

        print(f"  3 分箱峰值：{max(cx_3bucket)}")
        print(f"  调整后的 7 分箱峰值：{max(cx_7adjusted)}")

    print("\n无相互作用模型调整后的 7 分箱结果应显著更低，说明伪影已消除。")
