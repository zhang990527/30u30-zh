"""练习 4：调整后的粗粒化（第 6 节）

论文发现，3 分箱阈值化（第 5 节）产生伪影：
边界像素在不同分箱值间波动，产生虚假复杂度，
尤其是在无相互作用模型中。

第 6 节的修正方法：
  1. 增加分箱数（由 3 改为 7）
  2. 行众数调整：将接近该行众数的格点归并到众数

任务：
  1. 实现行众数调整算法
  2. 用 3 分箱（不调整）和 7 分箱（调整）运行无相互作用模型
  3. 展示调整后的方法能消除无相互作用模型的虚假复杂度，
     同时保留有相互作用模型的复杂度

论文参考：第 6 节"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import (
    CoffeeAutomaton, coarse_grain, threshold_array, gzip_size, grid_to_bytes
)


def row_majority_adjust(thresholded: np.ndarray) -> np.ndarray:
    """TODO：实现第 6.1 节的行众数调整。

对每行：
  1. 找到众数（最常见的值）
  2. 对每个格点：若与众数相差不超过 1 个分箱，则设为众数

“若某格点的值与该行众数相差不超过一个阈值间隔，
就将其调整为众数。”（第 6.1 节）

返回调整后的数组，不要修改输入。"""
    # 在此编写代码
    pass


def measure_with_method(grid, grain_size, num_buckets, use_adjustment):
    """TODO：用指定方法测量复杂度。

1. 对网格粗粒化
2. 按阈值分为 num_buckets 箱
3. 若 use_adjustment 为真，应用 row_majority_adjust
4. 压缩并返回大小"""
    # 在此编写代码
    pass


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

            if c3 is not None:
                cx_3bucket.append(c3)
                cx_7adjusted.append(c7)

        if cx_3bucket:
            print(f"  3 分箱峰值：{max(cx_3bucket)}")
            print(f"  调整后的 7 分箱峰值：{max(cx_7adjusted)}")

    # 预期：调整后的无相互作用 7 分箱结果应显著低于
    # 无相互作用 3 分箱结果（伪影已消除）
    print("\n若无相互作用模型调整后的 7 分箱结果远低于 3 分箱，")
    print("说明已成功消除阈值化伪影（第 6 节）。")
