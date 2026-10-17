"""练习 1：粗粒化与复杂度测量

从零构建论文的核心测量工具。

论文第 5.1 节通过以下步骤测量“表观复杂度”：
  1. 将网格划分为 g x g 块
  2. 对每块求平均
  3. 按阈值离散分箱
  4. 用 gzip 压缩
  5. 压缩大小 = 复杂度估计值

任务：实现 coarse_grain()，并在不同时间步的咖啡元胞自动机网格上
同时测量熵与复杂度。

论文参考：第 4、5 节"""

import sys
import os
import gzip
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import CoffeeAutomaton


def coarse_grain(grid: np.ndarray, grain_size: int) -> np.ndarray:
    """TODO：实现粗粒化。

给定 N x N 二值网格和块边长 g，生成 (N//g) x (N//g) 数组，
其中每个格点为原网格中对应 g x g 块的平均值。

返回值位于 [0, 1] 的浮点数组。"""
    # 在此编写代码
    pass


def threshold(coarse: np.ndarray, num_buckets: int = 3) -> np.ndarray:
    """TODO：将浮点数按阈值离散分箱。

把 [0, 1] 的数值映射到 [0, num_buckets-1] 的整数。
3 分箱时：0 = 以咖啡为主，1 = 混合，2 = 以奶油为主。

返回 uint8 数组。"""
    # 在此编写代码
    pass


def gzip_complexity(grid: np.ndarray, grain_size: int = 10,
                    num_buckets: int = 3) -> int:
    """TODO：测量表观复杂度。

1. 对网格粗粒化
2. 按阈值分箱
3. 用 gzip 压缩
4. 返回以字节计的压缩大小"""
    # 在此编写代码
    pass


def gzip_entropy(grid: np.ndarray) -> int:
    """TODO：测量熵估计值。

直接用 gzip 压缩细粒度网格。
返回以字节计的压缩大小。"""
    # 在此编写代码
    pass


if __name__ == '__main__':
    # 测试你的实现：
    # 创建 50x50 自动机，在 t=0、t=10000、t=50000 处测量
    ca = CoffeeAutomaton(grid_size=50, model='interacting', seed=42)

    print("时间 | 熵 | 复杂度")
    print("-----|---------|----------")

    for t in [0, 10000, 50000]:
        if t > 0:
            ca.step(t - ca.time)
        grid = ca.get_binary_grid()
        e = gzip_entropy(grid)
        c = gzip_complexity(grid, grain_size=5, num_buckets=3)
        print(f"{t:>5} | {e:>7} | {c:>10}")

    # 预期行为：
    # - 熵应随时间增加
    # - 复杂度在 t=0 时应低，t=10000 时更高，t=50000 时更低
    print("\n若复杂度在中间时刻达到峰值，则实现符合预期。")
