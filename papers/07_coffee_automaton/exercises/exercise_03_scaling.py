"""练习 3：标度分析

复现论文图 6-8 的标度结果。

论文发现：
  - 最大熵 ~ n^2（与网格边长成二次关系）
  - 最大复杂度 ~ n（与网格边长成线性关系）
  - 达到最大复杂度的时间 ~ n^2（二次关系）

任务：
  1. 在多个网格大小下运行有相互作用模型
  2. 记录最大熵、最大复杂度、达到峰值的时间
  3. 拟合曲线并检查标度指数
  4. 绘制结果

论文参考：第 5.2 节，图 6-8"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import run_simulation


def scaling_experiment(sizes=None, steps_multiplier=40):
    """TODO：在多个网格大小下模拟，收集标度数据。

对每个网格边长 n：
  - 运行 total_steps = steps_multiplier * n^2 步，使较大网格有更多时间
  - 记录最大熵、最大复杂度、达到最大复杂度的时间

参数：
    sizes：要测试的网格边长列表，如 [10, 20, 30, 50]
    steps_multiplier：步数 = multiplier * n^2

返回：
    sizes：网格边长数组
    max_entropy：最大熵数组
    max_complexity：最大复杂度数组
    peak_time：达到复杂度峰值的时间数组"""
    if sizes is None:
        sizes = [10, 20, 30, 50]

    # 在此编写代码
    pass


def fit_and_plot(sizes, max_entropy, max_complexity, peak_time):
    """TODO：拟合曲线并绘制结果。

1. 对 max_entropy 与 n 拟合二次曲线（预期 ~ n^2）
2. 对 max_complexity 与 n 拟合直线（预期 ~ n）
3. 对 peak_time 与 n 拟合二次曲线（预期 ~ n^2）
4. 绘制三者及拟合曲线和 r^2 值"""
    import matplotlib.pyplot as plt

    # 在此编写代码
    # 提示：np.polyfit(sizes, values, degree) 返回系数
    #       R^2 = 1 - SS_res / SS_tot
    pass


if __name__ == '__main__':
    print("标度实验（图 6-8）")
    print("较大网格可能需要几分钟。")
    print("=" * 50)

    result = scaling_experiment(sizes=[10, 20, 30, 50])

    if result is not None:
        sizes, max_e, max_c, peak_t = result
        print(f"\n{'边长':>6} {'最大熵':>12} {'最大复杂度':>14} {'峰值时间':>10}")
        for i in range(len(sizes)):
            print(f"{sizes[i]:>6} {max_e[i]:>12} {max_c[i]:>14} {peak_t[i]:>10}")

        fit_and_plot(sizes, max_e, max_c, peak_t)
