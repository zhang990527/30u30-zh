"""练习 2：有相互作用与无相互作用模型的比较

论文的关键实证发现（第 6 节）：真正的复杂性需要相互作用。
无相互作用模型始终不会产生宏观复杂性，因为各粒子都在独立随机游走。

任务：
  1. 在 50x50 网格上运行两种模型
  2. 定期测量复杂度
  3. 为两种模型绘制复杂度随时间变化的曲线
  4. 观察：有相互作用模型先升后降，无相互作用模型保持平坦

论文参考：第 3、5.2、6.2 节"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import CoffeeAutomaton, measure_complexity, measure_entropy


def run_and_measure(model: str, grid_size: int = 50,
                    total_steps: int = 50000, num_snapshots: int = 50):
    """TODO：运行咖啡元胞自动机模拟并收集测量结果。

参数：
    model：'interacting' 或 'non_interacting'
    grid_size：N
    total_steps：总步数
    num_snapshots：测量点数量

返回：
    times：时间步列表
    complexities：复杂度测量值列表
    entropies：熵测量值列表"""
    # 在此编写代码
    # 提示：
    #   1. 使用给定模型创建 CoffeeAutomaton
    #   2. 计算 steps_per_snapshot = total_steps // num_snapshots
    #   3. 循环调用 step()，然后测量复杂度与熵
    #   4. 返回收集的数据
    pass


def plot_comparison(times_inter, cx_inter, times_non, cx_non):
    """TODO：在同一坐标系中绘制两种模型的复杂度时间曲线。

有相互作用模型应明显先升后降。
无相互作用模型应保持相对平坦。"""
    import matplotlib.pyplot as plt

    # 在此编写代码
    # 提示：plt.plot(times_inter, cx_inter, label='Interacting')
    #       plt.plot(times_non, cx_non, label='Non-interacting')
    pass


if __name__ == '__main__':
    print("正在运行有相互作用模型……")
    t_i, cx_i, e_i = run_and_measure('interacting')

    print("正在运行无相互作用模型……")
    t_n, cx_n, e_n = run_and_measure('non_interacting')

    # 打印概要
    if cx_i is not None and cx_n is not None:
        print(f"\n有相互作用：峰值复杂度 = {max(cx_i)}")
        print(f"无相互作用：峰值复杂度 = {max(cx_n)}")
        print(f"比值：{max(cx_i) / max(cx_n):.1f} 倍")
        print("\n有相互作用模型的峰值复杂度应显著更高。")

        plot_comparison(t_i, cx_i, t_n, cx_n)
