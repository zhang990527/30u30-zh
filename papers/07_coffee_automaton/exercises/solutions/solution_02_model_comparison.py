"""解答 2：有相互作用与无相互作用模型的比较"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import CoffeeAutomaton, measure_complexity, measure_entropy


def run_and_measure(model: str, grid_size: int = 50,
                    total_steps: int = 50000, num_snapshots: int = 50):
    """运行模拟并收集测量结果。"""
    ca = CoffeeAutomaton(grid_size, model, seed=42)
    steps_per = total_steps // num_snapshots

    times = [0]
    complexities = [ca.complexity()]
    entropies = [ca.entropy()]

    for i in range(num_snapshots):
        ca.step(steps_per)
        times.append(ca.time)
        complexities.append(ca.complexity())
        entropies.append(ca.entropy())

    return times, complexities, entropies


def plot_comparison(times_inter, cx_inter, times_non, cx_non):
    """绘制两种模型的复杂度。"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    plt.plot(times_inter, cx_inter, 'r-', label='有相互作用', linewidth=2)
    plt.plot(times_non, cx_non, 'b--', label='无相互作用', linewidth=2)

    plt.xlabel('时间步')
    plt.ylabel('表观复杂度（gzip 字节数）')
    plt.title('有相互作用与无相互作用模型\n'
              '只有有相互作用模型呈现真正的复杂性')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("正在运行有相互作用模型……")
    t_i, cx_i, e_i = run_and_measure('interacting')

    print("正在运行无相互作用模型……")
    t_n, cx_n, e_n = run_and_measure('non_interacting')

    print(f"\n有相互作用：峰值复杂度 = {max(cx_i)}")
    print(f"无相互作用：峰值复杂度 = {max(cx_n)}")
    print(f"比值：{max(cx_i) / max(cx_n):.1f} 倍")

    plot_comparison(t_i, cx_i, t_n, cx_n)
