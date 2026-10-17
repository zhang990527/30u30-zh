"""练习 2：咖啡混合模拟

构建本章根据 Aaronson 博文描述的二维像素网格模拟：
- 初始上半部分黑（咖啡）、下半部分白（牛奶）
- 每步随机选取一对相邻咖啡—牛奶像素并交换
- 随时间跟踪状态

这是检验复杂性度量的模型系统。
参考：Aaronson（2011，博文），模型系统部分。

任务：
1. 实现 create_grid(size) -> 二维数组
2. 实现 swap_step(grid) -> 执行一次随机交换
3. 实现 batch_swap(grid, n) -> 高效尝试 n 次交换
4. 运行模拟并保存快照
5. 检查：系统起初分离，发展出卷须纹理，最终在粗粒度下均匀"""

import numpy as np


def create_grid(size: int = 64) -> np.ndarray:
    """创建初始咖啡—牛奶网格。
上半部分 = 1（咖啡/黑），下半部分 = 0（牛奶/白）。

参数：
    size：方形网格的边长。
返回：
    由 0 和 1 组成的二维 NumPy 数组。
待完成：实现此函数。"""
    # 在此编写代码
    raise NotImplementedError('请实现 create_grid')


def swap_step(grid: np.ndarray) -> np.ndarray:
    """执行一次随机交换：找出所有相邻咖啡—牛奶对，均匀随机选择一对并交换。

参数：
    grid：二维二值数组（原地修改）。
返回：
    网格（同一对象）。
待完成：实现此函数。
提示：检查水平和垂直邻居，构建可交换像素对列表，随机选择一对交换。"""
    # 在此编写代码
    raise NotImplementedError('请实现 swap_step')


def batch_swap(grid: np.ndarray, n_swaps: int = 100) -> np.ndarray:
    """高效尝试 n_swaps 次随机交换。

不必每次找出所有可交换像素对（这很慢），而是随机选择像素和方向，检查是否可以交换。

参数：
    grid：二维二值数组（原地修改）。
    n_swaps：尝试交换的次数。
返回：
    网格（同一对象）。
待完成：实现此函数。"""
    # 在此编写代码
    raise NotImplementedError('请实现 batch_swap')


def run_and_snapshot(grid_size=32, n_steps=5000, swaps_per_step=20,
                     snapshot_interval=500):
    """运行模拟，按间隔收集网格快照。

返回：
    (step, grid_copy) 元组列表。
待完成：实现此函数。"""
    # 在此编写代码
    raise NotImplementedError('请实现 run_and_snapshot')


if __name__ == "__main__":
    print('练习 2：咖啡混合模拟')
    print("=" * 40)
    print()

    snapshots = run_and_snapshot()
    print(f"已收集 {len(snapshots)} 张快照")

    # 如果 matplotlib 可用，展示演化过程
    try:
        import matplotlib.pyplot as plt
        n = min(6, len(snapshots))
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        indices = np.linspace(0, len(snapshots) - 1, n, dtype=int)
        for ax, idx in zip(axes, indices):
            step, grid = snapshots[idx]
            ax.imshow(grid, cmap='gray', interpolation='nearest')
            ax.set_title(f"步数={step}")
            ax.axis('off')
        plt.tight_layout()
        plt.savefig("coffee_mixing_snapshots.png", dpi=100)
        print('已保存：coffee_mixing_snapshots.png')
    except ImportError:
        print('（matplotlib 不可用，跳过绘图）')
