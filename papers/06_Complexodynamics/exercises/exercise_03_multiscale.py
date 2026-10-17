"""练习 3：多尺度复杂性分析

Luca Trevisan（博文评论 #17）建议：在各尺度将网格划分成块，计算每块平均含量，随时间跟踪各尺度的 KC。这给出复杂性“隆起”所在位置的二维图景（尺度 x 时间）。

Sean Carroll（#6-7）也强调粗粒化：测量模糊状态的 KC，以捕捉宏观结构。
参考：Aaronson（2011，博文），评论 #6-7 和 #17（编号沿用原仓库，未独立核验）。

任务：
1. 实现 coarse_grain(grid, block_size) -> 较小数组
2. 实现 coarse_grained_kc(grid, block_size) -> int
3. 对咖啡模拟在多个尺度运行
4. 绘制各尺度 KC 随时间的变化
5. 创建 Trevisan 描述的尺度—时间热图"""

import numpy as np
import gzip


def coarse_grain(grid: np.ndarray, block_size: int) -> np.ndarray:
    """对 block_size x block_size 块内的网格值求平均。

参数：
    grid：二维数组。
    block_size：块的边长。
返回：
    较小的二维数组，每格为一个块的均值。
待完成：实现此函数。"""
    # 在此编写代码
    raise NotImplementedError('请实现 coarse_grain')


def coarse_grained_kc(grid: np.ndarray, block_size: int) -> int:
    """粗粒化网格的 KC 近似。

步骤：
1. 粗粒化网格
2. 量化为 uint8（0-255）
3. 用 gzip 压缩

参数：
    grid：二维二值数组。
    block_size：粗粒化尺度。
返回：
    压缩后的字节数。
待完成：实现此函数。"""
    # 在此编写代码
    raise NotImplementedError('请实现 coarse_grained_kc')


def multiscale_analysis(grid: np.ndarray) -> dict:
    """计算多个粗粒化尺度的 KC 近似。

参数：
    grid：二维二值数组。
返回：
    将 block_size 映射到压缩大小的字典。
待完成：
1. 使用不超过 grid_size/4 的 2 的幂作为尺度
2. 对每个尺度计算 coarse_grained_kc
3. 返回字典"""
    # 在此编写代码
    raise NotImplementedError('请实现 multiscale_analysis')


def run_multiscale_experiment(grid_size=64, n_steps=20000, swaps_per_step=50):
    """运行咖啡模拟，随时间跟踪多尺度 KC。

返回：
    (times, scale_data)，其中 scale_data[scale] 是 KC 值列表。
待完成：从练习 2 或 implementation.py 导入咖啡混合实现，运行模拟，按间隔测量多尺度 KC。"""
    # 在此编写代码
    raise NotImplementedError('请实现 experiment')


if __name__ == "__main__":
    print('练习 3：多尺度复杂性分析')
    print("=" * 40)
    print()

    times, scale_data = run_multiscale_experiment()

    try:
        import matplotlib.pyplot as plt

        # 折线图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for scale, values in scale_data.items():
            axes[0].plot(times, values, label=f"尺度={scale}")
        axes[0].set_xlabel('交换尝试总次数')
        axes[0].set_ylabel('压缩后字节数')
        axes[0].set_title('不同尺度的 KC')
        axes[0].legend()

        # 热图
        scales = sorted(scale_data.keys())
        matrix = np.array([scale_data[s] for s in scales])
        axes[1].imshow(matrix, aspect='auto', cmap='hot')
        axes[1].set_yticks(range(len(scales)))
        axes[1].set_yticklabels([str(s) for s in scales])
        axes[1].set_title('多尺度热图（Trevisan）')

        plt.tight_layout()
        plt.savefig("multiscale_analysis.png", dpi=100)
        print('已保存：multiscale_analysis.png')
    except ImportError:
        print('（matplotlib 不可用，跳过绘图）')
