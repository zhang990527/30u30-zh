"""练习 4：两段式编码（two-part code；精致复杂度代理量）

精致复杂度（sophistication；Koppel，1988；Gács–Tromp–Vitányi）衡量两段式描述中“模型”部分的复杂度。Aaronson 的复杂熵（complextropy）是其资源受限版本。

使用两段式编码近似精致复杂度：
  第一部分：粗粒度描述（“模型”）
  第二部分：残差（重建精确状态所需的剩余信息）

第一部分大小被预期呈现隆起：t=0 时低（模型简单），中间时刻高（边界复杂），平衡时低。
参考：Aaronson（2011，博文），精致复杂度部分。

任务：
1. 实现 two_part_code(grid, block_size) -> (part1_size, part2_size)
2. 检查简单网格（全黑、全白）的第一部分是否小
3. 检查随机网格的第一部分是否小（理论模型 = “所有状态”）
4. 展示有结构网格（中间混合阶段）第一部分的峰值
5. 将第一部分轨迹与原始 gzip 轨迹比较
核验提示：具体粗粒化代理量未必满足理论精致复杂度的全部性质。"""

import numpy as np
import gzip


def gzip_size(data: bytes) -> int:
    """gzip 压缩后的大小。"""
    return len(gzip.compress(data, compresslevel=9))


def two_part_code(grid: np.ndarray, block_size: int = 4):
    """计算二维网格的两段式编码。

第一部分（模型）：压缩的粗粒化版本。
第二部分（残差）：压缩的实际状态与重建状态之差。
参数：
    grid：二维二值（0/1）数组。
    block_size：粗粒化尺度。
返回：
    (part1_size, part2_size)，单位为字节。
待完成：实现此函数。
步骤：
1. 对 block_size x block_size 块求平均，粗粒化网格
2. 将粗网格量化为 uint8，压缩得到第一部分
3. 从粗网格重建（扩展回原尺寸）
4. 计算残差 = 原始值 - 重建值
5. 将残差从 [-1,1] 映射到 [0,255]，压缩得到第二部分"""
    # 在此编写代码
    raise NotImplementedError('请实现 two_part_code')


def verify_on_simple_grids():
    """检查第一部分（精致复杂度代理量）是否对简单和随机网格较小，对有结构网格较大。

待完成：
1. 创建全零、全一、随机网格和棋盘格
2. 分别计算 two_part_code
3. 输出第一部分大小
4. 检查简单和随机网格的第一部分是否较小"""
    # 在此编写代码
    raise NotImplementedError('请用简单网格验证')


def sophistication_trajectory(grid_size=64, n_steps=20000, swaps_per_step=50):
    """随时间跟踪咖啡混合过程中的第一部分大小（精致复杂度代理量）。

返回：
    (times, part1_values, gzip_values)，供绘图使用。
待完成：运行咖啡模拟，按间隔测量第一部分和 gzip，检查第一部分是否也有隆起。"""
    # 在此编写代码
    raise NotImplementedError('请实现 trajectory')


if __name__ == "__main__":
    print('练习 4：两段式编码（精致复杂度代理量）')
    print("=" * 40)
    print()

    print('用简单网格验证……')
    verify_on_simple_grids()
    print()

    print('计算精致复杂度代理量轨迹……')
    times, part1_vals, gzip_vals = sophistication_trajectory()

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(times, gzip_vals, 'r-', label='gzip (KC)')
        axes[0].set_title('gzip 复杂度')
        axes[0].legend()
        axes[1].plot(times, part1_vals, 'g-', label='第一部分（精致复杂度）')
        axes[1].set_title('精致复杂度代理量')
        axes[1].legend()
        plt.tight_layout()
        plt.savefig("sophistication_trajectory.png", dpi=100)
        print('已保存：sophistication_trajectory.png')
    except ImportError:
        print('（matplotlib 不可用）')
