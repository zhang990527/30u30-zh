"""练习 5：比较复杂性度量

综合练习：对同一次咖啡混合模拟运行全部复杂性度量，比较曲线，观察哪些度量最清晰地呈现特征性“隆起”。

比较以下度量：
1. gzip 复杂度（原始 KC 代理量）
2. 多尺度粗粒化 KC（Carroll 的想法）
3. 两段式编码的第一部分（精致复杂度代理量）
4. 边界比例（界面长度）
5. 香农熵（原说明预期单调，不出现隆起；全局像素熵在本模拟中守恒）

参考：Aaronson（2011，博文），关于哪种度量“合适”的讨论。
任务：
1. 运行一次模拟
2. 每个测量时刻记录全部 5 个量
3. 在同一张归一化图上绘制它们
4. 找出隆起最清晰的度量
5. 讨论：哪个最接近 Aaronson 的复杂熵？"""

import numpy as np
import gzip
import sys
import os

# 将上级目录加入导入路径，以导入 implementation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_comparison(grid_size=64, n_steps=30000, swaps_per_step=50,
                   measure_interval=300):
    """运行模拟，收集全部复杂性度量。

返回：
    包含 'times'、'gzip'、'coarse_kc_4'、'coarse_kc_8'、'part1'、'boundary'、'entropy' 的字典。
待完成：
1. 导入或实现 create_grid、batch_swap、gzip_complexity、coarse_grained_kc、two_part_code、grid_entropy、boundary_fraction
2. 运行模拟，按间隔测量
3. 将全部数据作为字典返回"""
    # 在此编写代码
    raise NotImplementedError('请实现 run_comparison')


def normalize_to_01(values):
    """将数值列表归一化到 [0, 1] 范围。"""
    arr = np.array(values, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def plot_comparison(data):
    """将全部度量绘制在同一张归一化图上。

待完成：
1. 将各度量归一化到 [0, 1]
2. 在相同坐标轴上绘制
3. 标出各自峰值时刻
4. 添加图例
5. 说明哪些度量呈现隆起，哪些没有"""
    # 在此编写代码
    raise NotImplementedError('请绘制度量比较图')


def analyze_results(data):
    """输出各度量是否呈现隆起的分析。

对每个度量给出：
- 峰值时刻
- 峰值/初值之比
- 峰值/终值之比
- 是否单调（无隆起），或是否有明显峰值
待完成：实现此分析。"""
    # 在此编写代码
    raise NotImplementedError('请分析结果')


if __name__ == "__main__":
    print('练习 5：比较复杂性度量')
    print("=" * 40)
    print()

    print('运行模拟并计算全部度量……')
    data = run_comparison()
    print()

    print('分析：')
    analyze_results(data)
    print()

    print('绘制度量比较图……')
    plot_comparison(data)
    print()

    print('思考问题：')
    print('1. 哪种度量的隆起最明显？')
    print('2. gzip 与精致复杂度代理量是否同时达到峰值？')
    print('3. 边界比例是好的复杂性代理量吗？为什么？')
    print('4. 哪种度量最接近 Aaronson 的复杂熵？')
