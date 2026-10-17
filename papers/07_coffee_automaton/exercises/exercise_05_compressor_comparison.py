"""练习 5：比较压缩器

论文图 5 表明，不同文件压缩程序产生的复杂度曲线在定性上相似。

“如图 5 所示，采用不同压缩程序时，粗粒化度量所得结果
在定性上相似。”（第 5.2 节）

任务：
  1. 运行有相互作用的咖啡元胞自动机
  2. 在每个快照处，用多个压缩器压缩粗粒化状态
  3. 为每个压缩器绘制复杂度曲线
  4. 验证所有曲线都呈先升后降模式

Python 中可用的压缩器：
  - gzip（LZ77 + Huffman）
  - bz2（Burrows-Wheeler + Huffman）
  - lzma（LZMA2）
  - zlib（与 gzip 相同算法，不同封装）

论文参考：第 5.2 节，图 5"""

import sys
import os
import gzip
import bz2
import lzma
import zlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import (
    CoffeeAutomaton, coarse_grain, threshold_array,
    row_majority_adjust, grid_to_bytes
)


def compress_with_all(data: bytes) -> dict:
    """TODO：使用多种算法压缩数据。

返回从压缩器名称到压缩大小的字典。
使用 gzip、bz2、lzma、zlib。"""
    # 在此编写代码
    pass


def run_compressor_comparison(grid_size=50, total_steps=50000, num_snapshots=60):
    """TODO：运行自动机，收集各压缩器的复杂度估计值。

返回：
    times：时间步列表
    results：{压缩器名称: [各时间步的大小]} 字典"""
    # 在此编写代码
    # 提示：
    #   1. 创建 CoffeeAutomaton
    #   2. 在每个快照处，获取网格、粗粒化、阈值化、调整
    #   3. 转为字节
    #   4. 用所有压缩器压缩
    #   5. 收集大小
    pass


def plot_comparison(times, results):
    """TODO：为所有压缩器绘制复杂度曲线。

尽管绝对数值不同，所有曲线都应表现出定性一致的先升后降模式。"""
    import matplotlib.pyplot as plt

    # 在此编写代码
    pass


if __name__ == '__main__':
    print("压缩器比较（图 5）")
    print("=" * 50)

    result = run_compressor_comparison()

    if result is not None:
        times, compressor_results = result
        print("\n各压缩器的复杂度峰值：")
        for name, sizes in compressor_results.items():
            peak = max(sizes)
            peak_t = times[sizes.index(peak)] if isinstance(sizes, list) else times[np.argmax(sizes)]
            print(f"  {name:>6}：{peak:>6} 字节，位于 t={peak_t}")

        plot_comparison(times, compressor_results)
