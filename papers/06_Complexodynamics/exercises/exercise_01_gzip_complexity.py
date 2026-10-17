"""练习 1：用 gzip 近似柯尔莫哥洛夫复杂度（Kolmogorov complexity，KC）

Aaronson 讨论了用 gzip 压缩大小粗略近似 KC。本练习将构建该工具，并用已知信号验证。

参考：Aaronson（2011，博文）、Lauren Ouellette 的实验。

任务：
1. 实现 gzip_complexity(data: bytes) -> int
2. 测试三个已知情形：
   a. 全零（高度可压缩，KC 低）
   b. 随机字节（不可压缩，KC 高）
   c. 重复模式（原说明预期 KC 居中）
3. 绘制压缩大小与规律程度的关系
4. 比较 gzip 等级 1 和 9

预期结果：随机 > 模式 > 均匀；gzip 等级 9 比等级 1 给出更紧（更低）的上界。
核验提示：具体排序和压缩等级效果依赖输入，不是普遍定理。"""

import numpy as np
import gzip


def gzip_complexity(data: bytes, level: int = 9) -> int:
    """用 gzip 压缩大小近似柯尔莫哥洛夫复杂度。

参数：
    data：待压缩的原始字节。
    level：压缩等级（1-9）；使用 9 以获得较好的近似。
返回：
    压缩数据的字节数。
待完成：实现此函数。
提示：gzip.compress(data, compresslevel=level) 返回压缩后的字节。"""
    # 在此编写代码
    raise NotImplementedError('请实现 gzip_complexity')


def test_known_signals():
    """用具有已知性质的三种信号测试 gzip_complexity。

待完成：
1. 创建 1024 个全零字节
2. 创建 1024 个随机字节
3. 创建 1024 字节的重复模式（如 b"ABCD" * 256）
4. 输出各自的压缩大小
5. 检查：随机 > 模式 > 全零"""
    # 在此编写代码
    raise NotImplementedError('请创建并测试三种信号')


def compare_compression_levels():
    """对相同数据比较 gzip 等级 1 和 9。

待完成：
1. 创建 4096 字节的重复模式
2. 分别以等级 1、3、5、7、9 压缩
3. 输出各等级的压缩大小
4. 检查等级 9 是否最小（给出最紧上界）"""
    # 在此编写代码
    raise NotImplementedError('请比较压缩等级')


if __name__ == "__main__":
    print('练习 1：用 gzip 近似 KC')
    print("=" * 40)
    print()

    print('任务 1：测试已知信号……')
    test_known_signals()
    print()

    print('任务 2：比较压缩等级……')
    compare_compression_levels()
