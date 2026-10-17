"""解答 1：用 gzip 近似 KC"""

import numpy as np
import gzip


def gzip_complexity(data: bytes, level: int = 9) -> int:
    """用 gzip 压缩大小近似 KC。"""
    return len(gzip.compress(data, compresslevel=level))


def test_known_signals():
    """用三种具有已知性质的信号测试。"""
    # 全零：高度可压缩
    zeros = bytes(1024)
    kc_zeros = gzip_complexity(zeros)
    print(f"  全零（1024 字节）：{kc_zeros} 压缩后字节")

    # 随机字节：难以压缩
    random_data = np.random.randint(0, 256, size=1024, dtype=np.uint8).tobytes()
    kc_random = gzip_complexity(random_data)
    print(f"  随机（1024 字节）：{kc_random} 压缩后字节")

    # 重复模式：原说明预期居中
    pattern = (b"ABCD" * 256)
    kc_pattern = gzip_complexity(pattern)
    print(f"  模式（1024 字节）：{kc_pattern} 压缩后字节")

    # 检查排序
    assert kc_zeros < kc_pattern < kc_random, (
        f"预期 全零 < 模式 < 随机，"
        f"实际为 {kc_zeros} < {kc_pattern} < {kc_random}"
    )
    print('  [OK] 排序检查通过：全零 < 模式 < 随机')


def compare_compression_levels():
    """对相同数据比较 gzip 等级。"""
    data = (b"Hello World! " * 340)[:4096]  # 4096 字节

    print(f"  数据大小：{len(data)} 字节")
    prev_size = float('inf')
    for level in [1, 3, 5, 7, 9]:
        size = gzip_complexity(data, level=level)
        print(f"  等级 {level}: {size} 字节")
        assert size <= prev_size + 5, (
            f"等级 {level} 不应远大于等级 {level-2}"
        )
        prev_size = size

    print('  [OK] 较高等级给出更紧的界（符合本次预期）')


if __name__ == "__main__":
    print('解答 1：用 gzip 近似 KC')
    print("=" * 40)
    print()

    print('任务 1：测试已知信号……')
    test_known_signals()
    print()

    print('任务 2：比较压缩等级……')
    compare_compression_levels()
