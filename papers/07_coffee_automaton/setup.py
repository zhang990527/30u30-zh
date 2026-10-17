"""setup.py——检查第 7 天“咖啡元胞自动机”的依赖

必需：numpy、matplotlib（仅此而已，不需要 scipy）。
论文模型只使用二值网格和 gzip 压缩。"""

import subprocess
import sys


def check_package(name):
    """检查软件包是否可以导入。"""
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def main():
    required = ['numpy', 'matplotlib']
    builtin = ['gzip', 'bz2', 'lzma', 'zlib']  # 用于压缩，均属于标准库

    print("第 7 天：咖啡元胞自动机——依赖检查")
    print("=" * 50)

    all_ok = True

    # 检查必需的软件包
    for pkg in required:
        if check_package(pkg):
            print(f"  [OK] {pkg}")
        else:
            print(f"  [缺失] {pkg}")
            all_ok = False

    # 检查标准库模块（正常情况下始终可用）
    for pkg in builtin:
        if check_package(pkg):
            print(f"  [OK] {pkg}（标准库）")
        else:
            print(f"  [缺失] {pkg}——标准库不应出现此问题")
            all_ok = False

    if all_ok:
        print("\n所有依赖均已满足，可以开始。")
        print("\n快速开始：")
        print("  python train_minimal.py --grid-size 50 --steps 100000")
    else:
        print("\n缺少软件包，请运行：")
        print("  pip install -r requirements.txt")

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
