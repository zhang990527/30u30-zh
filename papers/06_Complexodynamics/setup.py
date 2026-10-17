"""setup.py —— 检查第 6 天《复杂动力学第一定律》的环境

运行以下命令，检查依赖是否可用：
    python setup.py
参考：Scott Aaronson（2011）—— https://scottaaronson.blog/?p=762"""

import sys


def check_dependencies():
    """检查所需软件包是否安装。"""
    required = {
        'numpy': 'numpy',
        'gzip': 'gzip',      # 标准库
        'matplotlib': 'matplotlib',
    }

    missing = []
    for name, import_name in required.items():
        try:
            __import__(import_name)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [FAIL] {name} —— 安装命令：pip install {name}")
            missing.append(name)

    return missing


def check_implementation():
    """检查核心实现是否可以正常导入。"""
    try:
        from implementation import (
            create_initial_grid,
            gzip_complexity,
            coarse_grained_kc,
            two_part_code,
            run_simulation,
        )
        print('  [OK] implementation.py 导入成功')
    except ImportError as e:
        print(f"  [FAIL] implementation.py: {e}")
        return False

    # 快速冒烟检查
    try:
        import numpy as np
        grid = create_initial_grid(16)
        assert grid.shape == (16, 16), f"预期 (16,16)，实际为 {grid.shape}"
        assert grid[:8, :].sum() == 16 * 8, '上半部分应全为 1'
        assert grid[8:, :].sum() == 0, '下半部分应全为 0'

        kc = gzip_complexity(grid)
        assert isinstance(kc, int) and kc > 0, f"gzip_complexity 返回 {kc}"

        print('  [OK] 冒烟检查通过')
    except Exception as e:
        print(f"  [FAIL] 冒烟检查：{e}")
        return False

    return True


if __name__ == "__main__":
    print("=" * 50)
    print('第 6 天：复杂动力学第一定律')
    print('环境检查')
    print("=" * 50)
    print()

    print('检查依赖……')
    missing = check_dependencies()
    print()

    if missing:
        stdlib = {'gzip'}
        real_missing = [m for m in missing if m not in stdlib]
        if real_missing:
            print(f"缺少软件包：{', '.join(real_missing)}")
            print(f"安装命令：pip install {' '.join(real_missing)}")
            sys.exit(1)

    print('检查实现……')
    if check_implementation():
        print()
        print('全部检查通过，可以开始使用。')
        print()
        print('快速开始：')
        print("  python train_minimal.py --grid-size 64 --steps 50000")
        print()
        print('或运行完整可视化工具：')
        print("  python visualization.py")
    else:
        print()
        print('部分检查失败，请查看上方错误。')
        sys.exit(1)
