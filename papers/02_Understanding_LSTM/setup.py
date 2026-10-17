#!/usr/bin/env python3
"""
第 2 天环境设置脚本
==================

《理解 LSTM 网络》的快速设置。

本脚本将：
1. 检查 Python 版本
2. 安装所需软件包
3. 验证安装
4. 运行快速测试

用法：
    python setup.py
"""

import sys
import subprocess
import os


def check_python_version():
    """检查 Python 版本是否为 3.7 或以上。"""
    print('正在检查 Python 版本……')
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"失败：需要 Python 3.7 或以上，当前为 {version.major}.{version.minor}")
        return False
    print(f"通过：Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_requirements():
    """安装所需软件包。"""
    print('\n正在安装依赖……')
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print('通过：依赖安装完成')
        return True
    except subprocess.CalledProcessError:
        print('失败：依赖安装未完成')
        return False


def verify_imports():
    """验证所需软件包均能导入。"""
    print('\n正在验证导入……')
    required = ['numpy', 'matplotlib', 'jupyter']
    
    all_ok = True
    for package in required:
        try:
            __import__(package)
            print(f"通过：{package}")
        except ImportError:
            print(f"失败：{package} - 导入失败")
            all_ok = False
    
    return all_ok


def run_quick_test():
    """快速测试 LSTM 实现。"""
    print('\n正在运行快速测试……')
    try:
        from implementation import LSTM
        
        # 创建小型 LSTM
        lstm = LSTM(input_size=5, hidden_size=8, output_size=5)
        
        # 测试前向传播
        import numpy as np
        inputs = [0, 1, 2, 3, 4]
        targets = [1, 2, 3, 4, 0]
        h = np.zeros(8)
        C = np.zeros(8)
        
        loss = lstm.forward(inputs, targets, h, C)
        
        if loss > 0:
            print(f"通过：LSTM 测试通过（损失：{loss:.4f})")
            return True
        else:
            print('失败：LSTM 测试未通过')
            return False
            
    except Exception as e:
        print(f"失败：LSTM 测试未通过：{e}")
        return False


def main():
    """环境设置主函数。"""
    print("=" * 60)
    print('第 2 天：理解 LSTM 网络——环境设置')
    print("=" * 60)
    
    # 检查 Python 版本
    if not check_python_version():
        sys.exit(1)
    
    # 安装依赖
    if not install_requirements():
        print('\n安装失败。可以手动尝试：')
        print("    pip install -r requirements.txt")
        sys.exit(1)
    
    # 验证导入
    if not verify_imports():
        print('\n部分软件包导入失败')
        sys.exit(1)
    
    # 运行测试
    if not run_quick_test():
        print('\n测试失败，请检查 implementation.py')
        sys.exit(1)
    
    # 设置成功！
    print("\n" + "=" * 60)
    print('设置完成。')
    print("=" * 60)
    print('\n后续步骤：')
    print('  1. 阅读 README.md 了解概况')
    print('  2. 阅读 paper_notes.md 中的通俗解释')
    print('  3. 尝试：python train_minimal.py --data data/input.txt')
    print('  4. 在 Jupyter 中打开 notebook.ipynb')
    print('  5. 尝试 exercises/ 目录中的练习')


if __name__ == "__main__":
    main()
