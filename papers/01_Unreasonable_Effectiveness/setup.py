#!/usr/bin/env python3
"""
第 1 天环境安装脚本
==================

为《循环神经网络令人惊叹的有效性》快速准备环境。

本脚本执行：
1. 检查 Python 版本
2. 安装必需依赖
3. 验证安装
4. 运行快速检查

用法：
    python setup.py
"""

import sys
import subprocess
import os


def check_python_version():
    """检查 Python 版本是否至少为 3.7。"""
    print("正在检查 Python 版本……")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"Python 3.7+ required. You have {version.major}.{version.minor}")
        return False
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_requirements():
    """安装必需依赖。"""
    print("\n正在安装依赖……")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("依赖已安装")
        return True
    except subprocess.CalledProcessError:
        print("依赖安装失败")
        return False


def verify_imports():
    """验证所有必需依赖是否可以导入。"""
    print("\n正在验证导入……")
    required = ['numpy', 'matplotlib', 'jupyter']
    
    all_ok = True
    for package in required:
        try:
            __import__(package)
            print(f"{package}")
        except ImportError:
            print(f"{package} — 导入失败")
            all_ok = False
    
    return all_ok


def run_quick_test():
    """对 RNN 实现运行快速检查。"""
    print("\n正在运行快速检查……")
    try:
        from implementation import CharRNN
        
        # 创建微型 RNN
        rnn = CharRNN(vocab_size=5, hidden_size=8)
        
        # 检查前向传播
        import numpy as np
        inputs = [0, 1, 2, 3, 4]
        targets = [1, 2, 3, 4, 0]
        h = np.zeros(8)
        
        loss = rnn.forward(inputs, targets, h)
        
        if loss > 0:
            print(f"RNN 检查通过（损失：{loss:.4f})")
            return True
        else:
            print("RNN 检查失败")
            return False
            
    except Exception as e:
        print(f"RNN 检查失败：{e}")
        return False


def main():
    """安装主函数。"""
    print("=" * 60)
    print("第 1 天：循环神经网络令人惊叹的有效性——环境安装")
    print("=" * 60)
    
    # 检查 Python 版本
    if not check_python_version():
        sys.exit(1)
    
    # 安装依赖
    if not install_requirements():
        print("\n安装失败，请手动尝试：")
        print("    pip install -r requirements.txt")
        sys.exit(1)
    
    # 验证导入
    if not verify_imports():
        print("\n部分依赖导入失败")
        sys.exit(1)
    
    # 运行检查
    if not run_quick_test():
        print("\n检查失败，请检查 implementation.py")
        sys.exit(1)
    
    # 安装成功
    print("\n" + "=" * 60)
    print("环境安装完成！")
    print("=" * 60)
    print("\n后续步骤：")
    print("  1. 阅读 README.md，了解整体内容")
    print("  2. 阅读 paper_notes.md 中的通俗解释")
    print("  3. 尝试：python train_minimal.py --data data/input.txt")
    print("  4. 在 Jupyter 中打开 notebook.ipynb")
    print("  5. 尝试 exercises/ 目录中的练习")
    print("\n祝学习顺利！")


if __name__ == "__main__":
    main()
