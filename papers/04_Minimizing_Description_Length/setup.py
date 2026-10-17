#!/usr/bin/env python3
"""
第 4 天环境设置脚本
==================

快速设置 MDL 与贝叶斯神经网络环境。

此脚本将：
1. 检查 Python 版本
2. 安装所需依赖包（包括 SciPy）
3. 验证安装
4. 快速测试贝叶斯层

用法：
    python setup.py
"""

import sys
import subprocess
import os


def check_python_version():
    """Check if Python version is 3.7+."""
    print('检查 Python 版本……')
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"[FAIL] 需要 Python 3.7 或更高，当前版本为 {version.major}.{version.minor}")
        return False
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_requirements():
    """Install required packages."""
    print('\n安装依赖……')
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print('[OK] 依赖安装完成')
        return True
    except subprocess.CalledProcessError:
        print('[FAIL] 依赖安装失败')
        return False


def verify_imports():
    """Verify that all required packages can be imported."""
    print('\n验证导入……')
    required = ['numpy', 'matplotlib', 'scipy', 'seaborn', 'jupyter']
    
    all_ok = True
    for package in required:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[FAIL] {package} - 导入失败")
            all_ok = False
    
    return all_ok


def run_quick_test():
    """Run a quick test of the MDL Network implementation."""
    print('\n运行快速测试……')
    try:
        from implementation import MDLNetwork
        import numpy as np
        
        # 创建小型贝叶斯网络
        # [1 个输入] -> [5 个隐藏单元] -> [1 个输出]
        net = MDLNetwork(input_size=1, hidden_size=5, output_size=1)
        
        # 创建测试数据
        x = np.array([[0.5]])
        
        # 测试前向传播（采样权重）
        y1 = net.forward(x)
        y2 = net.forward(x) # 噪声应使结果略有不同！
        
        # 测试 KL 散度计算
        kl = net.total_kl()
        
        print(f"[OK] MDL 网络已初始化")
        print(f"   - 第 1 次前向传播：{y1.item():.4f}")
        print(f"   - 第 2 次前向传播：{y2.item():.4f}（注意：已启用噪声！）")
        print(f"   - 复杂度代价（KL）：{kl:.4f}")
        
        if y1.shape == (1, 1) and kl != 0:
            print('[OK] 测试通过')
            return True
        else:
            print('[FAIL] 维度或 KL 计算错误')
            return False
            
    except Exception as e:
        print(f"[FAIL] MDL 测试失败：{e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print('第 4 天：MDL 与贝叶斯网络——环境设置')
    print("=" * 60)
    
    # 检查 Python 版本
    if not check_python_version():
        sys.exit(1)
    
    # 安装依赖
    if not install_requirements():
        print('\n[NOTE] 安装失败，请手动尝试：')
        print("    pip install -r requirements.txt")
        sys.exit(1)
    
    # 验证导入
    if not verify_imports():
        print('\n[NOTE] 部分依赖包导入失败')
        sys.exit(1)
    
    # 运行测试
    if not run_quick_test():
        print('\n[NOTE] 测试失败，请检查 implementation.py')
        sys.exit(1)
    
    # 成功！
    print("\n" + "=" * 60)
    print('[OK] 环境设置完成！')
    print("=" * 60)
    print('\n接下来：')
    print('  1. 阅读 README.md 了解理论')
    print('  2. 阅读 paper_notes.md 中的直观解释')
    print('  3. 运行实验：python train_minimal.py')
    print('  4. 打开 notebook.ipynb 查看不确定性包络！')
    print('\n准备就绪，运行：python train_minimal.py')


if __name__ == "__main__":
    main()