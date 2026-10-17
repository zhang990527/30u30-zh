import torch
from implementation import RelationNetwork

def verify_invariance():
    """
    TODO：通过程序检查 RN 的排列不变性；有限测试不等于数学证明。
    第 2.1 节原仓库引文：“RN 对集合中对象的顺序保持不变。”
    
    1. 创建随机对象集合（B=1、N=5、D=8）。
    2. 创建 RelationNetwork。
    3. 对原始对象执行前向传播。
    4. 沿 N 维打乱对象。
    5. 对打乱后的对象执行前向传播。
    6. 断言两次输出一致。
    """
    # 在此填写代码
    pass

if __name__ == "__main__":
    print("-" * 50)
    print('练习 3：顺序不变性验证')
    print("-" * 50)
    verify_invariance()
    print('练习 3：[COMPLETE]')
