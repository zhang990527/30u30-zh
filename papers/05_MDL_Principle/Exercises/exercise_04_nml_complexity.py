"""
练习 4：NML（归一化最大似然）复杂度
==========================================================

NML 是 MDL 的重要理论构造，但通常更难计算。

核心思想：根据模型可能很好解释的
不同数据集范围，对模型进行惩罚。

餐厅类比：
-----------------------
- 简单模型（1 次）：像菜单很小的餐厅
  → 只能做少数菜品，但做得很好
  
- 复杂模型（9 次）：像有 500 道菜的餐厅
  → 几乎什么都能做，这也意味着更强灵活性
  
大菜单餐厅恰好有你要的菜，
相比小菜单餐厅也能提供这道菜，就没那么特别。

NML 复杂度类比为 log（餐厅能做好的菜品数量）

数学形式：
    COMP(M) = log Σ_z max_θ P(z | θ, M)
    
求和遍历全部可能数据集 z。
"""

import numpy as np
from typing import Tuple, Dict


def estimate_nml_complexity_polynomial(
    x: np.ndarray,
    degree: int,
    y_range: Tuple[float, float] = (-100, 100),
    n_samples: int = 1000
) -> float:
    """
    使用蒙特卡洛近似多项式模型的 NML 参数复杂度。
    
    参数复杂度反映模型能很好解释多少不同数据集。
    灵活性越高，通常需要越强惩罚；这里不等于完整随机复杂度。
    
    算法：
    1. 采样许多随机数据集（y 值）
    2. 对每组拟合多项式，获得最大似然
    3. 对最大似然取平均
    4. 取对数，得到复杂度近似（严格积分还需采样分布／体积校正）
    
    参数：
        x：输入值，即固定设计点
        degree：多项式次数
        y_range：用于采样的可能 y 值范围
        n_samples：蒙特卡洛样本数
        
    返回：
        估计的 NML 参数复杂度，单位为比特
        
    示例：
        >>> x = np.linspace(0, 10, 50)
        >>> comp_1 = estimate_nml_complexity_polynomial(x, 1)  # 较低
        >>> comp_5 = estimate_nml_complexity_polynomial(x, 5)  # 预期较高
        >>> assert comp_5 > comp_1, "更高次数应有更高复杂度"
    """
    # TODO：实现 NML 复杂度估计
    # 步骤：
    #   1. 重复 n_samples 次，每次：
    #      a. 从 y_range 采样随机 y 值
    #      b. 用 (x, y_random) 拟合多项式
    #      c. 计算最大对数似然
    #   2. 用 logsumexp 技巧计算平均值的对数
    #   3. 除以 log(2)，转换为比特
    
    pass  # 在此编写代码


def compute_nml_mdl(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    y_range: Tuple[float, float] = None
) -> Tuple[float, float, float]:
    """
    计算 NML 近似 MDL 得分。
    
    NML MDL = -log P(y | θ̂, M) + COMP(M)
            = 拟合项 + 复杂度项
    
    NML 不像两段式编码那样显式指定参数精度。
    复杂度惩罚来自信息论构造；连续数据仍需明确测量与模型范围。
    
    参数：
        x：输入值
        y：目标值
        degree：多项式次数
        y_range：复杂度估计范围，None 时自动确定
        
    返回：
        元组：(total_nml, fit_term, complexity_term)
        
    示例：
        >>> x = np.linspace(0, 10, 50)
        >>> y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.0
        >>> total, fit, comp = compute_nml_mdl(x, y, 2)
        >>> print(f"NML = {fit:.1f}（拟合）+ {comp:.1f}（复杂度）= {total:.1f}")
    """
    # TODO：实现 NML MDL
    # 步骤：
    #   1. 拟合多项式并计算负对数似然
    #   2. 估计 NML 复杂度
    #   3. 返回总和与各项
    
    pass  # 在此编写代码


def compare_nml_vs_two_part(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict[str, int]:
    """
    比较 NML 近似与两段式编码的模型选择。
    
    两者可能一致；严格 NML 不需显式参数精度，但本实现存在近似限制。
    
    参数：
        x：输入值
        y：目标值
        max_degree：尝试的最高次数
        
    返回：
        包含 nml_best 和 two_part_best 次数的字典
    """
    # TODO：实现比较
    
    pass  # 在此编写代码


def visualize_complexity_vs_degree(x: np.ndarray, max_degree: int = 10):
    """
    展示 NML 复杂度近似如何随模型灵活性变化。
    
    此图展示的核心思想是：
    模型越灵活，参数复杂度通常越高，惩罚越强。
    """
    # TODO：计算并打印各次数的复杂度
    # 若未安装 matplotlib，可视化代码为可选项
    
    pass  # 在此编写代码


# =============================================================================
# 辅助函数
# =============================================================================

def logsumexp(log_values: np.ndarray) -> float:
    """
    以数值稳定方式计算 log(sum(exp(values)))。
    
    这是避免 NML 计算溢出的重要工具。
    
    参数：
        log_values：对数值数组
        
    返回：
        log(sum(exp(log_values)))
    """
    # TODO：实现数值稳定的 logsumexp
    # 提示：log(Σ exp(xi)) = max(xi) + log(Σ exp(xi - max))
    
    pass  # 在此编写代码


def gaussian_log_likelihood(
    y: np.ndarray,
    predictions: np.ndarray,
    sigma: float = None
) -> float:
    """
    计算高斯对数似然。
    
    参数：
        y：实际值
        predictions：模型预测
        sigma：标准差，None 时根据残差估计
        
    返回：
        对数似然，单位为奈特，而非比特
    """
    # TODO：实现高斯对数似然
    
    pass  # 在此编写代码


# =============================================================================
# 测试实现
# =============================================================================

def test_implementation():
    """Test NML complexity implementation."""
    
    print("Testing NML (Normalized Maximum Likelihood) Implementation")
    print("=" * 55)
    
    np.random.seed(42)
    
    x = np.linspace(0, 10, 50)
    
    # 测试 1：NML 复杂度随次数增加
    print("\n1. Testing NML complexity vs degree...")
    try:
        complexities = {}
        for deg in [1, 2, 3, 5]:
            comp = estimate_nml_complexity_polynomial(x, deg, n_samples=200)
            complexities[deg] = comp
            print(f"   Degree {deg}: COMP = {comp:.2f} bits")
        
        # 复杂度一般应随次数增加；蒙特卡洛估计可能波动
        assert complexities[5] > complexities[1], "Higher degree should have higher complexity"
        print("   [ok] PASSED - Complexity increases with flexibility")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 2：logsumexp
    print("\n2. Testing logsumexp()...")
    try:
        values = np.array([1000, 1001, 1002])  # 直接使用 exp 会溢出
        result = logsumexp(values)
        
        # 应约为 1002 + log(1 + e^-1 + e^-2) ≈ 1002.41
        assert 1002 < result < 1003, f"Expected ~1002.4, got {result}"
        print(f"   logsumexp([1000, 1001, 1002]) = {result:.2f}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 3：NML MDL 得分
    print("\n3. Testing compute_nml_mdl()...")
    try:
        y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.5  # 真实次数为 2
        
        nml_1 = compute_nml_mdl(x, y, 1)
        nml_2 = compute_nml_mdl(x, y, 2)
        nml_5 = compute_nml_mdl(x, y, 5)
        
        print(f"   NML(degree=1): {nml_1[0]:.1f} = {nml_1[1]:.1f} + {nml_1[2]:.1f}")
        print(f"   NML(degree=2): {nml_2[0]:.1f} = {nml_2[1]:.1f} + {nml_2[2]:.1f}")
        print(f"   NML(degree=5): {nml_5[0]:.1f} = {nml_5[1]:.1f} + {nml_5[2]:.1f}")
        
        if nml_2[0] < nml_1[0] and nml_2[0] < nml_5[0]:
            print("   [ok] PASSED - NML correctly prefers degree 2!")
        else:
            print("   [NOTE] Score ordering varies (NML is stochastic)")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 4：NML 与两段式编码比较
    print("\n4. Testing compare_nml_vs_two_part()...")
    try:
        y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.5
        
        result = compare_nml_vs_two_part(x, y, max_degree=6)
        
        print(f"   NML selects: degree {result['nml_best']}")
        print(f"   Two-Part selects: degree {result['two_part_best']}")
        
        if result['nml_best'] == result['two_part_best']:
            print("   [ok] Methods agree!")
        else:
            print("   [NOTE] Methods disagree (both are valid)")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 5：复杂度可视化
    print("\n5. Testing visualize_complexity_vs_degree()...")
    try:
        visualize_complexity_vs_degree(x, max_degree=5)
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [NOTE] Skipped: {e}")
    
    print("\n" + "=" * 55)
    print("Testing complete!")
    print("\nKey insight: NML complexity grows with model flexibility.")
    print("This is why complex models get penalized more!")


if __name__ == "__main__":
    test_implementation()
