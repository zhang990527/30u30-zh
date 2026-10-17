"""
练习 1：两段式编码 MDL
==============================

两段式编码是 MDL 的基础形式：
    L(total) = L(H) + L(D|H)

其中：
    L(H)：描述模型（假设）的比特数
    L(D|H)：给定模型后描述数据的比特数

任务：实现多项式回归的两段式编码。

编码类比（原仓库称第 3 节框架；实际对应原文 §2.4）：
-------------------------------------------------------
向接收方发送数据时，可以：
1. 分别发送每个数字（朴素、代价高）
2. 发送公式与小修正（更精巧，可能更省）

两段式编码将其形式化：模型 + 残差 = 总消息。
"""

import numpy as np
from typing import Tuple


def universal_code_length(n: int) -> float:
    """
    计算正整数的通用码长。
    
    无需预先知道整数范围的通用编码方法。
    使用迭代对数和：log(n) + log(log(n)) + ...，仅保留正项
    
    参数：
        n：需要编码的正整数
        
    返回：
        以比特为单位的码长
        
    示例：
        >>> universal_code_length(1)  # 应较小
        >>> universal_code_length(100)  # 应较大
    """
    # TODO：实现通用整数编码
    # 提示：从 log2(c0) 开始，其中 c0 ≈ 2.865
    # 随后累加 log2(n) + log2(log2(n)) + ...，仅保留正项
    
    pass  # 在此编写代码


def model_description_length(degree: int, precision_bits: int = 32) -> float:
    """
    计算 L(H)：描述多项式模型所需的比特数。
    
    需要编码：
    1. 多项式次数
    2. 系数（每个 precision_bits，再加符号位）
    
    参数：
        degree：多项式次数，0 为常数，1 为线性，依此类推
        precision_bits：每个系数的比特数
        
    返回：
        模型描述所需总比特数
        
    示例：
        >>> model_description_length(0)  # 常数：1 个系数
        >>> model_description_length(2)  # 二次：3 个系数
    """
    # TODO：实现模型代价计算
    # 提示：
    #   - 次数代价：universal_code_length(degree + 1)
    #   - 系数代价：(degree + 1) * (precision_bits + 1)
    
    pass  # 在此编写代码


def data_description_length(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    precision_bits: int = 32
) -> float:
    """
    计算 L(D|H)：给定模型后描述数据的代价。
    
    拟合多项式后，对残差（误差）编码。
    使用高斯模型时，每个残差的代价近似为：
        0.5 * log2(2π σ²) + error² / (2σ² ln(2))
    
    参数：
        x：输入值
        y：目标值
        degree：要拟合的多项式次数
        precision_bits：编码 sigma 的比特数
        
    返回：
        数据描述总代价
        
    示例：
        >>> x = np.linspace(0, 10, 50)
        >>> y = 2*x + 1 + np.random.randn(50) * 0.1  # 近似线性
        >>> data_description_length(x, y, 1)  # 应较小，拟合好
        >>> data_description_length(x, y, 0)  # 应较大，拟合差
    """
    # TODO：实现数据代价计算
    # 步骤：
    #   1. 拟合多项式：coeffs = np.polyfit(x, y, degree)
    #   2. 计算预测：predictions = np.poly1d(coeffs)(x)
    #   3. 计算残差：residuals = y - predictions
    #   4. 估计 sigma：sigma = np.std(residuals)
    #   5. 计算以比特为单位的高斯 NLL
    #   6. 加入编码 sigma 的代价
    
    pass  # 在此编写代码


def two_part_mdl(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    precision_bits: int = 32
) -> Tuple[float, float, float]:
    """
    计算完整的两段式 MDL 得分。
    
    合并模型描述长度与数据描述长度。
    
    参数：
        x：输入值
        y：目标值
        degree：多项式次数
        precision_bits：编码精度
        
    返回：
        元组：(total_mdl, model_cost, data_cost)
        
    示例：
        >>> x = np.linspace(0, 10, 50)
        >>> y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.0
        >>> total, model, data = two_part_mdl(x, y, 2)
        >>> print(f"总计：{total:.1f} = 模型：{model:.1f} + 数据：{data:.1f}")
    """
    # TODO：合并模型与数据描述长度
    
    pass  # 在此编写代码


# =============================================================================
# 测试实现
# =============================================================================

def test_implementation():
    """Test your implementation against expected behavior."""
    
    print("Testing Two-Part Code MDL Implementation")
    print("=" * 50)
    
    np.random.seed(42)
    
    # 测试 1：通用码长
    print("\n1. Testing universal_code_length()...")
    try:
        len_1 = universal_code_length(1)
        len_10 = universal_code_length(10)
        len_100 = universal_code_length(100)
        
        assert len_1 < len_10 < len_100, "Larger integers should have longer codes"
        assert len_1 > 0, "Code length should be positive"
        print(f"   universal_code_length(1) = {len_1:.2f}")
        print(f"   universal_code_length(10) = {len_10:.2f}")
        print(f"   universal_code_length(100) = {len_100:.2f}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 2：模型描述长度
    print("\n2. Testing model_description_length()...")
    try:
        model_0 = model_description_length(0)
        model_2 = model_description_length(2)
        model_5 = model_description_length(5)
        
        assert model_0 < model_2 < model_5, "Higher degree = more parameters = longer code"
        print(f"   model_description_length(0) = {model_0:.2f}")
        print(f"   model_description_length(2) = {model_2:.2f}")
        print(f"   model_description_length(5) = {model_5:.2f}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 3：数据描述长度
    print("\n3. Testing data_description_length()...")
    try:
        x = np.linspace(0, 10, 50)
        y = 3 + 2*x + np.random.randn(50) * 0.5  # 低噪声线性数据
        
        data_0 = data_description_length(x, y, 0)  # 常数模型，拟合差
        data_1 = data_description_length(x, y, 1)  # 线性模型，拟合好
        
        assert data_1 < data_0, "Better fitting model should have lower data cost"
        print(f"   data_cost(degree=0) = {data_0:.2f} (poor fit)")
        print(f"   data_cost(degree=1) = {data_1:.2f} (good fit)")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 4：两段式 MDL
    print("\n4. Testing two_part_mdl()...")
    try:
        x = np.linspace(0, 10, 50)
        y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.5  # 二次函数
        
        mdl_1, m1, d1 = two_part_mdl(x, y, 1)
        mdl_2, m2, d2 = two_part_mdl(x, y, 2)
        mdl_5, m5, d5 = two_part_mdl(x, y, 5)
        
        # 2 次模型应有最低总 MDL，这是该测试的预期
        assert mdl_2 < mdl_1, "True degree (2) should beat underfit (1)"
        assert mdl_2 < mdl_5, "True degree (2) should beat overfit (5)"
        
        print(f"   MDL(degree=1) = {mdl_1:.1f} (model: {m1:.1f}, data: {d1:.1f})")
        print(f"   MDL(degree=2) = {mdl_2:.1f} (model: {m2:.1f}, data: {d2:.1f}) ← BEST")
        print(f"   MDL(degree=5) = {mdl_5:.1f} (model: {m5:.1f}, data: {d5:.1f})")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 5：模型选择
    print("\n5. Testing model selection...")
    try:
        x = np.linspace(0, 10, 50)
        y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.5  # 真实次数 = 2
        
        scores = {d: two_part_mdl(x, y, d)[0] for d in range(6)}
        best_degree = min(scores, key=scores.get)
        
        print(f"   MDL scores: {[f'{d}:{scores[d]:.0f}' for d in range(6)]}")
        print(f"   Selected degree: {best_degree}")
        print(f"   True degree: 2")
        
        if best_degree == 2:
            print("   [ok] PASSED - MDL found the true structure!")
        else:
            print(f"   [NOTE] Selected {best_degree} instead of 2 (might be noise)")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    print("\n" + "=" * 50)
    print("Testing complete!")


if __name__ == "__main__":
    test_implementation()
