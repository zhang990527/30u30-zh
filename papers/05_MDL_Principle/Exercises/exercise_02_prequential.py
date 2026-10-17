"""
练习 2：预序贯（预测式）MDL
=========================================

预序贯 MDL 根据模型预测未见数据的表现评分。

核心思想：
1. 从少量数据点开始
2. 用这些点拟合模型
3. 在看到下一个点之前先预测
4. 记录意外程度，即对数损失
5. 将该点加入训练数据，继续重复

总意外程度 = 预序贯 MDL 得分

天气预报员类比：
-------------------------------
天气预报员每天作出预测。
根据实际结果带来的意外程度对其评分。
好的预报员较少遇到意外，预序贯得分较低。

相对两段式编码的优点：
- 不必显式指定参数精度
- 自然适用于时间序列，不表示普遍更稳健
- 通过预测表现约束复杂度
"""

import numpy as np
from typing import Tuple, List


def predict_next_point(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_new: float,
    degree: int
) -> Tuple[float, float]:
    """
    用训练数据拟合模型，并预测下一个点。
    
    参数：
        x_train：训练 x 值
        y_train：训练 y 值
        x_new：需要预测的新 x 值
        degree：多项式次数
        
    返回：
        元组：(prediction, estimated_sigma)
        
    示例：
        >>> x_train = np.array([0, 1, 2, 3])
        >>> y_train = np.array([0, 1, 4, 9])  # 近似二次函数
        >>> pred, sigma = predict_next_point(x_train, y_train, 4, 2)
        >>> print(f"预测 y(4) = {pred:.2f} ± {sigma:.2f}")
    """
    # TODO：实现预测
    # 步骤：
    #   1. 用训练数据拟合多项式
    #   2. 计算训练残差以估计 sigma
    #   3. 预测 x_new 对应的 y 值
    
    pass  # 在此编写代码


def code_length_for_point(
    y_actual: float,
    y_predicted: float,
    sigma: float
) -> float:
    """
    计算单个观测的高斯编码代价，单位为比特。
    
    使用高斯编码：
        L = 0.5 * log2(2π σ²) + (y - ŷ)² / (2σ² ln(2))
    
    直观上，y_actual 距 y_predicted 越远，
    结果越令人意外，所需代价越高。
    
    参数：
        y_actual：实际观测值
        y_predicted：模型预测
        sigma：估计标准差
        
    返回：
        以比特为单位的码长
        
    示例：
        >>> # 小误差对应较短编码
        >>> code_length_for_point(10.0, 10.1, 1.0)  # 意外程度低
        >>> # 大误差对应较长编码
        >>> code_length_for_point(10.0, 15.0, 1.0)  # 意外程度高
    """
    # TODO：实现高斯编码代价
    # 提示：sigma 至少取 1e-10，避免除零
    
    pass  # 在此编写代码


def prequential_mdl(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    start_index: int = None
) -> float:
    """
    计算多项式回归的预序贯 MDL 得分。
    
    算法：
    1. 从 i = start_index 到 n：
       a. 用第 0 至 i-1 个点拟合模型
       b. 预测第 i 个点
       c. 根据实际值与预测值计算编码代价
       d. 累加至总和
    
    参数：
        x：输入值，形状为 [n]
        y：目标值，形状为 [n]
        degree：多项式次数
        start_index：开始预测的位置，默认 degree + 2
        
    返回：
        预序贯总编码代价，单位为比特
        
    示例：
        >>> x = np.linspace(0, 10, 50)
        >>> y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.0
        >>> score_1 = prequential_mdl(x, y, 1)  # 欠拟合
        >>> score_2 = prequential_mdl(x, y, 2)  # 预期合适
        >>> score_5 = prequential_mdl(x, y, 5)  # 可能过拟合
        >>> # 预期 score_2 最低
    """
    # TODO：实现预序贯 MDL
    # 步骤：
    #   1. 确定 start_index，至少需要 degree+2 个点
    #   2. 从 start_index 循环到 n
    #   3. 对每个点：拟合、预测、计算编码代价
    #   4. 累加全部编码代价
    
    pass  # 在此编写代码


def prequential_model_selection(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Tuple[int, dict]:
    """
    使用预序贯 MDL 选择最佳多项式次数。
    
    参数：
        x：输入值
        y：目标值
        max_degree：考虑的最高次数
        
    返回：
        元组：(best_degree, scores_dict)
        
    示例：
        >>> x = np.linspace(0, 10, 50)
        >>> y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.0
        >>> best, scores = prequential_model_selection(x, y)
        >>> print(f"最佳次数：{best}")
    """
    # TODO：计算每个次数的预序贯 MDL，返回最佳值
    
    pass  # 在此编写代码


# =============================================================================
# 测试实现
# =============================================================================

def test_implementation():
    """Test your prequential MDL implementation."""
    
    print("Testing Prequential MDL Implementation")
    print("=" * 50)
    
    np.random.seed(42)
    
    # 测试 1：predict_next_point
    print("\n1. Testing predict_next_point()...")
    try:
        x_train = np.array([0, 1, 2, 3, 4])
        y_train = np.array([0, 1, 4, 9, 16])  # 完全平方数
        
        pred, sigma = predict_next_point(x_train, y_train, 5, 2)
        
        # 应预测为接近 25
        assert 20 < pred < 30, f"Expected ~25, got {pred}"
        assert sigma >= 0, "Sigma should be non-negative"
        
        print(f"   Predicting y(5) from [0,1,4,9,16]")
        print(f"   Prediction: {pred:.2f}, Sigma: {sigma:.4f}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 2：code_length_for_point
    print("\n2. Testing code_length_for_point()...")
    try:
        sigma = 1.0
        
        # 小误差
        small_code = code_length_for_point(10.0, 10.1, sigma)
        # 大误差
        large_code = code_length_for_point(10.0, 15.0, sigma)
        
        assert large_code > small_code, "Larger errors should have longer codes"
        assert small_code > 0, "Code length should be positive"
        
        print(f"   Error = 0.1: {small_code:.2f} bits")
        print(f"   Error = 5.0: {large_code:.2f} bits")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 3：prequential_mdl
    print("\n3. Testing prequential_mdl()...")
    try:
        x = np.linspace(0, 10, 50)
        y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.5  # 二次函数
        
        score_1 = prequential_mdl(x, y, 1)
        score_2 = prequential_mdl(x, y, 2)
        score_5 = prequential_mdl(x, y, 5)
        
        print(f"   Prequential MDL (degree 1): {score_1:.2f} bits")
        print(f"   Prequential MDL (degree 2): {score_2:.2f} bits")
        print(f"   Prequential MDL (degree 5): {score_5:.2f} bits")
        
        # 2 次模型预期具有最低得分
        if score_2 < score_1 and score_2 < score_5:
            print("   [ok] PASSED - Degree 2 has lowest score!")
        else:
            print("   [NOTE] Score ordering unexpected (might be noise)")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 4：prequential_model_selection
    print("\n4. Testing prequential_model_selection()...")
    try:
        x = np.linspace(0, 10, 50)
        y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.5
        
        best_degree, scores = prequential_model_selection(x, y, max_degree=6)
        
        print(f"   Scores: {[(d, f'{s:.0f}') for d, s in scores.items()]}")
        print(f"   Selected: degree {best_degree}")
        print(f"   True: degree 2")
        
        if best_degree == 2:
            print("   [ok] PASSED - Found true degree!")
        else:
            print(f"   [NOTE] Selected {best_degree} (may vary with noise)")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 5：与两段式编码比较
    print("\n5. Comparing with Two-Part Code...")
    try:
        # 在结构明显的数据上，两种方法可能一致；不是普遍保证
        x = np.linspace(0, 10, 100)
        y = 5 * np.sin(x) + np.random.randn(100) * 0.5
        
        preq_best, _ = prequential_model_selection(x, y, max_degree=8)
        
        print(f"   Prequential selects: degree {preq_best}")
        print("   (Compare this with two-part code in exercise 1)")
        print("   [ok] Test complete")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    print("\n" + "=" * 50)
    print("Testing complete!")


if __name__ == "__main__":
    test_implementation()
