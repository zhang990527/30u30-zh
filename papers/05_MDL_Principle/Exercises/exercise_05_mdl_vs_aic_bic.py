"""
练习 5：MDL、AIC 与 BIC 的综合比较
======================================================

三种模型选择准则，比较其识别真实模型的表现；目标依准则而异。

MDL（最小描述长度）：
    Score = L(Model) + L(Data|Model)
    → 基于信息论的惩罚
    → 惩罚由编码定义，但编码选择仍有约定

AIC（赤池信息准则）：
    Score = 2k - 2ln(L)
    → 每个参数惩罚 2
    → 来自预测风险估计

BIC（贝叶斯信息准则）：
    Score = k·ln(n) - 2ln(L)
    → 每个参数惩罚 ln(n)
    → 来自贝叶斯模型选择近似

关键问题：
-----------------
这些方法何时一致、何时分歧？
应该依据什么选择方法？

答案取决于目标与适用条件！
- AIC：侧重预测风险，不保证所有预测任务都最佳
- BIC：在适当条件下具有真模型选择一致性
- MDL：侧重编码压缩，不等于语义理解保证
"""

import numpy as np
from typing import Dict, Tuple, List


def compute_aic(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    """
    计算 AIC（赤池信息准则）。
    
    AIC = 2k - 2ln(L)
    
    其中 k 为参数数量，L 为最大似然
    
    越低越好！
    
    参数：
        x：输入值
        y：目标值
        degree：多项式次数
        
    返回：
        AIC 得分，越低越好
        
    示例：
        >>> x = np.linspace(0, 10, 50)
        >>> y = 2*x + 1 + np.random.randn(50) * 0.5
        >>> aic_1 = compute_aic(x, y, 1)  # 预期较低，拟合好
        >>> aic_5 = compute_aic(x, y, 5)  # 可能较高，存在过拟合
    """
    # TODO：实现 AIC
    # 步骤：
    #   1. k = degree + 2，即多项式系数与方差
    #   2. 拟合多项式
    #   3. 计算 sigma 的最大似然估计：sqrt(RSS/n)
    #   4. 计算对数似然：-n/2 * (log(2π σ²) + 1)
    #   5. 返回 2k - 2*log_likelihood

    pass  # 在此编写代码


def compute_bic(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    """
    计算 BIC（贝叶斯信息准则）。
    
    BIC = k·ln(n) - 2ln(L)
    
    样本较大时，BIC 的参数惩罚强于 AIC。
    
    参数：
        x：输入值
        y：目标值
        degree：多项式次数
        
    返回：
        BIC 得分，越低越好
    """
    # TODO：实现 BIC
    # 与 AIC 类似，但惩罚是 k * log(n)，而非 2k
    
    pass  # 在此编写代码


def compute_mdl(x: np.ndarray, y: np.ndarray, degree: int,
                precision: int = 32) -> float:
    """
    计算两段式 MDL 得分。
    
    MDL = L(Model) + L(Data|Model)
    
    参数：
        x：输入值
        y：目标值
        degree：多项式次数
        precision：每个系数的比特数
        
    返回：
        MDL 得分，单位为比特，越低越好
    """
    # TODO：实现 MDL，可复制练习 1 或重新实现
    
    pass  # 在此编写代码


def select_with_all_criteria(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict[str, Dict]:
    """
    运行并比较三种模型选择准则。
    
    参数：
        x：输入值
        y：目标值
        max_degree：考虑的最高次数
        
    返回：
        包含各方法结果的字典：
        {
            'mdl': {'scores': {...}, 'best': int},
            'aic': {'scores': {...}, 'best': int},
            'bic': {'scores': {...}, 'best': int}
        }
    """
    # TODO：计算所有得分，并为各准则找出最佳次数
    
    pass  # 在此编写代码


def monte_carlo_comparison(
    true_degree: int = 2,
    n_samples: int = 50,
    noise_std: float = 1.5,
    n_trials: int = 100
) -> Dict[str, float]:
    """
    通过多次试验比较 MDL、AIC、BIC 的真实次数选中率。
    
    该实验检查在指定生成机制下，哪种方法更常
    找到真实模型；不能证明方法在所有任务上最佳。
    
    参数：
        true_degree：实际多项式次数
        n_samples：每次试验的数据点数
        noise_std：噪声水平
        n_trials：实验次数
        
    返回：
        包含各方法选中率的字典：
        {'mdl_accuracy': 0.85, 'aic_accuracy': 0.72, 'bic_accuracy': 0.88}
    """
    # TODO：实现蒙特卡洛比较
    # 步骤：
    #   1. 对每次试验：
    #      a. 生成 true_degree 次随机多项式
    #      b. 添加噪声
    #      c. 应用各准则
    #      d. 检查是否找到真实次数
    #   2. 计算各准则的选中率

    pass  # 在此编写代码


def analyze_sample_size_effect(
    true_degree: int = 2,
    sample_sizes: List[int] = [20, 50, 100, 200, 500],
    noise_std: float = 1.5,
    n_trials: int = 50
) -> Dict[str, List[float]]:
    """
    分析样本数如何影响各准则的选中率。
    
    关键认识：
    - AIC 在真有限维模型条件下可能持续选择多余参数
    - BIC 在适当条件下随样本数增加更易选择真实模型
    - MDL 的表现需结合编码与样本数实测，不保证总是稳定
    
    返回：
        方法到各样本数下选中率列表的映射
    """
    # TODO：实现样本数分析
    
    pass  # 在此编写代码


def analyze_noise_level_effect(
    true_degree: int = 2,
    n_samples: int = 50,
    noise_levels: List[float] = [0.5, 1.0, 2.0, 4.0],
    n_trials: int = 50
) -> Dict[str, List[float]]:
    """
    分析噪声水平如何影响各准则的选中率。
    
    关键认识：所有方法都可能受高噪声影响，
    稳健性因具体条件而异。
    
    返回：
        方法到各噪声水平下选中率列表的映射
    """
    # TODO：实现噪声水平分析
    
    pass  # 在此编写代码


def when_do_they_disagree(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict[str, any]:
    """
    寻找方法分歧的情形并分析原因。
    
    返回分歧分析，包括：
    - 各方法选择的次数
    - 得分差异
    - 可能解释
    """
    # TODO：实现分歧分析
    
    pass  # 在此编写代码


# =============================================================================
# 辅助函数
# =============================================================================

def generate_polynomial_data(
    degree: int,
    n_samples: int,
    noise_std: float,
    x_range: Tuple[float, float] = (0, 10),
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成合成多项式数据。
    
    返回：
        元组：(x, y_noisy, true_coefficients)
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.linspace(x_range[0], x_range[1], n_samples)
    
    # 随机系数
    true_coeffs = np.random.randn(degree + 1) * 2
    poly = np.poly1d(true_coeffs)
    
    y_true = poly(x)
    y_noisy = y_true + np.random.randn(n_samples) * noise_std
    
    return x, y_noisy, true_coeffs


# =============================================================================
# 测试实现
# =============================================================================

def test_implementation():
    """Test the criterion comparison implementation."""
    
    print("Testing MDL vs AIC vs BIC Comparison")
    print("=" * 55)
    
    np.random.seed(42)
    
    # 生成测试数据
    true_degree = 2
    x, y, true_coeffs = generate_polynomial_data(
        degree=true_degree,
        n_samples=50,
        noise_std=1.5,
        seed=42
    )
    
    print(f"\nTest data: n=50, true degree={true_degree}")
    
    # 测试 1：各个准则
    print("\n1. Testing individual criteria...")
    try:
        aic_2 = compute_aic(x, y, 2)
        bic_2 = compute_bic(x, y, 2)
        mdl_2 = compute_mdl(x, y, 2)
        
        print(f"   AIC(degree=2) = {aic_2:.2f}")
        print(f"   BIC(degree=2) = {bic_2:.2f}")
        print(f"   MDL(degree=2) = {mdl_2:.2f}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 2：完整选择流程
    print("\n2. Testing select_with_all_criteria()...")
    try:
        results = select_with_all_criteria(x, y, max_degree=8)
        
        print(f"   MDL selects: degree {results['mdl']['best']}")
        print(f"   AIC selects: degree {results['aic']['best']}")
        print(f"   BIC selects: degree {results['bic']['best']}")
        print(f"   True degree: {true_degree}")
        
        # 统计选对的准则数量
        correct = sum([
            results['mdl']['best'] == true_degree,
            results['aic']['best'] == true_degree,
            results['bic']['best'] == true_degree
        ])
        print(f"   {correct}/3 methods correct")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 3：蒙特卡洛试验
    print("\n3. Testing monte_carlo_comparison()...")
    try:
        mc_results = monte_carlo_comparison(
            true_degree=2,
            n_samples=50,
            noise_std=1.5,
            n_trials=50
        )
        
        print(f"   MDL accuracy: {mc_results['mdl_accuracy']*100:.1f}%")
        print(f"   AIC accuracy: {mc_results['aic_accuracy']*100:.1f}%")
        print(f"   BIC accuracy: {mc_results['bic_accuracy']*100:.1f}%")
        
        # 找出获胜方法
        winner = max(['mdl', 'aic', 'bic'],
                     key=lambda m: mc_results[f'{m}_accuracy'])
        print(f"   Winner: {winner.upper()}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 4：样本数影响
    print("\n4. Testing analyze_sample_size_effect()...")
    try:
        size_results = analyze_sample_size_effect(
            true_degree=2,
            sample_sizes=[20, 50, 100],
            n_trials=30
        )
        
        for method in ['mdl', 'aic', 'bic']:
            accs = size_results[method]
            print(f"   {method.upper()}: n=20→{accs[0]*100:.0f}%, n=50→{accs[1]*100:.0f}%, n=100→{accs[2]*100:.0f}%")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [NOTE] Skipped: {e}")
    
    # 测试 5：分歧分析
    print("\n5. Testing when_do_they_disagree()...")
    try:
        disagreement = when_do_they_disagree(x, y, max_degree=8)
        
        if disagreement.get('methods_agree', False):
            print("   All methods agree on this dataset")
        else:
            print(f"   Methods disagree!")
            print(f"   Analysis: {disagreement.get('explanation', 'N/A')}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [NOTE] Skipped: {e}")
    
    print("\n" + "=" * 55)
    print("Testing complete!")
    print("\nKey takeaways:")
    print("- AIC: Optimizes prediction, may overfit with large n")
    print("- BIC: Consistent, gets better with more data")
    print("- MDL: Information-theoretic, no arbitrary constants")


if __name__ == "__main__":
    test_implementation()
