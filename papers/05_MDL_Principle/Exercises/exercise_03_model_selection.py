"""
练习 3：使用 MDL 进行实用模型选择
===============================================

本练习构建一个完整的 MDL 模型选择系统。

目标：给定含噪数据，尝试寻找底层真实模式。

实际应用：
- 选择神经网络结构
- 选择回归特征
- 确定聚类数量
- 为语言建模选择语法

侦探类比：
---------------------
一名侦探对案件有多种解释。
每种解释有“代价”（复杂程度），
也有“解释力”（与证据的吻合程度）。

MDL 寻找两者最合适的平衡。
"""

import numpy as np
from typing import Dict, Tuple, List, Optional


class MDLModelSelector:
    """
    基于 MDL 的多项式模型选择类。
    
    封装完整工作流程：
    1. 计算不同模型的 MDL 得分
    2. 选择最佳模型
    3. 分析权衡
    4. 进行预测
    """
    
    def __init__(self, max_degree: int = 10, precision_bits: int = 32):
        """
        初始化模型选择器。
        
        参数：
            max_degree：考虑的最高多项式次数
            precision_bits：编码系数的比特数
        """
        self.max_degree = max_degree
        self.precision_bits = precision_bits
        self.scores: Dict[int, float] = {}
        self.model_costs: Dict[int, float] = {}
        self.data_costs: Dict[int, float] = {}
        self.best_degree: Optional[int] = None
        self.best_coefficients: Optional[np.ndarray] = None
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> 'MDLModelSelector':
        """
        用数据拟合模型选择器。
        
        计算从 0 到 max_degree 的所有次数的 MDL 得分，
        然后选择总 MDL 最低的次数。
        
        参数：
            x：输入值
            y：目标值
            
        返回：
            self，用于链式调用
            
        示例：
            >>> selector = MDLModelSelector(max_degree=10)
            >>> selector.fit(x, y)
            >>> print(f"最佳次数：{selector.best_degree}")
        """
        # TODO：实现拟合过程
        # 步骤：
        #   1. 遍历 0 至 max_degree 的次数
        #   2. 为每个次数计算 MDL，即模型代价与数据代价之和
        #   3. 将得分等保存至 self.scores
        #   4. 找出总 MDL 最小的 best_degree
        #   5. 拟合最终模型并存储系数
        
        pass  # 在此编写代码
        
        return self
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        使用选定模型进行预测。
        
        参数：
            x：需要预测的输入值
            
        返回：
            预测的 y 值
            
        异常：
            ValueError：尚未调用 fit()
        """
        # TODO：实现预测
        
        pass  # 在此编写代码
    
    def get_compression_ratio(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算压缩比估计。
        
        压缩比 = 原始数据参照比特数 / MDL 得分
        
        比值 > 1 表示估计有压缩收益；不是实际文件压缩验证。
        
        参数：
            x：输入值
            y：目标值
            
        返回：
            压缩比估计
        """
        # TODO：实现压缩比估计计算
        # 原始比特数：len(y) * 32，假设使用 32 比特浮点数
        # MDL 得分：self.scores[self.best_degree]
        
        pass  # 在此编写代码
    
    def get_model_probability(self, degree: int) -> float:
        """
        获取给定次数的近似模型概率权重。
        
        使用：P(model) ∝ 2^(-MDL score)
        
        参数：
            degree：多项式次数
            
        返回：
            近似概率权重，所有次数之和为 1
        """
        # TODO：将 MDL 得分转换为概率权重
        # 提示：先减去最小得分再归一化，以防溢出
        
        pass  # 在此编写代码
    
    def summary(self) -> str:
        """
        返回模型选择结果摘要。
        """
        if self.best_degree is None:
            return "Model not fitted. Call fit() first."
        
        lines = [
            "MDL Model Selection Summary",
            "=" * 40,
            f"Best degree: {self.best_degree}",
            f"Model cost: {self.model_costs.get(self.best_degree, 0):.1f} bits",
            f"Data cost: {self.data_costs.get(self.best_degree, 0):.1f} bits",
            f"Total MDL: {self.scores.get(self.best_degree, 0):.1f} bits",
            "",
            "All scores:",
        ]
        
        for deg in sorted(self.scores.keys()):
            marker = " ← BEST" if deg == self.best_degree else ""
            lines.append(f"  Degree {deg}: {self.scores[deg]:.1f}{marker}")
        
        return "\n".join(lines)


def compare_with_cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
    n_folds: int = 5
) -> Dict[str, int]:
    """
    比较 MDL 选择与 k 折交叉验证。
    
    观察 MDL 是否与交叉验证一致，以及计算开销差异；不是普遍保证。
    
    参数：
        x：输入值
        y：目标值
        max_degree：尝试的最高次数
        n_folds：交叉验证折数
        
    返回：
        包含 mdl_best 与 cv_best 次数的字典
        
    示例：
        >>> result = compare_with_cross_validation(x, y)
        >>> print(f"MDL：{result['mdl_best']}，交叉验证：{result['cv_best']}")
    """
    # TODO：实现比较
    # 步骤：
    #   1. 使用 MDLModelSelector 找出最佳次数
    #   2. 实现 k 折交叉验证，找出最佳次数
    #   3. 返回两者
    
    pass  # 在此编写代码


def monte_carlo_evaluation(
    true_degree: int = 2,
    n_samples: int = 50,
    noise_std: float = 1.5,
    n_trials: int = 100
) -> Dict[str, float]:
    """
    通过多次随机试验评估 MDL 模型选择。
    
    检查 MDL 找到真实模型的可靠程度。
    
    参数：
        true_degree：实际多项式次数
        n_samples：每次试验的数据点数量
        noise_std：噪声标准差
        n_trials：随机试验次数
        
    返回：
        包含选中率统计的字典
    """
    # TODO：实现蒙特卡洛评估
    # 步骤：
    #   1. 对每次试验：
    #      a. 生成 true_degree 次随机多项式
    #      b. 添加噪声
    #      c. 使用 MDLModelSelector 找出最佳次数
    #      d. 检查是否匹配 true_degree
    #   2. 计算选中率
    
    pass  # 在此编写代码


# =============================================================================
# 辅助函数（实现后类才能正常工作）
# =============================================================================

def universal_code_length(n: int) -> float:
    """Universal code length for positive integers."""
    # TODO：复制练习 1 或自行实现
    pass


def compute_mdl(x: np.ndarray, y: np.ndarray, degree: int, 
                precision: int = 32) -> Tuple[float, float, float]:
    """
    计算两段式 MDL。
    
    返回：(total, model_cost, data_cost)
    """
    # TODO：复制练习 1 或自行实现
    pass


# =============================================================================
# 测试实现
# =============================================================================

def test_implementation():
    """Test the MDL model selector."""
    
    print("Testing MDL Model Selection System")
    print("=" * 50)
    
    np.random.seed(42)
    
    # 生成测试数据
    x = np.linspace(0, 10, 50)
    true_degree = 2
    true_coeffs = [-0.5, 2, 3]  # -0.5x² + 2x + 3
    y_true = 3 + 2*x - 0.5*x**2
    y = y_true + np.random.randn(50) * 1.5
    
    print(f"\nTest data: n={len(x)}, true degree={true_degree}")
    
    # 测试 1：基本拟合
    print("\n1. Testing MDLModelSelector.fit()...")
    try:
        selector = MDLModelSelector(max_degree=8)
        selector.fit(x, y)
        
        assert selector.best_degree is not None, "best_degree should be set"
        assert len(selector.scores) > 0, "scores should be populated"
        
        print(f"   Best degree selected: {selector.best_degree}")
        print(f"   True degree: {true_degree}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 2：预测
    print("\n2. Testing MDLModelSelector.predict()...")
    try:
        x_test = np.array([11, 12, 13])
        predictions = selector.predict(x_test)
        
        assert len(predictions) == len(x_test), "Wrong number of predictions"
        print(f"   Predictions for x=[11,12,13]: {predictions}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 3：压缩比估计
    print("\n3. Testing get_compression_ratio()...")
    try:
        ratio = selector.get_compression_ratio(x, y)
        
        assert ratio > 0, "Compression ratio should be positive"
        print(f"   Compression ratio: {ratio:.2f}x")
        
        if ratio > 1:
            print("   [ok] Data is compressible - structure found!")
        else:
            print("   [NOTE] No compression - data might be random")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 4：模型概率权重
    print("\n4. Testing get_model_probability()...")
    try:
        total_prob = sum(selector.get_model_probability(d) for d in range(9))
        
        assert abs(total_prob - 1.0) < 0.01, "Probabilities should sum to 1"
        
        print(f"   P(degree=2): {selector.get_model_probability(2):.4f}")
        print(f"   Sum of all probabilities: {total_prob:.4f}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 5：摘要
    print("\n5. Testing summary()...")
    try:
        summary = selector.summary()
        assert "Best degree" in summary, "Summary should contain best degree"
        print("   Summary generated successfully")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")
    
    # 测试 6：与交叉验证比较，若已实现
    print("\n6. Testing compare_with_cross_validation()...")
    try:
        result = compare_with_cross_validation(x, y, max_degree=6)
        print(f"   MDL selects: degree {result['mdl_best']}")
        print(f"   CV selects: degree {result['cv_best']}")
        
        if result['mdl_best'] == result['cv_best']:
            print("   [ok] MDL and CV agree!")
        else:
            print("   [NOTE] Methods disagree (happens sometimes)")
    except Exception as e:
        print(f"   [NOTE] Skipped (not implemented): {e}")
    
    print("\n" + "=" * 50)
    print("Testing complete!")
    print("\nFull summary:")
    print(selector.summary())


if __name__ == "__main__":
    test_implementation()
