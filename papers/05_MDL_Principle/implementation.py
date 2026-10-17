"""
第 5 天：最小描述长度原理——实现
=====================================================================

基于 MDL 的模型选择教学实现，包含：
- 两段式编码（粗略 MDL）
- 预序贯（预测式）MDL
- 归一化最大似然（NML）的教学近似
- 多项式与线性模型的实用模型选择

论文：《最小描述长度原理导论》（A Tutorial Introduction to the Minimum Description Length Principle）
        作者：Peter Grünwald（2004）

可将其理解为寻找数据较短描述的工具库。
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings


# =============================================================================
# 第 1 部分：核心 MDL 计算
# =============================================================================

def log2(x: float) -> float:
    """Safe log base 2, returns 0 for x <= 0."""
    if x <= 0:
        return 0.0
    return np.log2(x)


def universal_code_integers(n: int) -> float:
    """
    正整数的通用先验码长。
    
    当预先不知道整数取值范围时，使用这种
    通用编码；用于描述模型参数。
    
    L(n) = log*(n) + log(c₀)
    其中 log*(n) = log(n) + log(log(n)) + ...，只保留正项
    
    编码类比：
    ================
    假设要向接收方传达一个取值范围未知的整数。
    不能只用 4 比特，因为那隐含最大值为 15。
    
    通用编码的思路是：“先告诉我有多少位，再告诉我这个数。”
    这可以处理任意大小的正整数。
    
    参数：
        n：需要编码的正整数
        
    返回：
        以比特为单位的码长
    """
    if n <= 0:
        return float('inf')
    
    # 迭代对数和：重复取对数并累加正项
    c0 = 2.865064  # 用于有效前缀码的归一化常数
    length = log2(c0)
    
    current = float(n)
    while current > 1:
        length += log2(current)
        current = log2(current)
        if current <= 0:
            break
    
    return max(length, 0)


def precision_code_length(precision_bits: int = 32) -> float:
    """
    指定实数精度所需的码长。
    
    编码实数时，必须指定精度。
    精度越高，通常需要更多比特。
    
    参数：
        precision_bits：每个浮点数使用的比特数
        
    返回：
        描述精度所需的比特数
    """
    return universal_code_integers(precision_bits)


def real_number_code_length(value: float, precision_bits: int = 32) -> float:
    """
    编码单个实数所需的码长。
    
    需要编码：
    1. 符号（1 比特）
    2. 指数（使用通用整数编码）
    3. 尾数（precision_bits）
    
    参数：
        value：要编码的实数
        precision_bits：尾数位数
        
    返回：
        总码长，单位为比特
    """
    if value == 0:
        return 1  # 仅编码“零”标志
    
    sign_bit = 1
    
    # 指数编码
    if abs(value) >= 1:
        exponent = int(np.floor(np.log2(abs(value)))) + 1
    else:
        exponent = -int(np.ceil(-np.log2(abs(value))))
    
    exponent_length = universal_code_integers(abs(exponent) + 1) + 1  # 加 1 表示符号位
    
    return sign_bit + exponent_length + precision_bits


# =============================================================================
# 第 2 部分：两段式编码 MDL
# =============================================================================

def two_part_mdl_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    precision_bits: int = 32
) -> Tuple[float, float, float]:
    """
    计算多项式回归的两段式 MDL 得分。
    
    两段式编码 = L(Model) + L(Data|Model)
    
    编码类比（原仓库称论文第 3 节，实际粗略 MDL 对应 §2.4）：
    ==========================================
    向接收方发送温度读数。
    
    L(Model)：发送多项式公式的代价
              “使用：T = 23.5 + 0.02*hour - 0.001*hour²”
              
    L(Data|Model)：发送偏差的代价
                   “实际读数相对公式的偏差为：+0.1、-0.3、……”
    
    最佳多项式使总消息长度最小。
    
    参数：
        x：输入值，形状为 [n]
        y：目标值，形状为 [n]
        degree：多项式次数，0 为常数，1 为直线，依此类推
        precision_bits：每个浮点系数的比特数
        
    返回：
        元组：(total_mdl, model_cost, data_cost)
    """
    n = len(x)
    
    # 拟合多项式
    try:
        coefficients = np.polyfit(x, y, degree)
        poly = np.poly1d(coefficients)
        predictions = poly(x)
        residuals = y - predictions
    except np.linalg.LinAlgError:
        return float('inf'), float('inf'), 0.0
    
    # =========================================================================
    # 第一段：模型描述长度 L(H)
    # =========================================================================
    
    # 编码次数的代价（使用哪个多项式？）
    degree_cost = universal_code_integers(degree + 1)
    
    # 编码系数的代价
    num_params = degree + 1
    param_cost = num_params * (precision_bits + 1)  # 加 1 表示符号位
    
    # 编码精度的代价
    precision_cost = precision_code_length(precision_bits)
    
    model_cost = degree_cost + param_cost + precision_cost
    
    # =========================================================================
    # 第二段：数据描述长度 L(D|H)
    # =========================================================================
    
    # 使用高斯模型编码残差
    # 原注释给出的每残差编码表达式如下，常数项与代码需分别核对：
    # log(σ√(2πe)) + (residual²)/(2σ²ln2) bits
    
    sigma = np.std(residuals) if np.std(residuals) > 0 else 1e-10
    
    # 简化形式：转换对数似然
    # bits = -log₂(P(data|model))
    
    if sigma > 0:
        # 高斯对数似然
        nll = 0.5 * n * np.log2(2 * np.pi * sigma**2) + \
              np.sum(residuals**2) / (2 * sigma**2 * np.log(2))
        
        # 加入编码 sigma 的代价
        sigma_cost = real_number_code_length(sigma, precision_bits)
        data_cost = nll + sigma_cost
    else:
        data_cost = 0  # 完美拟合，较少见
    
    total_mdl = model_cost + data_cost
    
    return total_mdl, model_cost, data_cost


def select_polynomial_degree(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
    precision_bits: int = 32
) -> Dict[str, Any]:
    """
    使用两段式 MDL 选择最佳多项式次数。
    
    这是可直接使用的多项式模型选择函数。
    
    示例：
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y = 3 + 2*x - 0.5*x**2 + np.random.randn(100) * 0.5  # 真实次数：2
    >>> result = select_polynomial_degree(x, y)
    >>> print(f"最佳次数：{result['best_degree']}")  # 预期为 2

    参数：
        x：输入值
        y：目标值
        max_degree：考虑的最高次数
        precision_bits：每个系数的比特数

    返回：
        字典，包含：
        - best_degree：最佳多项式次数
        - scores：次数到 MDL 得分的映射
        - coefficients：最佳模型的拟合系数
        - model_cost：最佳模型的 L(H)
        - data_cost：最佳模型的 L(D|H)
    """
    scores = {}
    details = {}
    
    for degree in range(max_degree + 1):
        total, model, data = two_part_mdl_polynomial(x, y, degree, precision_bits)
        scores[degree] = total
        details[degree] = {'model_cost': model, 'data_cost': data}
    
    best_degree = min(scores, key=scores.get)
    
    # 获取最佳模型的系数
    coefficients = np.polyfit(x, y, best_degree)
    
    return {
        'best_degree': best_degree,
        'scores': scores,
        'details': details,
        'coefficients': coefficients,
        'model_cost': details[best_degree]['model_cost'],
        'data_cost': details[best_degree]['data_cost'],
        'total_mdl': scores[best_degree]
    }


# =============================================================================
# 第 3 部分：预序贯（预测式）MDL
# =============================================================================

def prequential_mdl_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    start_index: int = None
) -> float:
    """
    计算多项式回归的预序贯 MDL 得分。
    
    预序贯：先预测每个点，再用它更新模型。
    
    天气预报员类比：
    ================================
    设想一名天气预报员：
    1. 在看到结果之前预测明天气温
    2. 记录实际值带来的意外程度
    3. 用新观测更新模型
    4. 每天重复
    
    所有天数的总意外程度对应预序贯 MDL 得分
    
    相对于两段式编码，其优点包括（并非普遍更稳健）：
    - 不必显式指定参数精度
    - 通过预测表现处理模型复杂度
    - 自然适用于时间序列与顺序数据
    
    数学形式：
    ===================
    L_prequential = -Σ log₂ P(yᵢ | y₁, ..., yᵢ₋₁, model)
    
    参数：
        x：输入值，形状为 [n]
        y：目标值，形状为 [n]
        degree：多项式次数
        start_index：开始预测的位置，默认 degree + 2
        
    返回：
        预序贯 MDL 得分，单位为比特
    """
    n = len(x)
    
    # 拟合至少需要 degree+1 个点，另需一个待预测点
    if start_index is None:
        start_index = degree + 2
    
    if start_index >= n:
        return float('inf')
    
    total_code_length = 0.0
    
    for i in range(start_index, n):
        # 在第 0 至 i-1 个点上拟合模型
        x_train = x[:i]
        y_train = y[:i]
        
        try:
            coefficients = np.polyfit(x_train, y_train, degree)
            poly = np.poly1d(coefficients)
            
            # 预测第 i 个点
            y_pred = poly(x[i])
            
            # 用训练残差估计方差
            train_pred = poly(x_train)
            residuals = y_train - train_pred
            sigma = np.std(residuals) if np.std(residuals) > 1e-10 else 1e-10
            
            # 该预测的编码代价（高斯模型）
            error = y[i] - y_pred
            code_length = 0.5 * np.log2(2 * np.pi * sigma**2) + \
                          (error**2) / (2 * sigma**2 * np.log(2))
            
            total_code_length += code_length
            
        except (np.linalg.LinAlgError, np.RankWarning):
            total_code_length += 100  # 数值问题的惩罚
    
    return total_code_length


def prequential_model_selection(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict[str, Any]:
    """
    使用预序贯 MDL 选择最佳多项式。
    
    示例：
    --------
    >>> x = np.linspace(0, 5, 50)
    >>> y = np.sin(x) + 0.1 * np.random.randn(50)
    >>> result = prequential_model_selection(x, y)
    >>> print(f"最佳次数：{result['best_degree']}")
    
    参数：
        x：输入值
        y：目标值
        max_degree：尝试的最高次数
        
    返回：
        包含 best_degree、scores 等项的字典
    """
    scores = {}
    
    for degree in range(max_degree + 1):
        scores[degree] = prequential_mdl_polynomial(x, y, degree)
    
    best_degree = min(scores, key=scores.get)
    coefficients = np.polyfit(x, y, best_degree)
    
    return {
        'best_degree': best_degree,
        'scores': scores,
        'coefficients': coefficients,
        'total_mdl': scores[best_degree]
    }


# =============================================================================
# 第 4 部分：归一化最大似然（NML）
# =============================================================================

def nml_complexity_polynomial(
    x: np.ndarray,
    degree: int,
    y_range: Tuple[float, float] = (-100, 100),
    num_samples: int = 1000
) -> float:
    """
    估计多项式模型的 NML 参数复杂度（原文误称随机复杂度）。
    
    NML 参数复杂度反映模型能很好解释多少不同数据集。
    
    餐厅类比：
    =======================
    模型复杂度就像餐厅菜单：
    - 简单模型（1 次）：只能做“直线”，菜单很小
    - 复杂模型（9 次）：能拟合许多形状，菜单很大
    
    NML 的类比：“菜单越大，对点什么菜的不确定性越大。”
    
    复杂模型因能解释太多不同情形而受到惩罚，
    碰巧拟合当前数据就不那么令人意外。
    
    数学形式：
    ==================
    COMP(M) = log Σ_{z∈Z} P(z | θ̂(z), M)
    
    其中 θ̂(z) 为每个可能数据集 z 的最大似然估计。
    
    这里采用蒙特卡洛教学近似；未包含完整积分体积校正，不能视为严格 NML。
    
    参数：
        x：输入值，即固定设计点
        degree：多项式次数
        y_range：可能 y 值的范围
        num_samples：蒙特卡洛样本数
        
    返回：
        估计的 NML 参数复杂度，单位为比特
    """
    n = len(x)
    
    max_log_likelihood = []
    
    for _ in range(num_samples):
        # 采样随机 y 值
        y_sample = np.random.uniform(y_range[0], y_range[1], n)
        
        # 拟合多项式（最大似然）
        try:
            coefficients = np.polyfit(x, y_sample, min(degree, n - 1))
            poly = np.poly1d(coefficients)
            predictions = poly(x)
            
            # 在高斯噪声假设下计算最大似然
            residuals = y_sample - predictions
            sigma_ml = np.sqrt(np.mean(residuals**2)) if np.any(residuals) else 1e-10
            sigma_ml = max(sigma_ml, 1e-10)
            
            # 最大似然估计处的对数似然
            log_lik = -0.5 * n * np.log(2 * np.pi * sigma_ml**2) - \
                      np.sum(residuals**2) / (2 * sigma_ml**2)
            
            max_log_likelihood.append(log_lik)
            
        except:
            continue
    
    if not max_log_likelihood:
        return 0.0
    
    # NML 复杂度近似：平均最大似然的对数（原有实现缺少积分体积校正）
    # log(1/N * Σ exp(log_lik)) ≈ logsumexp(log_liks) - log(N)
    max_ll = np.max(max_log_likelihood)
    complexity = max_ll + np.log(np.mean(np.exp(np.array(max_log_likelihood) - max_ll)))
    
    return complexity / np.log(2)  # 转换为比特


def nml_mdl_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    y_range: Tuple[float, float] = None
) -> Tuple[float, float, float]:
    """
    计算多项式回归的 NML 近似 MDL 得分。
    
    NML MDL = -log P(data | ML params) + COMP(model)
    
    参数：
        x：输入值
        y：目标值
        degree：多项式次数
        y_range：复杂度估计范围，None 时自动确定
        
    返回：
        元组：(total_nml, fit_term, complexity_term)
    """
    n = len(x)
    
    if y_range is None:
        margin = 2 * (np.max(y) - np.min(y))
        y_range = (np.min(y) - margin, np.max(y) + margin)
    
    # 拟合多项式
    try:
        coefficients = np.polyfit(x, y, degree)
        poly = np.poly1d(coefficients)
        predictions = poly(x)
        residuals = y - predictions
    except:
        return float('inf'), float('inf'), 0.0
    
    # 最大似然拟合项（负对数似然，单位为比特）
    sigma_ml = np.sqrt(np.mean(residuals**2)) if np.any(residuals) else 1e-10
    sigma_ml = max(sigma_ml, 1e-10)
    
    fit_term = (0.5 * n * np.log2(2 * np.pi * sigma_ml**2) + 
                np.sum(residuals**2) / (2 * sigma_ml**2 * np.log(2)))
    
    # 复杂度惩罚
    complexity = nml_complexity_polynomial(x, degree, y_range, num_samples=500)
    
    total = fit_term + complexity
    
    return total, fit_term, complexity


# =============================================================================
# 第 5 部分：通用 MDL 模型类
# =============================================================================

@dataclass
class MDLModel(ABC):
    """
    兼容 MDL 的模型抽象基类。
    
    创建自定义 MDL 模型的方法：
    1. 继承 MDLModel
    2. 实现 fit()、predict()、model_code_length()、data_code_length()
    
    示例：
    --------
    class MyCustomModel(MDLModel):
        def fit(self, x, y): ...
        def predict(self, x): ...
        def model_code_length(self): ...
        def data_code_length(self, x, y): ...
    """
    
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit model to data."""
        pass
    
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def model_code_length(self) -> float:
        """Return L(H) - bits to describe the model."""
        pass
    
    @abstractmethod
    def data_code_length(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return L(D|H) - bits to describe data given model."""
        pass
    
    def total_mdl(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return total MDL = L(H) + L(D|H)."""
        return self.model_code_length() + self.data_code_length(x, y)


class PolynomialMDL(MDLModel):
    """
    带 MDL 评分的多项式模型。
    
    示例：
    --------
    >>> model = PolynomialMDL(degree=2)
    >>> model.fit(x, y)
    >>> score = model.total_mdl(x, y)
    >>> predictions = model.predict(x_test)
    """
    
    def __init__(self, degree: int, precision_bits: int = 32):
        self.degree = degree
        self.precision_bits = precision_bits
        self.coefficients = None
        self.sigma = None
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit polynomial to data."""
        self.coefficients = np.polyfit(x, y, self.degree)
        self.poly = np.poly1d(self.coefficients)
        residuals = y - self.poly(x)
        self.sigma = np.std(residuals) if np.std(residuals) > 0 else 1e-10
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict y values."""
        if self.coefficients is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.poly(x)
    
    def model_code_length(self) -> float:
        """Bits to describe polynomial model."""
        if self.coefficients is None:
            return float('inf')
        
        # 次数描述
        degree_cost = universal_code_integers(self.degree + 1)
        
        # 系数
        num_params = len(self.coefficients)
        param_cost = num_params * (self.precision_bits + 1)
        
        return degree_cost + param_cost
    
    def data_code_length(self, x: np.ndarray, y: np.ndarray) -> float:
        """Bits to describe residuals."""
        if self.coefficients is None:
            self.fit(x, y)
        
        predictions = self.predict(x)
        residuals = y - predictions
        n = len(y)
        
        # 高斯编码
        nll = 0.5 * n * np.log2(2 * np.pi * self.sigma**2) + \
              np.sum(residuals**2) / (2 * self.sigma**2 * np.log(2))
        
        # 加入 sigma 编码代价
        sigma_cost = real_number_code_length(self.sigma, self.precision_bits)
        
        return nll + sigma_cost


class GaussianMDL(MDLModel):
    """
    带 MDL 评分的简单高斯模型。
    
    将数据建模为 y ~ N(μ, σ²)。
    
    可用作基线：若更复杂模型无法胜过它，
    说明当前候选和编码下尚未获得更强结构证据，并非证明数据毫无结构。
    """
    
    def __init__(self, precision_bits: int = 32):
        self.precision_bits = precision_bits
        self.mean = None
        self.std = None
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit Gaussian to y values (ignores x)."""
        self.mean = np.mean(y)
        self.std = np.std(y) if np.std(y) > 0 else 1e-10
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict mean everywhere."""
        return np.full(len(x), self.mean)
    
    def model_code_length(self) -> float:
        """Bits to describe μ and σ."""
        if self.mean is None:
            return float('inf')
        
        return (real_number_code_length(self.mean, self.precision_bits) +
                real_number_code_length(self.std, self.precision_bits))
    
    def data_code_length(self, x: np.ndarray, y: np.ndarray) -> float:
        """Bits to describe deviations from mean."""
        if self.mean is None:
            self.fit(x, y)
        
        n = len(y)
        nll = 0.5 * n * np.log2(2 * np.pi * self.std**2) + \
              np.sum((y - self.mean)**2) / (2 * self.std**2 * np.log(2))
        
        return nll


# =============================================================================
# 第 6 部分：与 AIC/BIC 比较
# =============================================================================

def aic_score(
    x: np.ndarray,
    y: np.ndarray,
    degree: int
) -> float:
    """
    计算赤池信息准则（AIC）得分。
    
    AIC = 2k - 2ln(L)
    
    其中 k 为参数数量，L 为最大似然。
    
    越低越好。
    """
    n = len(x)
    k = degree + 2  # 多项式系数与方差
    
    try:
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        residuals = y - poly(x)
        rss = np.sum(residuals**2)
        
        # 高斯对数似然
        sigma_ml = np.sqrt(rss / n)
        if sigma_ml <= 0:
            sigma_ml = 1e-10
        
        log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma_ml**2) + 1)
        
        return 2 * k - 2 * log_likelihood
        
    except:
        return float('inf')


def bic_score(
    x: np.ndarray,
    y: np.ndarray,
    degree: int
) -> float:
    """
    计算贝叶斯信息准则（BIC）得分。
    
    BIC = k*ln(n) - 2ln(L)
    
    其中 k 为参数数量，n 为样本数，L 为最大似然。
    
    越低越好。n 较大时，BIC 比 AIC 的参数惩罚更强。
    """
    n = len(x)
    k = degree + 2  # 多项式系数与方差
    
    try:
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        residuals = y - poly(x)
        rss = np.sum(residuals**2)
        
        # 高斯对数似然
        sigma_ml = np.sqrt(rss / n)
        if sigma_ml <= 0:
            sigma_ml = 1e-10
        
        log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma_ml**2) + 1)
        
        return k * np.log(n) - 2 * log_likelihood
        
    except:
        return float('inf')


def compare_mdl_aic_bic(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict[str, Any]:
    """
    比较 MDL、AIC、BIC 的多项式模型选择。
    
    使用此函数观察不同准则之间的差异。
    
    示例：
    --------
    >>> x = np.linspace(0, 10, 50)
    >>> y = 2 + 3*x - 0.5*x**2 + np.random.randn(50) * 2  # 真实次数：2
    >>> results = compare_mdl_aic_bic(x, y)
    >>> print(f"MDL 选择：{results['mdl_best']}")
    >>> print(f"AIC 选择：{results['aic_best']}")
    >>> print(f"BIC 选择：{results['bic_best']}")
    
    返回：
        包含各准则得分与最佳选择的字典。
    """
    mdl_scores = {}
    aic_scores = {}
    bic_scores = {}
    
    for degree in range(max_degree + 1):
        mdl_scores[degree] = two_part_mdl_polynomial(x, y, degree)[0]
        aic_scores[degree] = aic_score(x, y, degree)
        bic_scores[degree] = bic_score(x, y, degree)
    
    return {
        'mdl_scores': mdl_scores,
        'aic_scores': aic_scores,
        'bic_scores': bic_scores,
        'mdl_best': min(mdl_scores, key=mdl_scores.get),
        'aic_best': min(aic_scores, key=aic_scores.get),
        'bic_best': min(bic_scores, key=bic_scores.get)
    }


# =============================================================================
# 第 7 部分：工具函数
# =============================================================================

def compression_ratio(
    data: np.ndarray,
    model_mdl: float
) -> float:
    """
    计算 MDL 模型的启发式压缩比。
    
    比值 > 1：估计的模型描述短于原始数据参照
    比值 < 1：估计的描述比原始参照更长
    比值 = 1：未估计出压缩收益，不能据此证明数据随机
    
    参数：
        data：原始数据数组
        model_mdl：模型 MDL 得分
        
    返回：
        压缩比估计
    """
    # 原始数据参照代价：每个元素按 float32 计
    raw_bits = len(data) * 32
    
    return raw_bits / model_mdl if model_mdl > 0 else 0


def mdl_model_probability(
    mdl_scores: Dict[int, float]
) -> Dict[int, float]:
    """
    将 MDL 得分转换为归一化码长权重；不普遍等同于贝叶斯后验概率。
    
    使用：P(model) ∝ 2^(-MDL score)
    
    用于直观比较模型之间的相对码长。
    
    参数：
        mdl_scores：模型 ID 到 MDL 得分的映射
        
    返回：
        模型 ID 到概率权重的映射
    """
    # 归一化以防溢出
    min_score = min(mdl_scores.values())
    normalized = {k: -(v - min_score) for k, v in mdl_scores.items()}
    
    # 转换为概率权重
    powers = {k: 2**v for k, v in normalized.items()}
    total = sum(powers.values())
    
    return {k: v / total for k, v in powers.items()}


def generate_test_data(
    true_degree: int = 2,
    n_points: int = 50,
    noise_std: float = 1.0,
    x_range: Tuple[float, float] = (0, 10),
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成用于测试的合成多项式数据。
    
    参数：
        true_degree：真实多项式次数
        n_points：数据点数量
        noise_std：高斯噪声标准差
        x_range：x 的取值范围
        seed：用于复现的随机种子
        
    返回：
        元组：(x, y_noisy, true_coefficients)
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    # 生成随机系数
    true_coeffs = np.random.randn(true_degree + 1) * 2
    poly = np.poly1d(true_coeffs)
    
    y_true = poly(x)
    y_noisy = y_true + np.random.randn(n_points) * noise_std
    
    return x, y_noisy, true_coeffs


# =============================================================================
# 第 8 部分：主程序演示
# =============================================================================

def main():
    """
    在合成数据上演示 MDL 模型选择。
    """
    print("=" * 70)
    print("MDL (Minimum Description Length) Principle - Implementation Demo")
    print("=" * 70)
    print()
    
    # 生成测试数据
    np.random.seed(42)
    true_degree = 2
    x, y, true_coeffs = generate_test_data(
        true_degree=true_degree,
        n_points=50,
        noise_std=1.5,
        seed=42
    )
    
    print(f"Generated data with TRUE degree = {true_degree}")
    print(f"True coefficients: {true_coeffs}")
    print(f"Number of points: {len(x)}")
    print()
    
    # =========================================================================
    # 两段式编码 MDL
    # =========================================================================
    print("-" * 50)
    print("METHOD 1: Two-Part Code MDL")
    print("-" * 50)
    
    result = select_polynomial_degree(x, y, max_degree=10)
    
    print(f"\nMDL Scores by Degree:")
    for deg in range(6):
        score = result['scores'][deg]
        marker = " <-- BEST" if deg == result['best_degree'] else ""
        print(f"  Degree {deg}: {score:.2f} bits{marker}")
    
    print(f"\nTwo-Part MDL selects: Degree {result['best_degree']}")
    print(f"  Model cost: {result['model_cost']:.2f} bits")
    print(f"  Data cost: {result['data_cost']:.2f} bits")
    print(f"  Total: {result['total_mdl']:.2f} bits")
    
    # =========================================================================
    # 预序贯 MDL
    # =========================================================================
    print()
    print("-" * 50)
    print("METHOD 2: Prequential MDL")
    print("-" * 50)
    
    preq_result = prequential_model_selection(x, y, max_degree=10)
    
    print(f"\nPrequential MDL Scores:")
    for deg in range(6):
        score = preq_result['scores'][deg]
        marker = " <-- BEST" if deg == preq_result['best_degree'] else ""
        print(f"  Degree {deg}: {score:.2f} bits{marker}")
    
    print(f"\nPrequential MDL selects: Degree {preq_result['best_degree']}")
    
    # =========================================================================
    # 与 AIC、BIC 比较
    # =========================================================================
    print()
    print("-" * 50)
    print("COMPARISON: MDL vs AIC vs BIC")
    print("-" * 50)
    
    comparison = compare_mdl_aic_bic(x, y, max_degree=10)
    
    print(f"\nBest degree selected by each criterion:")
    print(f"  MDL: Degree {comparison['mdl_best']}")
    print(f"  AIC: Degree {comparison['aic_best']}")
    print(f"  BIC: Degree {comparison['bic_best']}")
    print(f"  True: Degree {true_degree}")
    
    # =========================================================================
    # 模型概率权重
    # =========================================================================
    print()
    print("-" * 50)
    print("MDL MODEL PROBABILITIES")
    print("-" * 50)
    
    probs = mdl_model_probability(result['scores'])
    
    print("\nPosterior probability of each degree:")
    for deg in range(6):
        bar = "█" * int(probs[deg] * 50)
        print(f"  Degree {deg}: {probs[deg]:.4f} {bar}")
    
    # =========================================================================
    # 压缩分析
    # =========================================================================
    print()
    print("-" * 50)
    print("COMPRESSION ANALYSIS")
    print("-" * 50)
    
    raw_bits = len(y) * 32
    best_mdl = result['total_mdl']
    ratio = compression_ratio(y, best_mdl)
    
    print(f"\nRaw data size: {raw_bits} bits ({len(y)} × 32-bit floats)")
    print(f"MDL compressed: {best_mdl:.0f} bits")
    print(f"Compression ratio: {ratio:.2f}x")
    print(f"\n{'[ok] Found compressible structure!' if ratio > 1 else '[FAIL] Data appears random.'}")
    
    print()
    print("=" * 70)
    print("Demo complete! Run visualization.py for plots.")
    print("=" * 70)


if __name__ == "__main__":
    main()
