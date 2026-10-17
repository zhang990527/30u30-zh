"""
第 5 天最小描述长度练习参考解答
================================================================

注意：请先自己尝试练习！
这些解答供独立尝试后参考。

每份解答包括：
1. 完整代码
2. 方法解释
3. 关键认识
"""

import numpy as np
from typing import Tuple, Dict, List


# =============================================================================
# 练习 1 解答：两段式编码
# =============================================================================

def solution_universal_code_length(n: int) -> float:
    """
    使用迭代对数和编码正整数。
    
    L(n) = log₂(c₀) + log₂(n) + log₂(log₂(n)) + ...
    
    求和中只包含正项。
    """
    if n <= 0:
        return float('inf')
    
    c0 = 2.865064  # 归一化常数
    length = np.log2(c0)
    
    current = float(n)
    while current > 1:
        length += np.log2(current)
        current = np.log2(current)
        if current <= 0:
            break
    
    return max(length, 0)


def solution_model_description_length(degree: int, precision_bits: int = 32) -> float:
    """
    模型代价 = 次数编码 + 系数编码。
    """
    # 编码次数
    degree_cost = solution_universal_code_length(degree + 1)
    
    # 编码系数，共 degree+1 个
    num_coefficients = degree + 1
    coefficient_cost = num_coefficients * (precision_bits + 1)  # 加 1 表示符号位
    
    return degree_cost + coefficient_cost


def solution_data_description_length(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    precision_bits: int = 32
) -> float:
    """
    数据代价 = 使用高斯模型编码残差。
    """
    n = len(x)
    
    # 拟合多项式
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    predictions = poly(x)
    residuals = y - predictions
    
    # 估计 sigma
    sigma = np.std(residuals)
    sigma = max(sigma, 1e-10)  # 避免除零
    
    # 高斯负对数似然，单位为比特
    nll = 0.5 * n * np.log2(2 * np.pi * sigma**2) + \
          np.sum(residuals**2) / (2 * sigma**2 * np.log(2))
    
    # 加入编码 sigma 的代价
    sigma_cost = precision_bits + 10  # 简化形式
    
    return nll + sigma_cost


def solution_two_part_mdl(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    precision_bits: int = 32
) -> Tuple[float, float, float]:
    """
    完整两段式 MDL：L(H) + L(D|H)。
    """
    model_cost = solution_model_description_length(degree, precision_bits)
    data_cost = solution_data_description_length(x, y, degree, precision_bits)
    total = model_cost + data_cost
    
    return total, model_cost, data_cost


# =============================================================================
# 练习 2 解答：预序贯 MDL
# =============================================================================

def solution_predict_next_point(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_new: float,
    degree: int
) -> Tuple[float, float]:
    """
    在训练数据上拟合模型，并预测下一个点。
    """
    # 拟合多项式
    coeffs = np.polyfit(x_train, y_train, degree)
    poly = np.poly1d(coeffs)
    
    # 用训练残差估计 sigma
    train_pred = poly(x_train)
    residuals = y_train - train_pred
    sigma = np.std(residuals) if np.std(residuals) > 1e-10 else 1e-10
    
    # 预测新点
    prediction = poly(x_new)
    
    return prediction, sigma


def solution_code_length_for_point(
    y_actual: float,
    y_predicted: float,
    sigma: float
) -> float:
    """
    单个观测的高斯编码代价。
    """
    sigma = max(sigma, 1e-10)
    error = y_actual - y_predicted
    
    code_length = 0.5 * np.log2(2 * np.pi * sigma**2) + \
                  (error**2) / (2 * sigma**2 * np.log(2))
    
    return code_length


def solution_prequential_mdl(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    start_index: int = None
) -> float:
    """
    预序贯 MDL：预测每个点，并累加对数损失。
    """
    n = len(x)
    
    # 拟合至少需要 degree+1 个点，另需 1 个预测点
    if start_index is None:
        start_index = degree + 2
    
    if start_index >= n:
        return float('inf')
    
    total_code_length = 0.0
    
    for i in range(start_index, n):
        # 训练数据：第 i 个点之前的全部数据
        x_train = x[:i]
        y_train = y[:i]
        
        # 预测第 i 个点
        prediction, sigma = solution_predict_next_point(
            x_train, y_train, x[i], degree
        )
        
        # 该预测的编码代价
        code_length = solution_code_length_for_point(y[i], prediction, sigma)
        total_code_length += code_length
    
    return total_code_length


def solution_prequential_model_selection(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Tuple[int, dict]:
    """
    使用预序贯 MDL 选择最佳次数。
    """
    scores = {}
    
    for degree in range(max_degree + 1):
        scores[degree] = solution_prequential_mdl(x, y, degree)
    
    best_degree = min(scores, key=scores.get)
    
    return best_degree, scores


# =============================================================================
# 练习 3 解答：模型选择
# =============================================================================

class SolutionMDLModelSelector:
    """Complete implementation of MDL model selector."""
    
    def __init__(self, max_degree: int = 10, precision_bits: int = 32):
        self.max_degree = max_degree
        self.precision_bits = precision_bits
        self.scores = {}
        self.model_costs = {}
        self.data_costs = {}
        self.best_degree = None
        self.best_coefficients = None
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> 'SolutionMDLModelSelector':
        """Fit model selector to data."""
        for deg in range(self.max_degree + 1):
            total, model, data = solution_two_part_mdl(
                x, y, deg, self.precision_bits
            )
            self.scores[deg] = total
            self.model_costs[deg] = model
            self.data_costs[deg] = data
        
        self.best_degree = min(self.scores, key=self.scores.get)
        self.best_coefficients = np.polyfit(x, y, self.best_degree)
        
        return self
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions using selected model."""
        if self.best_coefficients is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.poly1d(self.best_coefficients)(x)
    
    def get_compression_ratio(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate compression ratio."""
        raw_bits = len(y) * 32
        mdl_bits = self.scores[self.best_degree]
        return raw_bits / mdl_bits if mdl_bits > 0 else 0
    
    def get_model_probability(self, degree: int) -> float:
        """Get approximate posterior probability."""
        min_score = min(self.scores.values())
        normalized = {k: -(v - min_score) for k, v in self.scores.items()}
        powers = {k: 2**v for k, v in normalized.items()}
        total = sum(powers.values())
        return powers[degree] / total if total > 0 else 0


def solution_compare_with_cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
    n_folds: int = 5
) -> Dict[str, int]:
    """Compare MDL with k-fold cross-validation."""
    # MDL 选择
    selector = SolutionMDLModelSelector(max_degree=max_degree)
    selector.fit(x, y)
    mdl_best = selector.best_degree
    
    # K 折交叉验证
    n = len(x)
    fold_size = n // n_folds
    
    cv_errors = {}
    
    for deg in range(max_degree + 1):
        fold_errors = []
        
        for fold in range(n_folds):
            # 划分数据
            val_start = fold * fold_size
            val_end = val_start + fold_size
            
            val_idx = np.arange(val_start, val_end)
            train_idx = np.concatenate([np.arange(0, val_start), 
                                         np.arange(val_end, n)])
            
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]
            
            # 拟合并评估
            coeffs = np.polyfit(x_train, y_train, deg)
            predictions = np.poly1d(coeffs)(x_val)
            mse = np.mean((y_val - predictions)**2)
            fold_errors.append(mse)
        
        cv_errors[deg] = np.mean(fold_errors)
    
    cv_best = min(cv_errors, key=cv_errors.get)
    
    return {'mdl_best': mdl_best, 'cv_best': cv_best}


# =============================================================================
# 练习 4 解答：NML 复杂度
# =============================================================================

def solution_logsumexp(log_values: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    max_val = np.max(log_values)
    return max_val + np.log(np.sum(np.exp(log_values - max_val)))


def solution_gaussian_log_likelihood(
    y: np.ndarray,
    predictions: np.ndarray,
    sigma: float = None
) -> float:
    """Gaussian log-likelihood (in nats, not bits)."""
    residuals = y - predictions
    n = len(y)
    
    if sigma is None:
        sigma = np.sqrt(np.mean(residuals**2))
    sigma = max(sigma, 1e-10)
    
    return -0.5 * n * np.log(2 * np.pi * sigma**2) - \
           np.sum(residuals**2) / (2 * sigma**2)


def solution_estimate_nml_complexity_polynomial(
    x: np.ndarray,
    degree: int,
    y_range: Tuple[float, float] = (-100, 100),
    n_samples: int = 1000
) -> float:
    """Estimate NML complexity via Monte Carlo."""
    n = len(x)
    max_log_likelihoods = []
    
    for _ in range(n_samples):
        # 采样随机 y 值
        y_random = np.random.uniform(y_range[0], y_range[1], n)
        
        # 拟合多项式，得到最大似然估计
        try:
            coeffs = np.polyfit(x, y_random, min(degree, n - 1))
            predictions = np.poly1d(coeffs)(x)
            
            # sigma 的最大似然估计
            residuals = y_random - predictions
            sigma_ml = np.sqrt(np.mean(residuals**2))
            sigma_ml = max(sigma_ml, 1e-10)
            
            # 对数似然
            log_lik = solution_gaussian_log_likelihood(
                y_random, predictions, sigma_ml
            )
            max_log_likelihoods.append(log_lik)
        except:
            continue
    
    if not max_log_likelihoods:
        return 0.0
    
    # 复杂度近似 = 平均最大似然的对数；实际积分还需体积等校正
    log_liks = np.array(max_log_likelihoods)
    complexity = solution_logsumexp(log_liks) - np.log(len(log_liks))
    
    return complexity / np.log(2)  # 转换为比特


def solution_compute_nml_mdl(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    y_range: Tuple[float, float] = None
) -> Tuple[float, float, float]:
    """Compute NML-based MDL score."""
    n = len(x)
    
    # 自动设置 y_range
    if y_range is None:
        margin = 2 * (np.max(y) - np.min(y))
        y_range = (np.min(y) - margin, np.max(y) + margin)
    
    # 拟合多项式
    coeffs = np.polyfit(x, y, degree)
    predictions = np.poly1d(coeffs)(x)
    residuals = y - predictions
    
    # sigma 的最大似然估计
    sigma_ml = np.sqrt(np.mean(residuals**2))
    sigma_ml = max(sigma_ml, 1e-10)
    
    # 拟合项（负对数似然，单位为比特）
    fit_term = -solution_gaussian_log_likelihood(y, predictions, sigma_ml) / np.log(2)
    
    # 复杂度
    complexity = solution_estimate_nml_complexity_polynomial(
        x, degree, y_range, n_samples=500
    )
    
    return fit_term + complexity, fit_term, complexity


# =============================================================================
# 练习 5 解答：MDL、AIC 与 BIC
# =============================================================================

def solution_compute_aic(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    """Compute AIC score."""
    n = len(x)
    k = degree + 2  # 系数与方差
    
    coeffs = np.polyfit(x, y, degree)
    residuals = y - np.poly1d(coeffs)(x)
    sigma_ml = np.sqrt(np.sum(residuals**2) / n)
    sigma_ml = max(sigma_ml, 1e-10)
    
    log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma_ml**2) + 1)
    
    return 2 * k - 2 * log_likelihood


def solution_compute_bic(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    """Compute BIC score."""
    n = len(x)
    k = degree + 2
    
    coeffs = np.polyfit(x, y, degree)
    residuals = y - np.poly1d(coeffs)(x)
    sigma_ml = np.sqrt(np.sum(residuals**2) / n)
    sigma_ml = max(sigma_ml, 1e-10)
    
    log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma_ml**2) + 1)
    
    return k * np.log(n) - 2 * log_likelihood


def solution_select_with_all_criteria(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict[str, Dict]:
    """Run all three criteria."""
    results = {
        'mdl': {'scores': {}, 'best': None},
        'aic': {'scores': {}, 'best': None},
        'bic': {'scores': {}, 'best': None}
    }
    
    for deg in range(max_degree + 1):
        results['mdl']['scores'][deg] = solution_two_part_mdl(x, y, deg)[0]
        results['aic']['scores'][deg] = solution_compute_aic(x, y, deg)
        results['bic']['scores'][deg] = solution_compute_bic(x, y, deg)
    
    for method in ['mdl', 'aic', 'bic']:
        results[method]['best'] = min(
            results[method]['scores'],
            key=results[method]['scores'].get
        )
    
    return results


def solution_monte_carlo_comparison(
    true_degree: int = 2,
    n_samples: int = 50,
    noise_std: float = 1.5,
    n_trials: int = 100
) -> Dict[str, float]:
    """Compare accuracy of all criteria."""
    x = np.linspace(0, 10, n_samples)
    
    correct = {'mdl': 0, 'aic': 0, 'bic': 0}
    
    for trial in range(n_trials):
        np.random.seed(trial)
        
        # 生成多项式
        true_coeffs = np.random.randn(true_degree + 1) * 2
        y_true = np.poly1d(true_coeffs)(x)
        y = y_true + np.random.randn(n_samples) * noise_std
        
        # 运行全部准则
        results = solution_select_with_all_criteria(x, y, max_degree=10)
        
        for method in ['mdl', 'aic', 'bic']:
            if results[method]['best'] == true_degree:
                correct[method] += 1
    
    return {
        'mdl_accuracy': correct['mdl'] / n_trials,
        'aic_accuracy': correct['aic'] / n_trials,
        'bic_accuracy': correct['bic'] / n_trials
    }


# =============================================================================
# 主程序：验证参考解答
# =============================================================================

def verify_solutions():
    """Quick verification that solutions work."""
    print("Verifying Exercise Solutions")
    print("=" * 50)
    
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.5
    
    print("\n[ok] Exercise 1: Two-Part MDL")
    total, model, data = solution_two_part_mdl(x, y, 2)
    print(f"  MDL(degree=2) = {total:.1f} bits")
    
    print("\n[ok] Exercise 2: Prequential MDL")
    preq = solution_prequential_mdl(x, y, 2)
    print(f"  Prequential(degree=2) = {preq:.1f} bits")
    
    print("\n[ok] Exercise 3: Model Selection")
    selector = SolutionMDLModelSelector(max_degree=8)
    selector.fit(x, y)
    print(f"  Selected degree: {selector.best_degree}")
    
    print("\n[ok] Exercise 4: NML Complexity")
    complexity = solution_estimate_nml_complexity_polynomial(x, 2, n_samples=200)
    print(f"  COMP(degree=2) = {complexity:.1f} bits")
    
    print("\n[ok] Exercise 5: MDL vs AIC vs BIC")
    mc = solution_monte_carlo_comparison(n_trials=50)
    print(f"  MDL: {mc['mdl_accuracy']*100:.0f}%, "
          f"AIC: {mc['aic_accuracy']*100:.0f}%, "
          f"BIC: {mc['bic_accuracy']*100:.0f}%")
    
    print("\n" + "=" * 50)
    print("All solutions verified!")


if __name__ == "__main__":
    verify_solutions()
