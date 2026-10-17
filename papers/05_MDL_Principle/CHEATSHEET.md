# 最小描述长度原理速查表

Grünwald（2004）及本实现的快速参考。

---

## 论文贡献（30 秒）

**问题：** 如何从候选模型中选择合适模型并抑制过拟合？
**洞见：** 将模型选择视为编码选择，使数据总描述尽可能短。
**结果：** 两段式、预序贯、NML 等形式提供有编码依据的选择方法；具体编码与模型类仍需合理选择。

这是一篇系统介绍 MDL 的教程／综述，原仓库称约 80 页。它解释第 4 天贝叶斯权重压缩的相关基础，并联系柯尔莫哥洛夫复杂度。

> 已核对 [arXiv:math/0406077v1 原文](https://arxiv.org/html/math/0406077)。本仓库使用教学近似；章节定位和技术限制见下文及论文笔记。

---

## 关键方程

### 两段式 MDL（粗略形式）

```
总代价 = L(H) + L(D|H)
  L(H)   = k * b              （k 个参数，每个 b 比特）
  L(D|H) = (n/2) * log2(2*pi*e*var)   （高斯残差）

最佳模型 = argmin { L(H) + L(D|H) }
```

高斯残差项是连续密度形式；实际离散码长还需指定精度，不能直接等同于文件比特数。

### 预序贯 MDL（顺序形式）

```
L_preq = sum_i  -log2 P(x_i | x_1, ..., x_{i-1})
```

无须单独描述模型，累计预测的对数损失。

### NML（归一化最大似然）

```
P_NML(x) = P(x | theta_hat(x)) / C_n

参数复杂度的渐近主项：
  COMP(M, n) ~ (k/2) * log(n / 2*pi)
```

**原文勘误：** COMP 为参数复杂度，即 log C_n；完整随机复杂度还包括负对数最大似然。上式省略 Fisher 信息体积等项，需要正则性与有限归一化常数，不能当作完整通用公式。

---

## MDL、AIC 与 BIC

| 准则 | 惩罚 | 公式 |
|-----------|---------|---------|
| AIC | 2k | -2*logL + 2k |
| BIC | k*log(n) | -2*logL + k*log(n) |
| MDL（NML 主项） | ~(k/2)*log(n/2pi) | L(H) + L(D\|H) |

**限制：** 前两者使用 `-2 log L`，NML 使用 `-log L`，需先统一尺度与对数底。细化 MDL 涉及模型几何；原仓库称“接近零的参数必然更省比特”不具有普遍性。本实现固定每参数比特数，也没有该自适应性质。

---

## 快速开始

```bash
# 多项式次数选择示例
python train_minimal.py

# 蒙特卡洛比较（MDL、AIC 与 BIC）
python train_minimal.py --monte-carlo
```

### 在 Python 中使用

```python
from implementation import PolynomialMDL

# 拟合 3 次多项式并获取 MDL 得分
model = PolynomialMDL(degree=3)
model.fit(X, y)
score = model.total_mdl(X, y)

# 也可使用独立函数
from implementation import two_part_mdl_polynomial
total, model_cost, data_cost = two_part_mdl_polynomial(X, y, degree=3)
```

---

## 常见模式

### 模型选择循环

```python
best_score = float('inf')
for degree in range(1, max_degree + 1):
    coeffs = np.polyfit(X, y, degree)
    residuals = y - np.polyval(coeffs, X)
    score = (degree + 1) * 32 + (n/2) * np.log2(2*np.pi*np.e*np.var(residuals))
    if score < best_score:
        best_score, best_degree = score, degree
```

### 比较准则

```python
def compare(log_likelihood, k, n):
    AIC = -2 * log_likelihood + 2 * k
    BIC = -2 * log_likelihood + k * np.log(n)
    MDL = -log_likelihood / np.log(2) + k * 32
    return AIC, BIC, MDL
```

> 此示例中的 AIC/BIC 与 MDL 数值尺度不同，应分别比较同一准则下的候选模型，不能直接比较三个返回值的大小。

---

## 调试

### MDL 选择过于简单的模型
- 检查每参数比特数（32 对应 float32 的位宽，但不是普遍最优的参数编码）。
- 可能需要更多数据；该粗略 MDL 在小样本时可能较保守。

### MDL 选择过于复杂的模型
- 残差方差可能被低估。
- 检查 L(D|H) 中方差估计与所用似然定义是否一致。原仓库建议无偏方差，但最大似然方差与无偏方差不同，不能无条件替换。

### 所有准则一致
- 当模型质量差异明显时很常见；质量接近时更容易出现分歧。

---

## 论文内容与本项目补充

| 概念 | 原文位置／来源 |
|---------|--------|
| 两段式编码：L(H) + L(D\|H) | §2.4 |
| 预序贯 MDL | §2.6.4 |
| NML／随机复杂度 | §2.6.1，并区分参数复杂度 |
| MDL 与其他推断准则的联系 | §2.9 |
| 柯尔莫哥洛夫复杂度的联系 | §1.2.1 |
| 多项式次数选择 | 论文的贯穿示例思想；本仓库提供具体实现 |
| 蒙特卡洛稳健性比较 | 本项目补充 |
| 所有练习 | 实现论文相关概念的教学补充 |

---

*论文详情见 [paper_notes.md](paper_notes.md)，实现指南见 [README.md](README.md)。*
