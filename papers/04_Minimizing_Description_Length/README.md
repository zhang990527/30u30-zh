# 第 4 天：保持神经网络简洁（MDL）

> Hinton 与 van Camp（1993）——[COLT '93 论文](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf)

**用时：** 3-5 小时
**前置知识：** 第 3 天（正则化概念）
**代码：** 仅使用 NumPy

---

## 这篇论文究竟研究什么

这是一篇 6 页的理论论文，联系了三个思想：

1. **最小描述长度（Minimum Description Length，MDL）**：最好的模型使描述模型及数据误差所需的总比特数最少。
2. **贝叶斯神经网络（Bayesian neural network）**：权重是概率分布，而不是固定数值。
3. **变分推断（variational inference）**：通过可处理的变分目标近似后验。

Hinton 与 van Camp 揭示了这些思想在其构造下的联系。如果用高斯分布表示权重，并利用“比特回收”（bits back）编码最小化描述长度，就会得到最小化变分自由能的目标：

$$\mathcal{L} = \underbrace{E_q[\text{error}]}_{\text{data fit}} + \underbrace{KL(q \| p)}_{\text{complexity cost}}$$

公式中的 data fit 表示数据拟合，complexity cost 表示复杂度代价。

在 1993 年，这种联系并不显然。这篇论文为后来许多贝叶斯深度学习工作奠定了理论基础。

---

## 核心思想

### 固定权重与含噪权重

普通神经网络中的每个权重都是一个固定数值（点估计）。

```
普通网络： w = 5.123456     （一个数值，精度高，需要很多比特）
MDL：      w ~ N(5.1, 0.2) （一个分布，包含位置与离散程度）
```

MDL 方法提出：如果权重带有噪声时网络仍能正常工作，就不需要那么高的精度。省下的精度直接对应省下的描述比特。

### “比特回收”论证

这是论文的关键思想。使用分布而非点值编码权重时：

1. 按分布 $q(w)$ 表示一个样本，其信息量为 $-\log q(w)$ 比特。
2. 接收方可以恢复用于采样的随机信息，从而“回收”比特。
3. 相对于双方共享的先验 $p(w)$，每个权重的净代价为 $\log \frac{q(w)}{p(w)}$。
4. 取期望后：$E_q[\log q/p] = KL(q \| p)$。

后验与先验之间的 KL 散度就是这里的描述代价。这样，信息论论证便得到与变分贝叶斯相同的目标。

### 完整目标

$$\mathcal{L} = \underbrace{KL(q(w) \| p(w))}_{\text{weight cost}} + \underbrace{E_{q(w)}[-\log p(D|w)]}_{\text{error cost}}$$

其中 weight cost 为权重代价，error cost 为误差代价。

- **KL 项**将权重拉向先验，使描述更简单、分布更分散。
- **误差项**促使权重拟合数据，使取值更精确。
- 训练需要在这两种作用之间权衡。

### KL 散度的闭式表达式

对于后验 $q = \mathcal{N}(\mu, \sigma^2)$ 和先验 $p = \mathcal{N}(0, 1)$：

$$KL = \frac{1}{2}\sum_i(\sigma_i^2 + \mu_i^2 - 1 - \log \sigma_i^2)$$

其性质如下：
- 较大的 $|\mu|$ 会受到惩罚（权重被拉向零）。
- 较小的 $\sigma$ 会受到惩罚（需要保留一定不确定性）。
- 很大的 $\sigma$ 也会被 $\sigma^2$ 项惩罚（不能有无限大的噪声）。

---

## 为什么有效：平坦极小值

权重噪声使网络难以停留在损失曲面的**尖锐极小值**附近。如果权重发生 $\sigma$ 尺度的扰动就使损失剧烈变化，期望误差便会很大。因此，网络倾向于寻找宽阔、平坦的区域，使损失在权重噪声下仍然稳定。

原仓库据此解释平坦极小值与泛化的联系：从训练数据切换到测试数据时，小变化不至于引起很大的误差增长。后来 Hochreiter 与 Schmidhuber（1997）、Keskar 等（2017）进一步研究了这一联系，本论文是较早提出相关论证的工作之一。

> **核验提示：** 平坦性与泛化之间的关系依赖参数化和具体假设，不能将“平坦”直接视为泛化良好的充分保证。上文保留原仓库的教学解释；原论文实验的具体任务及结论仍待一手全文核对。

---

## 论文展示了什么

原仓库将论文实验描述为一个简单回归任务上的概念验证：
- 两层网络学习一个目标函数。
- MDL 方法找到更简单、泛化更好的解。
- 描述长度中复杂度与准确性的权衡符合预期。

这是一篇**理论论文**，而不是基准评测论文。原仓库指出其中没有 MNIST 结果，也没有最先进性能的声明；贡献在于数学框架。以上实验细节尚未完成一手全文核验。

---

## 网络结构

### 普通神经元与贝叶斯神经元

```
普通神经元：                    贝叶斯神经元：
  w = 固定值                      w ~ N(mu, sigma^2)
  y = w*x + b                    w = mu + sigma * epsilon   (epsilon ~ N(0,1))
  每次输出相同                    y = w*x + b
                                  每次前向传播的输出不同！
```

### 前向传播（本项目实现）

```python
# 为每个权重存储 mu 与 rho，其中 sigma = softplus(rho)
sigma = log(1 + exp(rho))         # 保证 sigma > 0
epsilon = sample from N(0, 1)     # 随机噪声；此行为示意性伪代码
w = mu + sigma * epsilon          # 重参数化技巧（Kingma，2014）

# 使用采样权重进行普通前向传播
y = x @ w + b
```

注意：本实现采用的重参数化技巧（reparameterization trick，$w = \mu + \sigma \cdot \epsilon$）通常与 Kingma 与 Welling（2014）的工作联系在一起，并非来自本论文。使用它可以让梯度通过采样步骤进行训练；这不是对该技巧最早历史来源的断言。

### 训练循环

```python
for epoch in range(epochs):
    # 前向传播：采样权重并计算预测
    pred = net.forward(X)           # 内部进行权重采样

    # 损失 = 误差 + beta * 复杂度
    error = MSE(pred, y)            # 数据拟合得怎样？
    complexity = net.total_kl()     # 权重有多复杂？
    loss = error + beta * complexity

    # 反向传播与参数更新
    net.backward(d_loss)
    net.update_weights(lr, beta)
```

### 带不确定性的预测

```python
# 运行模型 N 次，每次采样不同的权重
predictions = [net.forward(X_test) for _ in range(100)]

mean = average(predictions)        # 均值估计
std = standard_deviation(predictions)  # 不确定性
```

网络见过数据的区域：不确定性通常较低，预测较集中。
网络未见过数据的区域：预期表现为更高的不确定性，预测更分散。

> 这是教学实验希望观察到的现象，不是任意近似贝叶斯网络都能保证满足的性质。

---

## 训练超参数

原仓库说明论文没有提供这里所用的实用训练细节；下表是本实现的建议设置：

| 参数 | 典型范围 | 控制的内容 |
|-----------|--------------|------------------|
| `kl_weight` ($\beta$) | 0.001 - 1.0 | 数据拟合与简洁性的权衡，是最重要的参数。 |
| `hidden_size` | 10 - 50 | 网络容量 |
| `rho_init` | -3.0 至 -5.0 | 初始不确定性（softplus(-3) ~ 0.05） |
| `lr` | 0.001 - 0.01 | 学习率 |

### 建议起点
```python
hidden_size = 20
kl_weight = 0.1
lr = 0.01
epochs = 2000
```

---

## 实现说明

`implementation.py` 中的代码在论文之外采用了以下技术：

- **重参数化技巧**：采用 Kingma 与 Welling（2014）中常见的形式，使反向传播能够通过权重采样。
- **Softplus 参数化**：学习 $\rho$，再计算 $\sigma = \log(1 + e^\rho)$，以保证 $\sigma > 0$。
- **蒙特卡洛不确定性估计**：通过多次前向传播估计预测区间。原仓库将其列为现代实践，而非论文内容。
- **带缺口的正弦波实验**：本项目为直观展示认知不确定性（epistemic uncertainty）而设计的教学实验。

需要注意：
- **每次前向传播结果不同**，因为每次都会采样权重。这正是设计目的。
- **$\beta$（kl_weight）至关重要**：过大时模型可能忽略数据，预测接近平线；过小时可能过拟合，表现得像缺少不确定性的普通神经网络。
- **梯度来自两个来源**：误差损失通过采样权重传播；KL 散度直接作用于 $\mu$ 和 $\sigma$。

---

## 实践任务

### 快速开始

```bash
python train_minimal.py
python train_minimal.py --kl-weight 0.5    # 更强的正则化
python train_minimal.py --kl-weight 0.001  # 更弱的正则化
```

### 练习（位于 `Exercises/`）

| # | 任务 | 学习内容 | 来源 |
|---|------|-------------------|--------|
| 1 | 重参数化技巧 | 以可微方式采样权重 | 本项目补充（采用 Kingma 2014 中的技术） |
| 2 | 缺口实验 | 认知不确定性的可视化 | 本项目补充 |
| 3 | Beta 参数研究 | 复杂度与准确性的权衡 | 本项目补充（探索论文核心方程） |
| 4 | 蒙特卡洛预测 | 汇总随机预测 | 本项目补充 |
| 5 | 帕累托前沿 | 压缩与准确性的分析 | 本项目补充（直观展示 MDL 原理） |

所有练习均为本项目的教学补充。论文自身主要讨论理论，并包含一个小型实验。练习 3 和 5 与论文核心思想最直接相关。

参考解答位于 `solutions.py` 和 `solutions_extra.py`，请先独立尝试。

---

## 关键收获

1. **MDL 与变分贝叶斯的联系。** 论文的核心贡献是：在所讨论的编码与近似假设下，最小化权重描述长度得到与变分推断相同的目标，权重代价对应 KL 散度。

2. **含噪权重倾向于平坦极小值。** 权重噪声促使网络进入宽阔、稳定的损失区域，小扰动不容易造成损失大幅变化；其泛化意义应结合前述限制理解。

3. **不确定性是有用信息。** 贝叶斯权重可以反映网络的认知不确定性，数据稀疏区域可能出现更大的预测差异。普通点估计网络本身不直接提供这种权重不确定性度量。

4. **简洁性需要代价。** $\beta$ 显式控制权衡。更简单的模型可能更稳健，但也可能牺牲拟合精度。

---

## 历史意义

这篇论文较早将信息论中的 MDL／压缩与神经网络的贝叶斯处理联系起来，早于以下工作：
- 变分自编码器（Variational Autoencoder，VAE；Kingma 与 Welling，2014）：使用相关的 ELBO 目标。
- 反向传播贝叶斯（Bayes by Backprop；Blundell 等，2015）：这些思想的现代实现。
- 蒙特卡洛随机失活（MC Dropout；Gal 与 Ghahramani，2016）：揭示 dropout 与近似变分推断的联系。

MacKay（1992）在相近时期独立发展了神经网络的贝叶斯框架，采用拉普拉斯近似而非变分方法。这两条研究路线共同推动了贝叶斯深度学习的发展。

---

## 本目录文件

| 文件 | 内容 |
|------|-----------|
| `implementation.py` | NumPy 贝叶斯神经网络：BayesianLinear 层与 MDLNetwork |
| `train_minimal.py` | 训练脚本：支持命令行参数的带缺口正弦波实验 |
| `visualization.py` | 不确定性包络、权重分布和损失曲线 |
| `notebook.ipynb` | MDL 概念的交互式讲解 |
| `Exercises/` | 5 个练习：重参数化、缺口实验、beta 研究、蒙特卡洛采样、帕累托前沿 |
| `paper_notes.md` | Hinton 与 van Camp 论文的详细笔记 |
| `CHEATSHEET.md` | 公式与超参数速查 |

---

## 延伸阅读

- [Hinton 与 van Camp（1993）](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf)：本篇论文。
- [MacKay（1992）](http://www.inference.org.uk/mackay/thesis.pdf)：采用拉普拉斯近似的并行贝叶斯神经网络研究路线。
- [Blundell 等（2015）](https://arxiv.org/abs/1505.05424)：《神经网络中的权重不确定性》（Weight Uncertainty in Neural Networks），本论文思想的现代发展。
- [Kingma 与 Welling（2014）](https://arxiv.org/abs/1312.6114)：变分自编码器，使用相关的 ELBO 目标与重参数化技巧。
- [Gal 与 Ghahramani（2016）](https://arxiv.org/abs/1506.02142)：将 MC Dropout 解释为近似变分推断。

---

**下一篇：** [第 5 天](../05_MDL_Principle/)
