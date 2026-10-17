<a id="cheatsheet-the-coffee-automaton"></a>
# 速查表：咖啡元胞自动机

> Aaronson、Carroll、Ouellette（2014），arXiv:1405.6903 的快速参考

---

<a id="the-model"></a>
## 模型

<a id="setup"></a>
### 设置
- N x N 二值网格：1 = 奶油，0 = 咖啡
- 初始状态：上半部 = 1（奶油），下半部 = 0（咖啡）

<a id="interacting-model-section-31--the-main-one"></a>
### 有相互作用模型（第 3.1 节）——主要模型
```
每个时间步：
  1. 随机选择一对相邻格点（水平方向或垂直方向）
  2. 若两者不同，则交换
  3. 每步一次交换
```

<a id="non-interacting-model-section-32--the-control"></a>
### 无相互作用模型（第 3.2 节）——对照模型
```
每个时间步：
  对每个奶油粒子：
    移动到随机相邻格点（上／下／左／右）
  多个粒子可以占据同一格点
  咖啡只作为背景
```

---

<a id="the-measurements"></a>
## 测量

<a id="entropy-estimate-proxy-for-kx"></a>
### 熵估计（K(x) 的代理量）
```python
entropy = len(gzip.compress(grid.tobytes(), compresslevel=9))
```
压缩细粒度（原始）二值网格，随时间单调增加。

<a id="apparent-complexity-estimate-proxy-for-kfx"></a>
### 表观复杂度估计（K(f(x)) 的代理量）
```python
<a id="step-1-coarse-grain--average-over-g-x-g-blocks"></a>
# 第 1 步：粗粒化，对 g x g 块取平均
coarse = average_blocks(grid, grain_size=g)

<a id="step-2-threshold-into-discrete-buckets"></a>
# 第 2 步：按阈值离散分箱
<a id="section-5-3-buckets-0-05-1"></a>
# 第 5 节：3 个分箱（0、0.5、1）
<a id="section-6-7-buckets-0-16-26--1"></a>
# 第 6 节：7 个分箱（0、1/6、2/6、……、1）
thresholded = threshold(coarse, num_buckets)

<a id="step-3-compress"></a>
# 第 3 步：压缩
complexity = len(gzip.compress(thresholded.tobytes(), compresslevel=9))
```
有相互作用模型先升后降；无相互作用模型在调整后保持平坦。

<a id="adjusted-coarse-graining-section-6"></a>
### 调整后的粗粒化（第 6 节）
```python
<a id="after-thresholding-apply-row-majority-adjustment"></a>
# 阈值化后应用行众数调整：
for each row in thresholded:
    majority_value = mode(row)
    for each cell in row:
        if abs(cell - majority_value) <= 1 bucket:
            cell = majority_value
```
消除边界像素伪影，无相互作用模型的复杂度降至接近零。

---

<a id="key-equations"></a>
## 关键公式

<a id="apparent-complexity-definition"></a>
### 表观复杂度（定义）
$$C_{\text{apparent}}(x) = K(f(x))$$
其中 $f$ = 平滑函数，$K$ = 柯尔莫哥洛夫复杂度。

<a id="sophistication-for-theory-reference--not-computed-in-experiments"></a>
### 精致度（供理论参考，实验中不计算）
$$\text{soph}_c(x) = \min\{K(S) : x \in S, \; K(S) + \log_2|S| \leq K(x) + c\}$$

<a id="apparent-complexity-as-resource-bounded-sophistication-section-25"></a>
### 表观复杂度作为资源受限的精致度（第 2.5 节）
$$K(f(x)) \approx K(S_{f,x}) \quad \text{where } S_{f,x} = \{y : f(y) = f(x)\}$$

<a id="non-interacting-random-walk-appendix"></a>
### 无相互作用随机游走（附录）
$$E[a_{t+1}(x,y)] = \frac{E[a_t(x-1,y)] + E[a_t(x+1,y)] + E[a_t(x,y-1)] + E[a_t(x,y+1)]}{4}$$

<a id="chernoff-concentration-appendix"></a>
### Chernoff 集中界（附录）
$$\Pr[|a_t(B) - E[a_t(B)]| > L^2/G] < 2\exp\left(-\frac{L^2}{3G^2}\right)$$

所需粗粒化块边长：$L \gg G\sqrt{3 \ln(2n^2)}$

---

<a id="scaling-results-figures-6-8"></a>
## 标度结果（图 6-8）

| 量 | 标度 | 拟合 $r^2$ |
|----------|---------|-----------|
| 最大熵 | ~ $n^2$（二次） | 0.9999 |
| 最大复杂度（有相互作用） | ~ $n$（线性） | 0.9798 |
| 最大复杂度（无相互作用） | ~ $n$（线性） | 0.9729 |
| 达到最大复杂度的时间（有相互作用） | ~ $n^2$（二次） | 0.9878 |
| 达到最大复杂度的时间（无相互作用） | ~ $n^2$（二次） | 0.9927 |

最大复杂度 ~ n 的原因：复杂性沿咖啡与奶油之间的一维边界发展。
最大熵 ~ n^2 的原因：熵计入全部 n^2 个粒子。

---

<a id="quick-api-reference-implementationpy"></a>
## API 速查（implementation.py）

### CoffeeAutomaton
```python
from implementation import CoffeeAutomaton

<a id="create-automaton"></a>
# 创建自动机
ca = CoffeeAutomaton(grid_size=100, model='interacting')

<a id="run-steps"></a>
# 推进时间步
ca.step(num_steps=1000)  # 批量交换以加速

<a id="get-current-state"></a>
# 获取当前状态
grid = ca.grid  # NumPy uint8 数组，值为 0 或 1

<a id="measure"></a>
# 测量
entropy = ca.entropy()        # gzip（细粒度状态）
complexity = ca.complexity(    # gzip（粗粒化状态）
    grain_size=10,
    num_buckets=7,
    adjust=True
)
```

<a id="measurements"></a>
### 测量函数
```python
from implementation import gzip_size, coarse_grain, threshold_array

<a id="raw-gzip-size-of-any-data"></a>
# 任意数据的原始 gzip 压缩大小
size = gzip_size(data_bytes)

<a id="coarse-grain-a-grid"></a>
# 对网格粗粒化
coarse = coarse_grain(grid, grain_size=10)

<a id="threshold-into-buckets"></a>
# 按阈值分箱
discrete = threshold_array(coarse, num_buckets=7)
```

---

<a id="four-complexity-measures-section-2--why-only-one-was-used"></a>
## 四种复杂性度量（第 2 节）——为何只用一种

| 度量 | 捕捉什么 | 未使用的原因 |
|---------|-----------------|--------------|
| 表观复杂度（apparent complexity） | 平滑后状态的 KC | **已使用**——可计算，符合直觉 |
| 精致度（sophistication） | 最小模型的大小 | 短程序输出的精致度始终不大 |
| 逻辑深度（logical depth） | 最短程序的运行时间 | 复杂图案不一定需要长时间计算 |
| 光锥复杂度（light-cone complexity） | 过去与未来的互信息 | 需要因果历史，而非仅有当前状态 |

---

<a id="common-gotchas"></a>
## 常见问题

1. **gzip 处理字节，而非浮点数：** 压缩前将网格转为字节
2. **先阈值化再压缩：** 浮点平均值不容易压缩
3. **每步交换一次很慢：** N=100 时需要约 10^7 步，使用批量交换
4. **无相互作用模型需要计数网格：** 粒子会重叠，因此使用 uint16，而非 uint8
5. **论文中的“熵” = gzip（细粒度状态）：** 不是香农熵公式
6. **3 分箱阈值化会产生伪影：** 使用 7 分箱加调整，结果更可靠
7. **粗粒化尺度很重要：** 过小则噪声未被平滑，过大则结构被抹去。默认 g = n/10。

---

<a id="paper-references"></a>
## 论文参考文献

- Koppel（1987）——精致度的定义
- Bennett（1995）——逻辑深度的定义
- Evans 等（2003）——OSCR 两段式编码，论文尝试过但未奏效
- Shalizi 等（2004）——光锥复杂度
- Gacs、Tromp、Vitanyi（2001）——算法统计
- Antunes 与 Fortnow（2009）——粗精致度 = 忙海狸深度
- Gell-Mann（1994）——《夸克与美洲豹》（The Quark and the Jaguar），对复杂性兴衰的非形式化讨论
- Carroll（2010）——《从永恒到此刻》（From Eternity to Here），时间之箭与复杂性
