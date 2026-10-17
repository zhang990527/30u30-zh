<a id="day-6-cheat-sheet-the-first-law-of-complexodynamics"></a>
# 第 6 天速查：复杂动力学第一定律

<a id="the-big-idea-30-seconds"></a>
## 核心思想（30 秒）

熵在相应物理条件下趋于增加，而“复杂性”（有趣程度）直观地在中间时刻达到峰值后回落。Aaronson 提出以**复杂熵（complextropy）**——资源受限的精致复杂度（sophistication）——形式化地捕捉这一现象。猜想是：复杂熵在初期低、中间高、平衡时低。原仓库称截至 2024 年尚未证明，本次未核实最新研究状态。

**记住这张图：**

```
复杂性
    ^
    |        *  *  *
    |      *          *
    |    *              *
    |  *                  *
    | *                      *
    +--*---------------------*--> 时间
    t=0    卷须纹理       平衡
```

---

<a id="quick-start"></a>
## 快速开始

```bash
cd papers/06_Complexodynamics

<a id="run-coffee-mixing-simulation-with-complexity-tracking"></a>
# 运行咖啡混合模拟并跟踪复杂性
python train_minimal.py --grid-size 64 --steps 50000

<a id="compare-different-complexity-measures"></a>
# 比较不同复杂性度量
python train_minimal.py --grid-size 64 --steps 50000 --compare-measures

<a id="multi-scale-analysis"></a>
# 多尺度分析
python train_minimal.py --grid-size 64 --steps 50000 --multiscale

<a id="generate-all-visualizations"></a>
# 生成所有可视化图
python visualization.py
```

---

<a id="complexity-measures-at-a-glance"></a>
## 复杂性度量一览

| 度量 | 定义 | 优点 | 局限 |
|---------|-----------|------|------|
| **柯尔莫哥洛夫复杂度（KC）** | 生成 x 的最短程序长度 | 理论明确 | 不可计算；随机字符串可有最大 KC |
| **gzip 大小** | 压缩后的字节数 | 快速、可复现 | 非最优；会漏掉复杂模式 |
| **精致复杂度** | 包含 x 且使 x 在其中随机的最短描述“模型”S 的 KC | 对简单和随机对象都低 | 一般不可计算；确定性系统中受 O(log t) 限制 |
| **复杂熵** | 资源受限的精致复杂度 | 被猜想在中间时刻达到峰值 | 此处定义非正式；猜想未证明 |
| **粗粒化 KC** | 对模糊或下采样状态进行 gzip 压缩 | 捕捉宏观结构 | 尺度选择依赖经验 |
| **两段式编码** | 模型描述 + 残差 | 试图直接近似精致复杂度 | 没有唯一公认的模型选择 |
| **逻辑深度** | 接近最短描述的程序的运行时间（简化说明） | 对简单和随机对象都低 | 难计算；是否出现峰值存在争议（Bennett 与 Aaronson） |

---

<a id="key-parameters-for-the-coffee-simulation"></a>
## 咖啡模拟的关键参数

| 参数 | 典型范围 | 作用 | 建议 |
|-----------|--------------|--------------|------|
| `grid_size` | 32-128 | 方形像素网格的边长 | 64 较均衡；128 较慢，但细节更多。 |
| `n_steps` | 10000-100000 | 批量交换步数 | 原说明估计达到平衡约需 ~grid_size^3；此估计未验证，步数也不等于成功交换次数。 |
| `compresslevel` | 9 | gzip 压缩等级 | 统一使用 9，保证一致。 |
| `block_size` | 2-16 | 粗粒化块大小 | 可以从 grid_size/8 开始尝试。 |
| `measure_interval` | 100-1000 | 两次复杂性测量间隔的步数 | 太频繁会慢，太稀疏可能错过峰值。 |

---

<a id="common-issues--fixes"></a>
## 常见问题与处理

<a id="gzip-gives-noisy-complexity-estimates"></a>
### gzip 复杂性估计噪声较大

```python
<a id="problem-single-gzip-measurement-is-noisy"></a>
# 问题：单次 gzip 测量可能有噪声
<a id="fix-average-over-multiple-serialization-orders-or-runs"></a>
# 处理：对多个序列化顺序或多次运行求平均

def robust_gzip(grid, n_samples=5):
    sizes = []
    data = grid.astype(np.uint8).tobytes()
    sizes.append(len(gzip.compress(data, compresslevel=9)))
    # 同时尝试列优先顺序
    data_col = grid.T.astype(np.uint8).tobytes()
    sizes.append(len(gzip.compress(data_col, compresslevel=9)))
    return np.mean(sizes)
```

<a id="complexity-curve-is-flat-no-hump"></a>
### 复杂性曲线平坦（没有隆起）

```python
<a id="problem-grid-too-small--structure-is-too-coarse-to-measure"></a>
# 问题：网格过小，结构太粗，难以测量
<a id="fix-1-increase-grid_size-to-at-least-64"></a>
# 处理 1：将 grid_size 增大到至少 64
grid_size = 64  # 不使用 16

<a id="fix-2-measure-more-frequently-in-the-early-phase"></a>
# 处理 2：初期更频繁地测量
<a id="most-of-the-action-happens-in-the-first-30-of-steps"></a>
# 原说明认为主要变化发生在前 30% 步数内；应实测核对
measure_points = np.concatenate([
    np.linspace(0, n_steps * 0.3, 100),
    np.linspace(n_steps * 0.3, n_steps, 50)
]).astype(int)
```

<a id="simulation-is-too-slow"></a>
### 模拟太慢

```python
<a id="problem-swapping-one-pair-per-step-is-very-slow"></a>
# 问题：每步只交换一对，速度很慢
<a id="fix-vectorize--swap-multiple-non-adjacent-pairs-per-step"></a>
# 原说明建议向量化、每步交换多对非相邻像素；下列示例实际上仍为顺序循环，且不保证像素对互不相邻
def batch_swap(grid, n_swaps=100):
    """每步尝试交换 n_swaps 对随机相邻的咖啡—牛奶像素。"""
    rows, cols = grid.shape
    for _ in range(n_swaps):
        r, c = np.random.randint(0, rows), np.random.randint(0, cols-1)
        if grid[r, c] != grid[r, c+1]:
            grid[r, c], grid[r, c+1] = grid[r, c+1], grid[r, c]
    return grid
```

<a id="coarse-graining-produces-all-zeros-or-all-ones"></a>
### 粗粒化结果全为零或全为一

```python
<a id="problem-block_size-too-large-relative-to-grid"></a>
# 问题：block_size 相对网格过大
<a id="fix-ensure-block_size--grid_size"></a>
# 处理：保证 block_size << grid_size
assert block_size <= grid_size // 4, (
    f"block_size {block_size} 相对于 grid_size {grid_size} 过大"
)
```

---

<a id="the-math-copy-paste-ready"></a>
## 数学与代码（可直接复制）

<a id="kolmogorov-complexity-kc"></a>
### 柯尔莫哥洛夫复杂度（KC）

```python
<a id="kc-is-uncomputable-gzip-gives-an-upper-bound"></a>
# KC 不可计算。gzip 在计入解压程序的固定开销后可给出描述长度上界。
import gzip

def kc_approx(data: bytes) -> int:
    """通过 gzip 压缩估计 KC 的上界（忽略固定解压开销）。"""
    return len(gzip.compress(data, compresslevel=9))
```

<a id="shannon-entropy-of-a-grid"></a>
### 网格的香农熵

```python
import numpy as np

def grid_entropy(grid):
    """像素取值分布的香农熵；交换守恒取值比例，因此此量不随混合增加。"""
    p = grid.mean()  # 1（咖啡）的比例
    if p == 0 or p == 1:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
```

<a id="coarse-grained-kc"></a>
### 粗粒化 KC

```python
def coarse_grained_kc(grid, block_size):
    """模糊网格的 KC 近似（Carroll 的度量）。"""
    h, w = grid.shape
    coarse = np.zeros((h // block_size, w // block_size))
    for i in range(coarse.shape[0]):
        for j in range(coarse.shape[1]):
            block = grid[i*block_size:(i+1)*block_size,
                        j*block_size:(j+1)*block_size]
            coarse[i, j] = block.mean()
    # 量化为 uint8 后压缩
    quantized = (coarse * 255).astype(np.uint8)
    return kc_approx(quantized.tobytes())
```

<a id="two-part-code-sophistication-proxy"></a>
### 两段式编码（精致复杂度代理量）

```python
def two_part_code(grid, block_size):
    """
    第一部分：粗粒度模型（精致复杂度代理量）
    第二部分：残差（随机性）
    """
    h, w = grid.shape
    coarse = np.zeros((h // block_size, w // block_size))
    for i in range(coarse.shape[0]):
        for j in range(coarse.shape[1]):
            block = grid[i*block_size:(i+1)*block_size,
                        j*block_size:(j+1)*block_size]
            coarse[i, j] = block.mean()

    # 第一部分：模型描述
    model_bytes = (coarse * 255).astype(np.uint8).tobytes()
    part1 = kc_approx(model_bytes)

    # 第二部分：残差（原注释写为“重建值 - 实际值”，但下方代码实际计算 grid - reconstructed）
    reconstructed = np.repeat(np.repeat(coarse, block_size, axis=0),
                              block_size, axis=1)[:h, :w]
    residual = (grid - reconstructed)
    residual_bytes = ((residual + 1) * 127).astype(np.uint8).tobytes()
    part2 = kc_approx(residual_bytes)

    return part1, part2
```

---

<a id="visualization-quick-reference"></a>
## 可视化速查

```python
import matplotlib.pyplot as plt

<a id="plot-complexity-over-time-the-hump"></a>
# 绘制复杂性随时间的变化（预期的“隆起”）
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

<a id="left-entropy-monotone-increase"></a>
# 左图：熵（原说明预期单调增加；本章实现的全局像素熵实际上不变）
axes[0].plot(times, entropies)
axes[0].set_title("熵（原说明预期单调）")
axes[0].set_xlabel("时间（交换次数）")

<a id="right-complexity-peaks-then-drops"></a>
# 右图：复杂性（预期先达到峰值再下降）
axes[1].plot(times, complexities)
axes[1].set_title("复杂性（隆起）")
axes[1].set_xlabel("时间（交换次数）")

plt.tight_layout()
plt.savefig("entropy_vs_complexity.png", dpi=150)
```

---

<a id="key-relationships"></a>
## 关键关系

```
KC(x) = 高     --> x 可能随机，也可能有结构
KC(x) = 低     --> x 简单（可压缩）

soph(x) = 高   --> x 具有真实结构（不是纯随机）
soph(x) = 低   --> x 简单或随机

complextropy(x, t) = 高 --> x 具有结构，且产生于
                              时间受限的物理过程（直观说明）
```

---

<a id="blog-comment-highlights-worth-reading"></a>
## 博文评论要点（值得阅读）

| 人物 | 评论编号 | 要点 |
|-----|----------|-----------|
| Sean Carroll | #6-7 | 粗粒化很重要；复杂性可能不平滑 |
| Charles Bennett | #110 | 支持逻辑深度；必须模拟混合，因此中间状态深 |
| Aaronson 的回应 | #112 | 质疑深度；报告 Ouellette 用 gzip 观察到隆起的结果 |
| Luca Trevisan | #17 | 多尺度方法：复杂性也出现在中间尺度 |
| Abram Demski | #47 | 认为若规则“有趣”，深度起初线性增加 |
