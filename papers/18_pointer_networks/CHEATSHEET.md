# 第 18 天速查：指针网络

指针网络（Vinyals 等，2015）的快速参考。

---

## 核心思想（30 秒）

标准序列模型从**固定**词表中选择；指针网络从**依赖输入**的词表中选择。注意力机制就是输出层。通过直接指向输入元素，模型可以解决凸包、德洛内三角剖分和 TSP 等组合问题；这些问题的输出词表恰好就是输入项集合。

---

## 快速开始

```bash
# 用示例排序数据训练
python train_minimal.py --seq_len 5 --epochs 50

# 可视化注意力热图
python visualization.py
```

---

## 关键超参数

### 论文设置（第 4.1 节）

单层 LSTM，256 或 512 个隐藏单元；SGD，lr=1.0；批大小 128；权重从 [-0.08, 0.08] 均匀初始化；L2 梯度裁剪阈值 2.0；100 万个训练样例。

> 核验状态：这些数值保留自原仓库，本次未成功获取论文正文核验。

### 本章简化实现

| 参数 | 数值 | 作用 | 提示 |
|:---|:---|:---|:---|
| `input_size` | 1-2 | 每项的特征数 | 排序为 1，(x, y) 坐标为 2 |
| `hidden_size` | 64-512 | 模型容量 | 论文使用 256 或 512 |
| `seq_len` | 5-100 | 输入长度 | Ptr-Net 可用于测试 N 大于训练长度的泛化情况 |
| `lr` | 0.001 | 更新步长 | 本章用 Adam；论文用 lr=1.0 的 SGD |

---

## 常见问题与处理

### 损失无穷大

```python
# 处理：确保掩码正确
scores = scores.masked_fill(mask, float('-inf'))
# 若屏蔽所有项，softmax 会返回 NaN。
```

### 广播维度不匹配

```python
# 处理：给解码器状态增加一个维度
decoder_proj = W2(decoder_hidden).unsqueeze(1) # (B, 1, H)
combined = torch.tanh(encoder_proj + decoder_proj) # (B, S, H)
```

---

## 数学实现（可复制的代码片段）

### 指针概率（公式 3）

```python
# 1. 投影查询（解码器）和键（编码器）
q = self.W2(decoder_hidden).unsqueeze(1)
k = self.W1(encoder_outputs)

# 2. 加性注意力
u = self.v(torch.tanh(q + k)).squeeze(-1)

# 3. softmax 就是输出
probs = F.softmax(u, dim=-1)
```

---

## 可视化示例

```python
from visualization import plot_attention

# 可视化指针热图
# inputs: [0.7, 0.2, 0.9]
# attention_probs: 形状 (3, 3)
plot_attention(inputs, attention_probs, save_path='ptr_viz.png')
```

---

## 实验思路

### 入门

- 用不同序列长度训练（5、10、15）。
- 测试泛化：长度 5 上训练，长度 10 上测试。
- 调整 softmax 的温度，使指针分布更尖锐。

### 中等

- 实现贪心解码与采样解码并比较。
- 增加“有放回采样”的开关。
- 在包含 5 个城市的简单 TSP 数据集上训练。

### 进阶

- 实现用于文本复制的指针生成网络。
- 增加束搜索，改进组合优化结果。
- 比较 LSTM 编码器与使用自注意力的 Transformer 编码器。

---

## 文件参考

| 文件 | 用途 |
|:---|:---|
| `implementation.py` | 核心模型与注意力逻辑 |
| `train_minimal.py` | 排序任务快速训练循环 |
| `visualization.py` | 热图与指针绘制 |
| `notebook.ipynb` | 交互式构建与演示 |
| `exercises/` | 练习实现公式 3 |

---

## 调试清单

- [ ] 输入坐标是否已投影到隐藏空间？
- [ ] 解码器隐藏状态是否匹配注意力投影的维度？
- [ ] 掩码是否避免选择填充词元？
- [ ] softmax 的 dim 是否为 -1，即沿输入位置归一化？

---

## 实践建议

1. **教师强制：**构建序列时，原仓库建议训练阶段使用教师强制；这并非所有训练策略的必要条件。
2. **掩码：**TSP 中很重要；对已经访问的城市设置 $u_j = -\infty$。
3. **预投影：**原仓库建议原始值先通过 `Linear` 层，再送入 LSTM；这是本实现选择，不是 LSTM 的普遍要求。

---

## 成功标准

- 注意力热图接近尖锐的排列矩阵。
- 长度 5 序列上的排序准确率 > 95%。
- 对略长于训练序列的输入仍有一定泛化能力。

> 这些是原仓库提出的实验目标，不是本次翻译已经达到的测量结果。

---

## 下一天

第 19 天：关系推理。

---

**有疑问？**查看 [README.md](README.md)，或在社区中提问。
