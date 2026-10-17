# LSTM 速查表

《理解 LSTM 网络》（Understanding LSTM Networks）快速参考

---

## 核心思想（30 秒）

LSTM 增加了**细胞状态（cell state）**，通过加性更新路径传递梯度，从而缓解梯度消失。Colah 将细胞状态比作“传送带”：只要 LSTM 不通过门控主动修改它，信息就可以沿这条路径保留。

---

## 架构：四组变换（三个门与候选状态）

```
遗忘门：       f_t = σ(W_f·[h_{t-1}, x_t] + b_f)    # 0 = 遗忘，1 = 保留
输入门：       i_t = σ(W_i·[h_{t-1}, x_t] + b_i)    # 0 = 忽略，1 = 写入
候选细胞状态： C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C) # 待加入的新信息
输出门：       o_t = σ(W_o·[h_{t-1}, x_t] + b_o)    # 0 = 隐藏，1 = 输出

细胞状态：     C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t     # 记忆
隐藏状态：     h_t = o_t ⊙ tanh(C_t)                 # 输出
```

**关键认识：** 细胞状态通过**加性更新**形成更利于梯度传递的路径。

---

## 快速开始

### 训练
```bash
# 使用文本文件训练
python train_minimal.py --data input.txt --epochs 50 --hidden-size 128

# 自定义超参数
python train_minimal.py --data input.txt --hidden-size 256 --seq-length 100 --lr 0.001
```

### 生成
```bash
# 使用训练好的模型生成文本
python train_minimal.py --generate --checkpoint lstm_model.pkl --length 500 --temperature 0.8
```

### 在 Python 中使用
```python
from implementation import LSTM

# 创建 LSTM
lstm = LSTM(input_size=vocab_size, hidden_size=128, output_size=vocab_size)

# 前向传播
loss = lstm.forward(inputs, targets, h_prev, C_prev)

# 反向传播
dh_next, dC_next = lstm.backward()

# 更新权重
lstm.update_weights(learning_rate=0.001)

# 采样文本
text = lstm.sample(idx_to_char, seed_char_idx, length=200, temperature=0.8)
```

---

## 超参数指南

| Parameter      | Typical Range | Description                           | Too Low                | Too High               |
|----------------|---------------|---------------------------------------|------------------------|------------------------|
| `hidden_size`  | 64-512        | Number of LSTM units                  | Can't capture patterns | Overfits, slow         |
| `seq_length`   | 25-100        | Length of training sequences          | No context             | Slow, memory issues    |
| `learning_rate`| 0.0001-0.01   | Step size for weight updates          | Slow learning          | Unstable, diverges     |
| `epochs`       | 10-100        | Full passes through dataset           | Underfits              | Overfits, time waste   |
| `temperature`  | 0.3-1.5       | Sampling randomness (generation only) | Too repetitive         | Too random/nonsensical |

### 建议起点
```python
hidden_size = 128
seq_length = 50
learning_rate = 0.001
epochs = 50
temperature = 0.8  # for sampling
```

---

## 常见问题与处理

### 1. 损失爆炸（NaN）
**现象：** 迭代几次后，损失变为 `nan`。

**可能原因：**
- 学习率过高
- 梯度爆炸

**处理方法：**
```python
# 降低学习率
learning_rate = 0.0001  # instead of 0.01

# 增加梯度裁剪（implementation.py 已包含）
np.clip(grad, -5, 5)
```

### 2. 损失不下降
**现象：** 损失一直很高，没有改善。

**可能原因：**
- 学习率过低
- 隐藏状态维度过小
- 遗忘门偏置没有初始化为 1

**处理方法：**
```python
# 提高学习率
learning_rate = 0.001

# 增大隐藏状态维度
hidden_size = 256

# 检查遗忘门偏置（__init__ 中应为 1.0）
self.bf = np.ones(hidden_size)  # Remember by default
```

### 3. 生成文本重复
**现象：** 模型不断生成相同的字符。

**可能原因：**
- 温度过低
- 模型可能停留在局部极小值附近
- 在小数据集上训练过度

**处理方法：**
```python
# 提高温度
temperature = 1.0  # instead of 0.5

# 增加多样性
sample = lstm.sample(idx_to_char, seed, length=200, temperature=1.2)

# 训练更久，或使用更多数据
```

### 4. 训练缓慢
**现象：** 每次迭代耗时过长。

**可能原因：**
- 隐藏状态维度过大
- 序列过长
- 词表过大

**处理方法：**
```python
# 减小隐藏状态维度
hidden_size = 64

# 缩短序列
seq_length = 25

# 使用批处理（当前实现尚未提供，但可以扩展）
```

---

## 调试检查表

出现问题时，检查：

- [ ] **数据是否正确加载？** 打印 `len(data)`、`vocab_size`。
- [ ] **形状是否匹配？** 原文使用 `input_size == vocab_size`、`output_size == vocab_size` 表达输入输出维度约束。
- [ ] **遗忘门偏置是否为 1？** 检查 `lstm.bf`（初始应全为 1）。
- [ ] **学习率是否合理？** 先尝试 0.001。
- [ ] **梯度是否裁剪？** 应位于 `[-5, 5]`。
- [ ] **损失是否下降？** 绘制损失曲线，检查下降趋势。
- [ ] **隐藏状态和细胞状态是否重置？** 将 `h_prev`、`C_prev` 初始化为零。
- [ ] **采样温度是否合适？** 尝试 0.7-1.0。

---

## 可视化

```python
from visualization import (
    plot_gate_activations,
    plot_cell_state_evolution,
    plot_gradient_flow_comparison,
    analyze_gate_patterns
)

# 绘制门的行为
plot_gate_activations(gates_dict, sequence_text)

# 观察细胞状态演化
plot_cell_state_evolution(cell_states, sequence_text)

# 比较梯度流
plot_gradient_flow_comparison(lstm_grads, rnn_grads)

# 分析门的模式
analyze_gate_patterns(gates_dict, sequence_text)
```

---

## 何时选择 LSTM 或其他模型

| Model       | Best For                              | Strengths                        | Weaknesses                  |
|-------------|---------------------------------------|----------------------------------|-----------------------------|
| **LSTM**    | Medium sequences (10-100 steps)       | Solves vanishing gradients       | Slower than GRU             |
|             | Interpretable gates needed            | Well-studied, stable             | Can't handle very long seqs |
| **GRU**     | Similar to LSTM                       | Faster (3 gates vs 4)            | Slightly less powerful      |
|             | When speed matters                    | Simpler architecture             |                             |
| **Vanilla RNN** | Very short sequences (<10 steps)  | Simple, fast                     | Vanishing gradients         |
|             | Real-time processing                  | Low memory                       | Can't learn long deps       |
| **Transformer** | Long sequences (100+ steps)       | Parallel processing              | Needs lots of data          |
|             | State-of-the-art performance          | Attention mechanism              | Memory intensive            |

**原文的经验建议：**
- 短序列（<10）：基本 RNN
- 中等序列（10-100）：LSTM 或 GRU
- 长序列（100 以上）：Transformer
- 很长的序列（1000 以上）：采用稀疏注意力的 Transformer

---

## 实用技巧

### 1. 初始化
```python
# 遗忘门偏置设为 1（初始倾向于保留信息）
self.bf = np.ones(hidden_size)

# 其他偏置可以为 0
self.bi = np.zeros(hidden_size)
self.bc = np.zeros(hidden_size)
self.bo = np.zeros(hidden_size)
```

### 2. 训练
- 从短序列开始，逐渐增加长度
- 使用梯度裁剪，防止梯度爆炸
- 同时观察损失与生成样本质量
- 定期保存检查点

### 3. 采样
- 可尝试温度 0.5-0.8，以生成更连贯的文本
- 可尝试温度 1.0-1.5，以增加生成多样性
- 尝试不同的起始字符
- 生成较长序列（500 字符以上），观察模式

### 4. 调试
- 打印门激活值（应位于 [0, 1]）
- 检查细胞状态的量级（不应爆炸）
- 绘制损失曲线（应呈下降趋势）
- 每 N 次迭代进行一次采样，观察进展

---

## 关键方程汇总

```
拼接输入：         concat = [h_{t-1}, x_t]

遗忘门：           f_t = σ(W_f · concat + b_f)
输入门：           i_t = σ(W_i · concat + b_i)
候选细胞状态：     C̃_t = tanh(W_C · concat + b_C)
输出门：           o_t = σ(W_o · concat + b_o)

更新细胞状态：     C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
更新隐藏状态：     h_t = o_t ⊙ tanh(C_t)

输出：             y_t = softmax(W_y · h_t + b_y)
损失：             L = -log(y_t[target])
```

**记住：**
- σ = sigmoid（取值 0 到 1，用于门）
- tanh = 双曲正切（取值 -1 到 1，用于候选值等）
- ⊙ = 逐元素乘法

---

## 参考资源

- **博客文章**：Colah 的《理解 LSTM 网络》（Understanding LSTM Networks）
- **原始论文**：Hochreiter 与 Schmidhuber（1997）
- **代码**：`implementation.py`（本仓库的 NumPy 版本）
- **练习**：见 `exercises/` 目录
- **可视化**：`visualization.py`

---

## 快速比较：LSTM 与 RNN

| Aspect              | Vanilla RNN                | LSTM                           |
|---------------------|----------------------------|--------------------------------|
| **Memory**          | Hidden state only          | Hidden + Cell state            |
| **Gates**           | None                       | 4 (forget, input, cell, output)|
| **Gradient Flow**   | Multiplicative (vanishes)  | Additive (preserved)           |
| **Long-term Deps**  | Poor (<10 steps)           | Good (100+ steps)              |
| **Complexity**      | Simple                     | More complex                   |
| **Training Speed**  | Fast                       | Slower                         |
| **Parameters**      | Few                        | 4x more                        |

---

练习见 `exercises/` 目录。

## 原仓库示例核对

上面的 Python 快速示例沿用了原仓库的接口写法，但与当前 `implementation.py` 不匹配：构造函数需要 `vocab_size`、`hidden_size`、`seq_length`；`forward` 返回四项；`backward` 需要 `inputs`、`targets`、`cache`；`update_weights` 接受梯度字典。训练与生成命令的参数由当前脚本支持；生成时还必须能读取训练时的数据文件以重建字符映射。为便于追溯，上述 Python 接口示例未在翻译中静默改写，不能当作已通过运行验证的代码；实际入口以 `python train_minimal.py --help` 和 Notebook 为准。序列长度、速度及参数量比较是经验性简化，具体限制见 README 的技术核对说明。
