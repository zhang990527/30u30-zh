# 第 2 天：理解长短期记忆网络（LSTM）

> *《理解 LSTM 网络》（Understanding LSTM Networks）* —— Christopher Olah（2015）

**原文：** http://colah.github.io/posts/2015-08-Understanding-LSTMs/

**用时：** 3-5 小时
**预备知识：** 第 1 天（基本循环神经网络），基础微积分
**代码：** 纯 NumPy（无需 PyTorch）

---

## 这篇文章实际讲什么

Colah 的博客是一篇通过图示讲解架构的文章。他逐步解释 LSTM 如何工作，其中的图示后来成为常用参考。你在网上见过的 LSTM 示意图，很可能就源于这篇文章。

原文不介绍训练、基准测试或实用技巧，而是专注于理解架构，以及它为什么能缓解梯度消失问题。这些图示降低了理解 LSTM 的门槛。

这很重要，因为 LSTM 在约 2015-2017 年是主流序列模型；理解它的门控机制，也有助于学习注意力机制（第 13 天及以后）。

---

## 核心思想

**问题：** 基本循环神经网络（vanilla RNN）在随时间反向传播（BPTT）时容易发生梯度消失，因此难以学习长程依赖。

**解决思路：** 增加**细胞状态（cell state）**，提供一条通过加性更新传递信息的独立路径。三个门控制写入、保留和输出哪些信息。

Colah 在文章中用语言模型跟踪主语性别的例子解释门控。在处理 “I grew up in France... I speak fluent ___”（我在法国长大……我能流利地说___）时，细胞状态可以跨过中间词语保留 “France”，帮助模型预测 “French”。

逐门讲解、Colah 原文的例子与比喻，以及变体讨论，详见 [paper_notes.md](paper_notes.md)。

---

## Colah 原文的内容

以下为要点，详细说明见 paper_notes.md：

### 长期依赖问题
两个例子：“the clouds are in the ___”（云在___中，短程、较容易）与 “I grew up in France... I speak fluent ___”（我在法国长大……我能流利地说___，长程、较困难）。原文引用 Hochreiter（1991）及 Bengio 等（1994），解释这种困难的根源。

### 作为“传送带”的细胞状态
这是原文的核心比喻。细胞状态与隐藏状态并行传递，沿途只有“少量线性交互”。只要门控不主动修改它，信息就能沿这条路径保持不变。

### 三个门
- **遗忘门（forget gate）**：丢弃细胞状态中的哪些信息（sigmoid，0-1）
- **输入门（input gate）与候选细胞状态**：写入哪些新信息（sigmoid + tanh）
- **输出门（output gate）**：输出细胞状态中的哪些信息（sigmoid）

每一部分都通过跟踪主语性别的例子来说明。

### 变体
包括窥孔连接（peephole connections，Gers 与 Schmidhuber，2000）、耦合遗忘门与输入门，以及门控循环单元（GRU，Cho 等，2014）。原文引用 Greff 等（2015）关于各变体表现“基本相当”的结论，以及 Jozefowicz 等（2015）对超过 10,000 种架构的研究。

### 结论
原文认为注意力是“下一步方向”。这一判断写于 2015 年，比 Transformer 的提出早两年。

---

## 架构

### LSTM 单元

```
         ┌─────────────────────────────────┐
         │    LSTM Cell at time t          │
         │                                 │
    x_t ──┬──► Forget Gate (f_t)          │
         ││                                │
  h_{t-1}─┼──► Input Gate (i_t)           │
         ││    Cell Candidate (~C_t)       │
         ││                                │
  C_{t-1}─┼──► Cell State Update          │──► C_t
         ││    C_t = f_t ⊙ C_{t-1}        │
         ││         + i_t ⊙ ~C_t           │
         ││                                │
         │└──► Output Gate (o_t)           │
         │     h_t = o_t ⊙ tanh(C_t)      │──► h_t
         └─────────────────────────────────┘
```

### 逐步说明一个时间步内的计算

**输入：**
- `x_t` = 当前输入（采用独热编码的字符）
- `h_{t-1}` = 上一时间步的隐藏状态
- `C_{t-1}` = 上一时间步的细胞状态

**步骤 1：遗忘门**——遗忘什么？
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

输出位于 0 与 1 之间的数值。0 表示“完全遗忘这一维的信息”，1 表示“完全保留”。

**步骤 2：输入门与候选细胞状态**——存储什么新信息？
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

`i_t` 控制允许写入多少信息（0 到 1）；`C̃_t` 包含候选值（-1 到 1）。

**步骤 3：更新细胞状态**——更新长期记忆
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

这是关键方程。通过乘以 f_t 遗忘旧信息，再加入乘以 i_t 的候选新信息。`C_{t-1}` 到 `C_t` 的加性更新路径有助于缓解梯度消失。

**步骤 4：输出门**——输出什么？
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

决定将细胞状态中的哪些部分作为输出。

### 权重矩阵

设词表大小为 V、隐藏状态维度为 H：

```python
# 4 sets of weights, one per gate (each takes concatenated [h, x])
W_f = np.random.randn(H, V + H) * 0.01  # Forget gate
W_i = np.random.randn(H, V + H) * 0.01  # Input gate
W_C = np.random.randn(H, V + H) * 0.01  # Cell candidate
W_o = np.random.randn(H, V + H) * 0.01  # Output gate

# Biases
b_f = np.ones((H, 1))   # Initialize to 1! (remember by default)
b_i = np.zeros((H, 1))
b_C = np.zeros((H, 1))
b_o = np.zeros((H, 1))
```

这里有基本 RNN 递归变换部分约 4 倍的参数。每个权重矩阵的形状为 (H, V+H)，因为输入由 h_{t-1} 与 x_t 拼接而成。

`b_f = 1` 的初始化很重要：它使遗忘门初始时倾向于保留信息，随后再通过训练调整。这一做法来自 Jozefowicz 等（2015）。

### 缓解梯度消失

**基本 RNN 为什么容易失效：**

在 BPTT 中，梯度沿时间反向传递：
$$\frac{\partial L}{\partial h_0} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

每一项都涉及权重矩阵 $W_{hh}$。原文以特征值 < 1 来直观解释梯度的指数衰减。例如每步缩小 0.9 倍，10 步后为 $0.9^{10} = 0.35$，50 步后为 $0.9^{50} = 0.005$。

**LSTM 为什么有效：**

细胞状态更新提供了一条直接路径：
$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

这里每一维的 $f_t$ 都是学习得到的门值，而不是固定的权重矩阵。LSTM 可以在需要时学习 $f_t \approx 1$，沿细胞状态的直接路径保留梯度，避免每步都经过同一权重矩阵的乘法。

---

## 实现指南

### 前向传播（一个时间步）

```python
def lstm_step_forward(x, h_prev, C_prev, Wf, Wi, WC, Wo, bf, bi, bC, bo):
    """One LSTM time step."""
    
    # Concatenate input and previous hidden state
    combined = np.vstack([h_prev, x])  # Shape: (H+V, 1)
    
    # 1. Forget gate
    f = sigmoid(Wf @ combined + bf)  # Shape: (H, 1)
    
    # 2. Input gate + candidate
    i = sigmoid(Wi @ combined + bi)
    C_candidate = np.tanh(WC @ combined + bC)
    
    # 3. Cell state update
    C = f * C_prev + i * C_candidate
    
    # 4. Output gate + hidden state
    o = sigmoid(Wo @ combined + bo)
    h = o * np.tanh(C)
    
    # Cache for backward pass
    cache = (x, h_prev, C_prev, f, i, C_candidate, o, C, combined)
    
    return h, C, cache
```

### 反向传播（一个时间步）

```python
def lstm_step_backward(dh_next, dC_next, cache):
    """Backward pass through one LSTM step."""
    
    x, h_prev, C_prev, f, i, C_cand, o, C, combined = cache
    
    # Output gate gradients
    do = dh_next * np.tanh(C)
    do_raw = do * o * (1 - o)  # sigmoid derivative
    
    # Cell state gradient (from output + from future)
    dC = dh_next * o * (1 - np.tanh(C)**2) + dC_next
    
    # Forget gate gradients
    df = dC * C_prev
    df_raw = df * f * (1 - f)
    
    # Input gate gradients
    di = dC * C_cand
    di_raw = di * i * (1 - i)
    
    # Cell candidate gradients
    dC_cand = dC * i
    dC_cand_raw = dC_cand * (1 - C_cand**2)
    
    # Weight gradients
    dW_f = df_raw @ combined.T
    dW_i = di_raw @ combined.T
    dW_C = dC_cand_raw @ combined.T
    dW_o = do_raw @ combined.T
    
    # Bias gradients
    db_f, db_i, db_C, db_o = df_raw, di_raw, dC_cand_raw, do_raw
    
    # Gradients to pass backward
    dcombined = (Wf.T @ df_raw + Wi.T @ di_raw + 
                 WC.T @ dC_cand_raw + Wo.T @ do_raw)
    
    dh_prev = dcombined[:H]
    dx = dcombined[H:]
    dC_prev = dC * f  # Cell state gradient flows through forget gate
    
    return dx, dh_prev, dC_prev, dW_f, dW_i, dW_C, dW_o, db_f, db_i, db_C, db_o
```

关键语句是 `dC_prev = dC * f`：沿细胞状态直接路径传递的梯度只需乘以遗忘门值，无需乘以权重矩阵。这是 LSTM 能够保留梯度的重要原因。

---

## 实现说明

`implementation.py` 使用纯 NumPy 从零实现 LSTM，与第 1 天的基本 RNN 对应。

与第 1 天的主要区别：
- **4 组变换权重矩阵**，而非 1 组（遗忘门、输入门、候选细胞状态、输出门）
- **细胞状态**与隐藏状态一起传递
- **遗忘门偏置 = 1**（而非 0），有助于训练稳定
- **递归变换部分约 4 倍参数**：LSTM 计算较慢，但能处理更长的依赖

容易踩坑的地方：
- **漏传细胞状态**：时间步之间必须同时传递 `h` 和 `C`。漏传 `C` 会使模型无法正常学习。
- **门的顺序**：各门方程形式相近，常见错误是给某个门用了另一门的权重矩阵。
- **仍需梯度裁剪**：LSTM 能缓解梯度消失，但仍可能出现梯度爆炸。这里裁剪至 [-5, 5]。
- **数值稳定性**：沿用第 1 天的 softmax 技巧（取指数前先减去最大值）。

---

## 训练建议

### 1. 初始化很重要

```python
# Forget gate bias = 1 (Jozefowicz et al., 2015)
b_f = np.ones((hidden_size, 1))

# Xavier initialization for weights
W_f = np.random.randn(H, V+H) * np.sqrt(2.0 / (V+H))
```

使初始遗忘门倾向于保留信息，相当于让模型先“默认记住”，再学习何时遗忘。

### 2. 仍需梯度裁剪

```python
for grad in [dW_f, dW_i, dW_C, dW_o]:
    np.clip(grad, -5, 5, out=grad)
```

LSTM 能缓解梯度消失，但梯度仍可能爆炸。

### 3. 学习率

```python
learning_rate = 0.001  # Start lower than vanilla RNN
```

LSTM 递归变换部分的参数量约为基本 RNN 的 4 倍；这里建议谨慎选择学习率。

### 4. 序列长度

```python
seq_length = 50  # LSTMs can handle longer sequences
```

原文建议基本 RNN 使用约 20 步的序列，而 LSTM 可尝试 50-100 步。具体效果取决于任务与训练设置。

---

## 可视化

### 1. 门激活模式

绘制整个序列上的遗忘门、输入门、输出门值，观察 LSTM 实际学到了什么：

```python
gates = {'forget': [], 'input': [], 'output': []}

for t in range(seq_len):
    h, C, cache = lstm_step_forward(...)
    f, i, o = cache[3], cache[4], cache[6]
    gates['forget'].append(f)
    gates['input'].append(i)
    gates['output'].append(o)

plt.imshow(np.array(gates['forget']).T, cmap='RdYlGn', aspect='auto')
plt.title('Forget Gate (0=forget, 1=keep)')
```

### 2. 细胞状态演化

```python
cell_states = np.array(all_cell_states).T  # (hidden_size, seq_len)
plt.imshow(cell_states, cmap='RdBu', aspect='auto')
plt.colorbar(label='Cell state value')
plt.title('LSTM Cell State Evolution')
```

### 3. 梯度流对比

```python
plt.plot(lstm_grads, label='LSTM', linewidth=2)
plt.plot(rnn_grads, label='Vanilla RNN', linewidth=2, alpha=0.7)
plt.yscale('log')
plt.xlabel('Time step (backward)')
plt.ylabel('Gradient norm')
plt.title('Gradient Flow: LSTM vs RNN')
```

---

## 动手实践

### 快速开始

```bash
python train_minimal.py --data data/input.txt --epochs 50 --hidden-size 128
```

### 练习（位于 `exercises/`）

| # | Task | What You'll Get Out of It |
|---|------|--------------------------|
| 1 | Build LSTM from scratch | Understand the 4-gate forward pass and BPTT |
| 2 | Gate analysis | Visualize what forget/input/output gates learn |
| 3 | Ablation study | Remove gates one at a time — which matters most? |
| 4 | Long-range dependencies | Synthetic task: LSTM succeeds where vanilla RNN fails |
| 5 | GRU comparison | Implement GRU (2 gates), compare with LSTM |

解答位于 `exercises/solutions/`。建议先自己尝试，遇到困难后再查阅。

---

## 进一步探索

### LSTM 变体

1. **窥孔连接（Peephole Connections）**（Gers 与 Schmidhuber，2000）
   - 门可以直接访问细胞状态
   - 在部分任务上略有改进，但并非普遍采用

2. **门控循环单元（GRU）**（Cho 等，2014）
   - 将遗忘门与输入门的作用合并为一个“更新门”
   - 合并细胞状态与隐藏状态
   - 用两个门代替三个门，参数更少，表现通常相近
   - 练习 5 将实现这一结构

3. **双向 LSTM（Bidirectional LSTM）**
   - 沿正向与反向分别处理序列
   - 原文列举了 BERT、ELMo（BERT 的架构误述见下方核对说明）

### 何时选择 LSTM 或其他模型

| Model | Best For | Strengths | Weaknesses |
|-------|----------|-----------|------------|
| **Vanilla RNN** | Very short sequences (<10 steps) | Simple, fast | Vanishing gradients |
| **LSTM** | Medium sequences (10-100 steps) | Solves vanishing gradients, interpretable gates | Slower than GRU, sequential |
| **GRU** | Similar to LSTM | Faster (fewer parameters) | Slightly less expressive |
| **Transformer** | Long sequences (100+ steps) | Parallel training, attention | Needs more data, memory intensive |

原文的经验建议：短序列 → RNN，中等序列 → LSTM/GRU，长序列 → Transformer。上述步数不是模型能力的硬边界。

---

## 核心要点

1. **细胞状态是关键创新。** 它提供加性更新路径，使梯度更容易沿时间反向传递。

2. **门控策略通过学习得到。** 网络通过训练发现何时记忆、何时遗忘，无需手工制定规则。

3. **所比较变体的差别不大。** Greff 等（2015）在其研究条件下发现，常见 LSTM 变体的表现大致相近；加性细胞状态与可学习门控这一核心设计很重要。

4. **这是通向注意力的桥梁。** Colah 将注意力视为“下一步方向”。注意力最初与 LSTM 等循环模型结合，后来 Transformer 采用不依赖循环的架构（第 13 天及以后）。

---

## 本目录文件

| File | What It Is |
|------|-----------|
| `implementation.py` | Complete LSTM in NumPy, heavily commented |
| `train_minimal.py` | Training script with CLI args |
| `visualization.py` | Gate activation heatmaps, cell state plots, gradient flow comparison |
| `notebook.ipynb` | Interactive walkthrough — build, train, visualize |
| `exercises/` | 5 exercises with solutions |
| `paper_notes.md` | Detailed notes on Colah's post with ELI5 |
| `CHEATSHEET.md` | Quick reference for hyperparameters and debugging |

---

## 延伸阅读

- [Colah 的博客原文](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)：建议先读，图示尤其重要
- [LSTM 原始论文](https://www.bioinf.jku.at/publications/older/2604.pdf)：Hochreiter 与 Schmidhuber（1997）
- [《学习遗忘》（Learning to Forget）](https://ieeexplore.ieee.org/document/818041)：Gers 等（2000），引入遗忘门
- [《LSTM：搜索空间漫游》（LSTM: A Search Space Odyssey）](http://arxiv.org/pdf/1503.04069.pdf)：Greff 等（2015），比较 LSTM 变体
- [Jozefowicz 等（2015）](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)：测试超过 10,000 种 RNN 架构
- [PyTorch LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)：框架实现

---

**下一篇：**[第 3 天——循环神经网络正则化](../03_RNN_Regularization/)

## 技术核对说明

- 这里介绍的是 Colah 的技术博客，不是新的研究论文；NumPy 训练、超参数建议和梯度推导是仓库的教学补充。
- LSTM 通常称为“三个门与一个候选细胞状态”。原仓库将四组变换统称为“四个门”；本章保留四组计算，但不将候选状态误称为 sigmoid 门。
- 上面的细胞状态导数表示固定门值时的直接路径贡献。门本身依赖上一隐藏状态，因此不是整个计算图的完整总导数；LSTM 也不保证所有情况下梯度都不消失。遗忘门偏置为 1 并不意味着门值等于 1，零输入时 sigmoid(1) 约为 0.73。
- 总参数量还包含输出层，“4 倍”仅是递归变换部分的近似比较。权重矩阵特征值只是梯度问题的直观解释，完整分析还需考虑激活函数导数与雅可比矩阵乘积。
- 原仓库称双向 LSTM“用于 BERT、ELMo”，其中 BERT 实际使用 Transformer，ELMo 使用双向 LSTM。此处明确标出原有误述，不改动实现。序列长度阈值及性能比较是教学经验，不能当作普遍结论。
