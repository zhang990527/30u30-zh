# 第 20 天速查：关系循环神经网络

## 核心思想（30 秒）

**问题：**标准 RNN（如 LSTM）将历史压缩到固定向量，长序列上可能丢失关系信息。  
**思路：**保留一组记忆槽位，让它们每一步更新前通过注意力相互“交谈”。  
**目标：**跨时间建模对象与关系，例如“t=5 看到的球与 t=100 看到的是同一个”。

**应记住的示意图：**

```text
       输入 x_t
          |
    +-----+-----+
    | 投影并复制 |  <-- 将输入与每个槽位拼接
    +-----+-----+
          |
    [ 槽位 1 ] [ 槽位 2 ] ... [ 槽位 N ]  （旧记忆 M_{t-1}）
          |        |              |
    +---------------------------------+
    |          多头自注意力            |  <-- “小组讨论”
    +---------------------------------+
          |        |              |
       [注意力处理后的信息]
          |        |              |
    +---------------------------------+
    |       门控更新（LSTM 风格）       |  <-- “写入／遗忘”
    +---------------------------------+
          |
    新记忆 M_t
```

> 译注：这是本仓库的简化示意。官方 Sonnet 实现将输入作为额外槽位，而非将其复制拼接到每个槽位的特征维。

---

## 快速开始

```bash
cd papers/20_Relational_RNNs

# 1. 训练并比较 RMC 与 LSTM
python train_minimal.py --model both --save_plot

# 2. 可视化注意力
python train_minimal.py --model rmc --visualize

# 3. 分步实现指南（须先完成练习占位符）
python exercises/exercise_01_attention.py
```

---

## RMC 的关键参数

| 参数 | 典型范围 | 作用 | 提示 |
|------|----------|------|------|
| `num_slots` | 4-16 | 记忆向量数量 | 更多槽位可跟踪更多“对象”，但计算更慢 |
| `slot_size` | 32-128 | 每个向量大小 | 总记忆大小 = 槽位数 × 向量大小 |
| `num_heads` | 4-8 | 注意力头数 | 允许同时关注不同*类型*的信息 |
| `num_blocks` | 1 | 堆叠的注意力处理块数 | 通常 1 个即可 |
| `gate_style` | LSTM | 输入门／遗忘门 | 原仓库建议将遗忘偏置初始化为 1.0 |

> 译注：此表是概念速查，部分参数名与当前构造函数不同；`gate_style` 的“LSTM”是结构说明，不一定是可直接传入的取值。

---

## 数学实现（代码片段）

### 1. 多头注意力（MHDPA）

```python
def attention(query, key, value):
    """
    标准缩放点积注意力。
    参数：
        query: [batch, heads, slots, d_k]
        key:   [batch, heads, slots, d_k]
        value: [batch, heads, slots, d_v]
    """
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, value), probs
```

### 2. 门控更新（LSTM 风格）

```python
def gated_update(m_old, m_attended, x_t):
    """
    m_old:      旧记忆
    m_attended: 自注意力结果
    x_t:        当前输入
    """
    # 门的输入包含旧记忆与新输入
    combined = torch.cat([m_old, x_t_tiled], dim=-1)

    f_gate = torch.sigmoid(W_f(combined) + b_f) # 遗忘
    i_gate = torch.sigmoid(W_i(combined) + b_i) # 输入

    # 候选记忆通常来自注意力输出
    candidate = torch.tanh(m_attended)

    # 最终更新
    m_new = f_gate * m_old + i_gate * candidate
    return m_new
```

> 这些片段依赖外部定义的维度、参数和中间变量，保留原仓库的教学用途，并非独立可执行程序。

---

## 常见问题与处理

### 损失不下降

```python
# 可能原因：梯度消失或记忆被冲淡
# 处理 1：检查遗忘门初始化
nn.init.constant_(self.gate_forget.bias, 1.0)

# 处理 2：检查 LayerNorm 的位置
# RMC 在循环内部、注意力之后应用 LayerNorm
x = self.norm1(x + self.attention(x))
```

### 模型运行缓慢

```python
# 注意力复杂度是 O(N^2)，N 为记忆槽位数
# 处理：保持较小的槽位数，例如 4-8。
# RMC 关注固定记忆，而不是持续增长的完整历史。
```

### “RuntimeError: size mismatch”

```python
# 可能原因：拼接轴错误
# 处理：复制输入，使其匹配槽位数
input_tiled = input_emb.unsqueeze(1).expand(-1, num_slots, -1)
memory_aug = torch.cat([memory, input_tiled], dim=-1)
```

---

## 与其他架构比较

| 特征 | LSTM | Transformer | 关系 RNN（RMC） |
|:---|:---|:---|:---|
| **记忆** | 固定向量 | 完整历史 | 向量集合（槽位） |
| **关系建模** | 隐式（门控） | 显式（注意力） | 显式（注意力） |
| **开销** | O(Seq) | O(Seq^2) | O(Seq x Slots^2) |
| **适用情形** | 短／中等序列 | 并行／离线 | 长序列／在线关系建模 |

> 译注：表中忽略隐藏维度等因素，“显式”只指交互结构，不是对推理正确性的保证；适用情形是教学性概括。

---

*论文细节见 [paper_notes.md](paper_notes.md)，实现指南见 [README.md](README.md)。*
