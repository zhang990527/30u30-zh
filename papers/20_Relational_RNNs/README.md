# 第 20 天：关系循环神经网络（Relational Recurrent Neural Networks）

> Adam Santoro 等（2018）— [《关系循环神经网络》（Relational recurrent neural networks）](https://arxiv.org/abs/1806.01822)

**用时：**4–6 小时  
**前置知识：**第 19 天（关系网络）、注意力机制基础  
**代码：**PyTorch

---

## 这篇论文究竟研究什么

深度学习模型往往难以**跨时间推理**。如果第 1 步告诉模型“红球在蓝盒子里”，第 100 步再问“红球在哪里？”，标准 RNN（如 LSTM）可能忘记“在……里面”这一特定关系。

论文提出**关系记忆核心（Relational Memory Core，RMC）**。它不把记忆只看作一个向量（如 LSTM 隐藏状态），而是看作一组**相互作用的槽位（slots）**。每个时间步，这些槽位用**自注意力（self-attention）**相互“交谈”，使模型在决定记住什么*之前*，先推理新旧信息之间的关系。

它将 RNN 的固定大小状态与 Transformer 的关系建模能力结合起来。

> 译注：上述是直观解释，不能视为对任意 LSTM 的能力上限或 RMC 具有可靠长期记忆的保证。本章是教学实现，不是论文全部实验的复现。

---

## 核心思想

核心思想是**循环自注意力（recurrent self-attention）**。

标准 Transformer 对输入历史应用注意力，计算成本随历史增长；RMC 则每一步对**固定大小的记忆矩阵**应用注意力。

1. **记忆矩阵：**隐藏状态是矩阵 $M$，例如 4 个槽位，每个大小为 128。
2. **交互：**每一步，槽位彼此关注。
3. **门控更新：**用类似 LSTM 的门控机制，将注意力处理后的信息写入槽位。

这使模型可以在固定记忆预算内进行“A 与 B 相关，B 又与 C 相关”等复杂关系推理。

---

## 作者实际展示了什么

作者比较了 RMC 与标准 LSTM 在长期关系理解任务上的表现：

1. **第 N 远（N-th Farthest）：**合成任务，需要记忆向量并比较相对距离。原仓库报告 RMC 准确率 **90%+**，而 LSTM 随序列增长表现恶化。
2. **WikiText-103：**语言建模中，RMC 以较少参数达到或超过 LSTM 的表现，展示结构化记忆的效率。
3. **强化学习：**在 Mini PacMan 等迷宫任务上优于标准智能体。

> 待核验：本段保留原仓库结果概述。正文与数据表未完成独立核验，不能将“90%+”当作本章训练结果。原仓库不同文件对 N-th Farthest 的描述不一致：本章生成器的任务定义应以 `implementation.py` 的实际标签计算为准，不能直接等同原论文基准。

---

## 架构

**关系记忆核心（RMC）**由三部分组成：

1. **多头点积注意力（Multi-Head Dot Product Attention，MHDPA）：**
   - 基于上一时刻记忆 $M_{t-1}$ 与输入 $x_t$ 计算查询（$Q$）、键（$K$）、值（$V$）。
   - 记忆槽位关注其他槽位及输入。
2. **逐行 MLP：**每个记忆槽位由小型前馈网络独立处理。
3. **门控更新：**使用类似 LSTM 的输入门和遗忘门计算最终记忆 $M_t$，帮助长序列训练保持稳定。

---

## 实现说明

- **记忆初始化：**可使用可训练参数或标准正态分布，但需注意槽位区分与尺度。
- **注意力实现：**可复用 PyTorch `MultiheadAttention`，要确认注意力施加在*记忆槽位*维度上。
- **门控：**按槽位应用，可用 `Linear` 层处理拼接后的输入与记忆。

> 一手实现核对：DeepMind Sonnet 的官方 `relational_memory.py` 使用单位矩阵经填充／截断构造初始状态，并支持逐元素、逐槽位或无门控。其默认遗忘偏置为 1.0；输入被投影后作为额外槽位参加注意力，再移除输入槽位。本仓库使用另一种简化路径；上面的初始化建议不是对官方实现的逐项复述。

---

## 基线与可视化

本章包含与标准 **LSTM 基线**的比较，用于观察关系任务上的差异。

### 1. 模型比较

训练脚本 `train_minimal.py` 支持同时训练 RMC 和 LSTM：

```bash
python papers/20_Relational_RNNs/train_minimal.py --model both --save_plot
```

- **RMC：**约 14k 参数（显式关系记忆）。
- **LSTM：**约 22k 参数（隐式循环状态）。
- **目标：**检验 RMC 是否能在长上下文任务上，以较少参数获得更好的关系推理表现。

> 译注：参数量随配置变化，以上为原仓库的近似描述；不能在未测量时断言比较已达到“严格”或显著性结论。

### 2. 注意力可视化

可以绘制自注意力权重，观察 RMC 的关注模式：

```bash
python papers/20_Relational_RNNs/train_minimal.py --model rmc --visualize
```

脚本生成热图 `attention_heatmap.png`，显示记忆槽位之间的注意力。

> 译注：热图是权重的可视化，不是对模型“思考过程”的直接或完整解释。

---

## 动手实现

### 快速开始

```bash
# 完成练习占位符后，验证注意力与门控实现
python exercises/exercise_01_attention.py
python exercises/exercise_02_gating.py

# 在本章距离任务上训练完整模型
python train_minimal.py
```

> 译注：原仓库的 5 个练习文件包含语法无效的未完成表达式，必须先完成 TODO 才能直接执行；参考解答可单独运行。这是原有练习状态，本次没有代填算法。

### 练习（位于 `exercises/`）

| # | 任务 | 学习目标 |
|---|------|----------|
| 1 | **点积注意力** | 手工实现核心自注意力机制 |
| 2 | **LSTM 式门控** | 实现记忆的门控更新 |
| 3 | **RMC 单步更新** | 组合注意力与门控 |
| 4 | **距离任务数据** | 编写基准任务的数据生成器 |
| 5 | **训练循环** | 将各部分组合并训练 |

> 译注：原 README 将第 2–5 项写为初始化、门控、构建 RMC 和数据，编号与实际文件不一致；上表按练习目录修正映射，原有初始化知识仍保留在实现说明中。

参考解答位于 `exercises/solutions/`，请先尝试自己解决。

---

## 要点

1. **记忆也包含交互：**不仅存储事实，还关联新旧信息。
2. **循环单元内部也可使用注意力：**将注意力放入循环更新，可在固定状态预算下增强关系建模。
3. **结构化状态：**将状态划分为槽位，有助于组织信息，例如槽位 1 表示对象 A、槽位 2 表示对象 B；这只是示意，不保证自动学到固定语义分工。

---

## 本目录文件

| 文件 | 内容 |
|------|------|
| `implementation.py` | `RelationalMemory` 类的完整教学实现 |
| `train_minimal.py` | 本章距离任务训练循环 |
| `paper_notes.md` | 详细笔记、类比与数学说明 |
| `notebook.ipynb` | RMC 交互式讲解 |
| `exercises/` | 分步实现任务 |

---

## 延伸阅读

- [原论文（Santoro 等，2018）](https://arxiv.org/abs/1806.01822)
- [DeepMind 博客](https://deepmind.google/discover/blog/relational-deep-reinforcement-learning/)：高层次介绍。
- [《注意力就是你所需要的一切》（Attention Is All You Need，Vaswani 等，2017）](https://arxiv.org/abs/1706.03762)：这里所用多头缩放点积注意力的来源。
- [DeepMind Sonnet 官方 RMC 源码](https://github.com/google-deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py)：本次已读取其源文件，未固定外部提交版本。

---

**上一章：**[第 19 天：关系推理](../19_relational_reasoning/)  
**下一章：**[第 21 天：神经消息传递](../21_Neural_Message_Passing/)
