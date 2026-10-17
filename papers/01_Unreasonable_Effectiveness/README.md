<a id="day-1-the-unreasonable-effectiveness-of-recurrent-neural-networks"></a>

# 第 1 天：循环神经网络令人惊叹的有效性（The Unreasonable Effectiveness of Recurrent Neural Networks）

> Andrej Karpathy（2015）——[原始博文](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

**学习时间：** 2–4 小时  
**先修知识：** Python 基础和初步的微积分直觉  
**代码：** 纯 NumPy，无需 PyTorch

---

<a id="what-this-post-is-actually-about"></a>

## 这篇博文究竟讲了什么

Karpathy 的博文展示了一件出人意料的事：让神经网络学习一个非常简单的任务——预测序列中的下一个字符——它学到的东西却远超预期。仅凭原始字符，不借助任何显式规则，模型便能学到拼写、语法、排版习惯，甚至特定领域的结构。

标题呼应 Eugene Wigner 于 1960 年发表的文章《数学在自然科学中令人惊叹的有效性》（The Unreasonable Effectiveness of Mathematics in the Natural Sciences）。Wigner 感叹数学在描述物理世界时有效得令人惊讶；Karpathy 则指出，字符级预测在学习语言结构时也有类似的表现。

这很重要，因为同一个核心思想——预测下一个词元（token）——也是 GPT 和现代语言模型的基础。第 1 天从这里开始。

---

<a id="the-core-idea"></a>

## 核心思想

**任务：** 根据已经出现的字符，预测下一个字符。

```
Input:  "The cat sat on th"
Target: "e"
```

为了做好预测，模型需要隐式学会：
- 哪些字符通常接在哪些字符之后，例如英语中的 'q' 后面几乎总是 'u'
- 词的边界：空格往往出现在特定模式之后
- 语法：例如主语后接动词
- 领域结构：C 代码的语句末尾有分号，莎士比亚剧本中的角色名使用大写字母

没有人把这些规则直接告诉模型。它们是在预测任务中逐渐形成的：要预测得好，就必须建模数据背后的结构。

---

<a id="what-karpathy-actually-showed"></a>

## Karpathy 实际展示了什么

<a id="shakespeare-44mb-training-data"></a>

### 莎士比亚作品（4.4MB 训练数据）

训练后，模型会生成这样的文本：

```
PANDARUS:
Alas, I think he shall be come approached and the day
When little srain would be attain'd into being never fed,
And who is but a chain and subjects of his death,
I should not sleep.
```

在没有显式指导的情况下，模型学到了：
- 角色名全部大写，后接冒号
- 对话采用缩进
- 诗行大致具有抑扬格节奏
- 词汇带有所属时代的风格

它也有明显错误："srain" 并不是单词，内容的含义也不连贯。模型学到的是莎士比亚作品的*形式*，而不是*内容*。这个区别很关键。

<a id="linux-kernel-source-code-474mb"></a>

### Linux 源代码（474MB）

模型生成的 C 代码*看起来*像是正确的：

```c
static void action_new_function(struct s_stat_info *wb)
{
    unsigned long flags;
    int lel_idx_bit = e->edd, *sys & ~((unsigned long) *FIRST_COMPAT);
    buf[0] = 0xFFFFFFFF & (bit << 4);
    ...
}
```

模型学会了括号、缩进、分号、变量声明、内核宏和注释风格。但它并不能生成可编译的代码：变量名看似合理，逻辑却没有意义。即便如此，字符级预测能够捕捉这么多句法结构，本身就是这里想说明的现象。

<a id="wikipedia"></a>

### 维基百科

模型会生成带有 `[[wiki links]]`、日期、地理信息和百科全书文风的文本。同样，它学到了结构，却不能保证事实准确。

---

<a id="the-architecture"></a>

## 网络结构

<a id="vanilla-rnn"></a>

### 基本循环神经网络（vanilla RNN）

**注意：** Karpathy 博文中的所有实验都使用长短期记忆网络（LSTM），而非基本 RNN。为了便于从零实现，本章采用基本 RNN，核心预测原则相同。第 2 天的 LSTM 引入门控机制，以缓解梯度消失问题。

```
Input:    h    e    l    l    o
          |    |    |    |    |
         [h0]-[h1]-[h2]-[h3]-[h4]
          |    |    |    |    |
Output:   e    l    l    o    _
```

在每个时间步 t：

```
h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
y_t = W_hy * h_t + b_y
p_t = softmax(y_t)
```

整个模型只有三个权重矩阵和两个偏置向量。

- `W_xh`（hidden x vocab）：将输入字符映射到隐藏空间
- `W_hh`（hidden x hidden）：将前一时刻的状态传递到当前时刻，也就是“记忆”连接
- `W_hy`（vocab x hidden）：将隐藏状态映射为字符预测分数
- `x_t`：当前字符的独热编码（one-hot encoding）
- `h_t`：隐藏状态，可理解为模型的“工作记忆”
- `p_t`：下一个字符的概率分布

隐藏状态 `h_t` 是模型对截至当前的整个序列的表示。它是固定大小的向量，必须编码模型掌握的上下文信息。当 hidden_size=100 时，就是试图用 100 个数表示这些英语上下文。

<a id="training-backpropagation-through-time-bptt"></a>

### 训练：随时间反向传播（BPTT）

把 RNN 展开 `seq_length` 步，再沿展开的计算链反向传播。梯度沿时间反向传递，因此会出现：

1. **梯度爆炸：** 反复乘以 W_hh 可能导致数值急剧增大。本实现将梯度逐元素裁剪到 [-5, 5]。
2. **梯度消失：** 反复乘以较小的量会使信号衰减。这使基本 RNN 难以学习长程依赖，也推动了 LSTM 的发展，第 2 天会继续讨论。

<a id="sampling"></a>

### 采样

生成时，将模型自己的输出重新作为输入。温度（temperature）控制概率分布的尖锐程度：

- **T < 1.0：** 分布更尖锐，倾向于高概率字符；重复更多，也更“保守”。
- **T = 1.0：** 使用模型学到的原始概率分布。
- **T > 1.0：** 分布更平坦，随机性和拼写错误增加，有时也更有创造性。

对于训练较好的模型，T=0.7–0.8 往往能生成较易读的文本。Karpathy 在 Paul Graham 随笔实验中演示了温度的影响：温度非常低时，模型会不断重复关于创业公司的句子。

---

<a id="implementation-notes"></a>

## 实现说明

`implementation.py` 基于 Karpathy 的 [min-char-rnn.py](https://gist.github.com/karpathy/d4dee566867f8291f086)，原实现约有 112 行 Python 代码。

主要选择：
- **纯 NumPy：** 无需深度学习框架，每一步运算都清晰可见。
- **Adagrad 优化器：** 按参数自适应调整学习率，在这里很实用，因为不同字符相关的参数可能需要不同的更新幅度。
- **独热编码：** 简单但占用空间较多，一个字符需要 vocab_size 维的向量；实际系统通常使用嵌入（embedding）。
- **梯度裁剪到 [-5, 5]：** 防止梯度爆炸；原说明指出，不裁剪时训练可能在数百步内发散。

容易出错的地方：
- **softmax 数值不稳定：** 指数运算前减去最大值，即 `exp(y - max(y))`，直接计算 exp 可能溢出。
- **隐藏状态初始化：** 每轮数据遍历开始时置零，遍历过程中继续传递状态；若每个批次都清零，模型无法利用跨批次的历史状态。
- **数据指针管理：** 输入和目标序列之间容易发生偏移一位的错误。

> 核对注：本章命令行参数 `--epochs` 在现有实现中实际控制训练更新次数，数据到末尾才重置指针与隐藏状态，并非每次循环都完整遍历语料。这里保留接口名称，不能将参数值直接理解为完整遍历数据集的轮数。

---

<a id="what-to-build"></a>

## 动手实践

<a id="quick-start"></a>

### 快速开始

```bash
python train_minimal.py --data data/tiny_shakespeare.txt --epochs 200
```

<a id="exercises-in-exercises"></a>

### 练习（位于 `exercises/`）

| 编号 | 任务 | 学习收获 |
|---|---|---|
| 1 | 从零构建 RNN | 理解前向传播、BPTT 和梯度裁剪 |
| 2 | 温度实验 | 建立采样与概率分布的直觉 |
| 3 | 使用自定义数据训练 | 理解数据领域如何影响学到的模式 |
| 4 | 损失可视化 | 调试训练过程、发现过拟合 |
| 5 | 作者分类器 | 将 RNN 特征用于下游任务 |

参考解答位于 `exercises/solutions/`。建议先独立尝试，遇到困难后再查看。

---

<a id="key-takeaways"></a>

## 主要收获

1. **预测下一个字符会促使模型学习结构。** 这并非因为我们显式设计了这些规则，而是因为准确预测需要建模数据模式。
2. **模型可以学到形式，而未必掌握含义。** 它生成的文本看似合理，却可能没有连贯意义；现代大语言模型也仍在一定程度上存在类似局限。
3. **基本 RNN 存在记忆瓶颈。** 固定大小的隐藏状态需要编码历史信息，因而难以处理长程依赖，例如在 200 个字符之后匹配起始括号。这也是第 2 天引入 LSTM 的动机。
4. **这是 GPT 的思想先驱之一。** 从字符 RNN 到 GPT，可以概括为：将 RNN 换成 Transformer，将字符换成子词词元，再把规模扩大 10,000 倍。“通过预测下一个词元来学习”这一核心思想并未改变。这里是教学概括，不是架构等价或规模定律。

---

<a id="files-in-this-directory"></a>

## 本目录文件

| 文件 | 内容 |
|---|---|
| `implementation.py` | 带有详细注释的 NumPy 字符级 RNN 完整实现 |
| `train_minimal.py` | 支持命令行参数的训练脚本 |
| `visualization.py` | 损失曲线、隐藏状态热力图和概率图 |
| `notebook.ipynb` | 逐步构建和训练模型的交互式教程 |
| `exercises/` | 5 个练习及参考解答 |
| `paper_notes.md` | 原始博文的精简笔记 |
| `CHEATSHEET.md` | 超参数与调试速查表 |

---

<a id="further-reading"></a>

## 延伸阅读

- [Karpathy 的原始博文](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)：建议首先阅读，讲解清晰。
- [min-char-rnn.py](https://gist.github.com/karpathy/d4dee566867f8291f086)：本实现所依据的约 112 行代码。
- [Colah 的《理解 LSTM 网络》（Understanding LSTM Networks）](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)：第 2 天的预习材料。
- [《使用循环神经网络生成序列》（Generating Sequences With RNNs）](https://arxiv.org/abs/1308.0850)：Alex Graves 关于 RNN 手写序列生成的论文。

---

**下一步：** [第 2 天——理解 LSTM 网络](../02_Understanding_LSTM/)
