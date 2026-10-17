<a id="paper-notes-the-unreasonable-effectiveness-of-rnns"></a>

# 原文笔记：循环神经网络令人惊叹的有效性

<a id="eli5-explain-like-im-5"></a>

## 通俗解释（ELI5）

<a id="the-story"></a>

### 一个类比

想象你正通过抄书学习写作，每次抄一个字母。最初，你只会写出随机字母；但抄过数百万个字母后，就开始注意到一些模式：

- 哪些字母通常相邻，例如英语中的 'q' 后面接 'u'。
- 如何正确拼写单词。
- 句子如何组织。
- 不同作者各自的写作风格。

**这就是字符级 RNN 的核心思想。** 它们通过大量样例学习预测下一个字符；在学习预测时，也学到了输入文本的结构。

注意：这个类比来自本教程，而不是 Karpathy 的原文。它试图说明的要点是，做好预测需要掌握数据结构。

---

<a id="what-the-post-actually-covers"></a>

## 原文实际讨论了什么

Karpathy 于 2015 年 5 月发表的这篇文章是技术博客，不是研究论文。作者在多个数据集上训练字符级语言模型，展示了一个简单目标——预测下一个字符——如何促使模型学到丰富的结构模式。

**原文开头明确说明的重要细节：所有实验都采用 LSTM，而不是基本 RNN。** 作者为了便于解释先介绍基本 RNN 的方程，随后说明会交替使用 RNN／LSTM 这两个名称，但文中所有实验都使用 LSTM。本章出于教学目的采用基本 RNN，原文的实验结果则来自 LSTM。

---

<a id="the-core-idea-from-the-post"></a>

## 核心思想（来自原文）

**任务：** 给定一个字符序列，预测下一个字符。

Karpathy 使用字符串 "hello" 和词表 "h, e, l, o" 举例：

```
输入 'h' → 模型应预测 'e'
输入 'e' → 模型应预测 'l'
输入 'l' → 模型应预测 'l'
输入 'l' → 模型应预测 'o'
```

训练采用独热编码、softmax 输出与交叉熵损失。RNN 每次处理一个字符，通过隐藏状态保留此前的信息。

**核心观察：** 为了预测好下一个字符，模型需要隐式学会拼写、语法、排版和特定领域的结构。没有人把这些规则写进程序，它们是在预测任务中学到的。

原文还链接到 [min-char-rnn.py](https://gist.github.com/karpathy/d4dee566867f8291f086)，用约 100 行 Python／NumPy 实现核心思想。本章实现基于该代码。

---

<a id="the-experiments-all-six"></a>

## 六组实验

Karpathy 在六种数据集上训练，这是博文的主体。

<a id="1-paul-graham-essays-1mb"></a>

### 1. Paul Graham 随笔（约 1MB）

**设置：** 2 层 LSTM，512 个隐藏单元，dropout 0.5。

生成文本带有 Paul Graham 讨论创业公司时的风格：句式、词汇，甚至 [2] 这样的编号引用习惯。含义并不连贯，但捕捉到了*文风*。

**本节中的温度演示：**

- 低温时，模型不断重复 “is that they were all the same thing that was a startup is that they were all the same thing that was a startup”，即反复生成关于创业公司的同一句话，陷入循环。
- 温度更高时，文本更多样，但拼写错误也更多。

原文在这一部分详细讨论了温度。

> 核对注：原笔记将低温采样描述为“总是选择概率最高的下一个字符”。有限正温度下仍是随机采样，只是概率更加集中；不能等同于严格的贪心解码。

<a id="2-shakespeare-44mb"></a>

### 2. 莎士比亚作品（4.4MB）

**设置：** 3 层循环网络，每层 512 个隐藏单元；按原文统一说明，实际使用 LSTM。

以下保留模型生成的英文原样，以便观察拼写和排版：

```
PANDARUS:
Alas, I think he shall be come approached and the day
When little srain would be attain'd into being never fed,
And who is but a chain and subjects of his death,
I should not sleep.
```

在没有显式规则的情况下，模型学会了：

- 角色名全部大写，后接冒号。
- 对话缩进。
- 带有时代风格的词汇。
- 大致的抑扬格节奏。

它也有错误："srain" 不是单词，文本含义并不连贯。模型学到了莎士比亚的**形式**，而非**内容**。Karpathy 打趣说，这些样本让他几乎难以与真正的莎士比亚作品区分。

作者还对训练集进行最近邻比较，检查生成样本是否只是记忆并复制训练文本。

<a id="3-wikipedia-100mb-hutter-prize-dataset"></a>

### 3. 维基百科（100MB，Hutter Prize 数据集）

**设置：** 参考 Graves 等人的设置，训练一夜。

模型学到了：

- 标题、项目列表、`[[wiki links]]` 等格式。
- XML 结构，包括成对标签与嵌套。
- 看似合理但实际不存在的 URL；Karpathy 指出模型“幻觉”出了一个 yahoo URL。
- 期刊引用模板。

生成过程会在散文、结构化标记和原始 XML 之间切换，并体现相应的格式习惯。

> 核对注：原笔记将维基格式统称为 Markdown；`[[wiki links]]` 等属于维基标记，不能与 Markdown 语法完全等同。

<a id="4-algebraic-geometry-latex-16mb"></a>

### 4. 代数几何 LaTeX（16MB）

**设置：** 与同实验室的 Justin Johnson 一同训练。

生成的 LaTeX“几乎能够编译”。人工修正后，它可以呈现貌似合理的数学内容，包括公式环境、定理／引理结构和数学符号。常见错误是以 `\begin{proof}` 开头，却用 `\end{lemma}` 结束：这种较远距离的配对关系超出了模型可靠跟踪的能力。

<a id="5-linux-source-code-474mb"></a>

### 5. Linux 源代码（474MB）

**设置：** 取 Linux GitHub 仓库中的源文件与头文件。使用 3 层 LSTM、约 1000 万参数，训练数天。

生成样本原样保留：

```c
/*
 * Increment the size file of the new incorrect UI_FILTER group information
 * of the size generatively.
 */
static int indicate_policy(void)
{
  int error;
  if (fd == MARN_EPT) {
    /*
     * The kernel blank will coeld it to userspace.
     */
    if (ss->segment < mem_total)
      unblock_graph_and_set_blocked();
    else
      ret = 1;
    goto bail;
  }
  ...
}
```

它能模仿括号、缩进、分号、指针写法、内核宏、注释、`#include`，甚至逐字符模仿 GPL 许可头。

错误包括使用未定义变量、声明后不使用的变量、从 `void` 函数返回值，以及比较 `tty == tty` 这样的恒真条件。Karpathy 明确表示，他不认为这些代码能够编译。

有时模型仿佛“决定开始生成一个新文件”，先输出完整 GPL 许可头、包含语句与宏，再开始正文代码。

<a id="6-baby-names-8000-names"></a>

### 6. 婴儿名字（8000 个名字）

这是一个小实验。模型生成 "Rudi"、"Levette"、"Berice"、"Lussa" 等看似合理的新名字，约 90% 未出现在训练数据中。Karpathy 开玩笑说，它也许能为新创业公司取名提供灵感。

---

<a id="understanding-whats-going-on"></a>

## 理解模型在做什么

这部分经常被忽略，但提供了很有价值的观察。

<a id="training-evolution-war-and-peace"></a>

### 训练过程中的变化：《战争与和平》

作者展示了生成样本随训练的演化：

- **迭代 100 次：** 仍是随机字符组合，但开始学会用空格分隔词。
- **迭代 300 次：** 开始掌握引号与句号的用法。
- **迭代 500 次：** 可以拼写 "we"、"He"、"His"、"and" 等常见短词。
- **迭代 700 次：** 更像英语，但仍有许多拼写错误。
- **迭代 1200 次：** 出现引号、问号与更长的词。
- **迭代 2000 次：** 出现拼写正确的单词、名字和引语。

**作者的观察：** 模型先发现词与空格的总体结构，再开始学习单词，先短词、后长词；跨越多个词的主题要到更后面才出现。

这一过程体现了模型如何逐步建立不同层次的结构。

<a id="hidden-state-visualization"></a>

### 隐藏状态可视化

处理维基百科文本时，作者可视化 LSTM 隐藏状态中的单个神经元。大多数单元的行为难以解释，但约 5% 学到了具体、有趣的功能：

1. **URL 检测单元：** 在 URL 内部激活，在外部关闭。
2. **维基链接检测单元：** 在 `[[ ]]` 内部激活，且要等第二个 `[` 出现；单独一个 `[` 还不够。
3. **位置跟踪单元：** 在 `[[ ]]` 范围内近似线性变化，表示当前已经走过链接文本的多长部分。
4. **“www”计数单元：** 在 "www" 的第一个 "w" 之后关闭，可能帮助模型判断何时停止输出 "w"。

作者还发现了**引号检测单元**，跟踪当前位置是否位于引号内部。

关键在于：我们没有硬编码“预测下一个字符时，记录自己是否处于引号内可能有用”。只是用原始数据训练 LSTM，模型便自行学到了跟踪这一信息。

---

<a id="the-math"></a>

## 数学表达

基本 RNN 在每个时间步进行以下运算。

<a id="hidden-state-update"></a>

### 隐藏状态更新

```
h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
```

- `h_t`：时间 t 的新隐藏状态。
- `x_t`：当前输入，即独热编码字符。
- `h_{t-1}`：上一隐藏状态。
- `W_xh, W_hh`：需要学习的权重矩阵。
- `tanh`：将数值压缩到 [-1, 1] 范围。

<a id="output"></a>

### 输出

```
y_t = W_hy * h_t + b_y
p_t = softmax(y_t)
```

- `y_t`：各个可能的下一个字符对应的原始分数。
- `p_t`：下一个字符的概率分布。

Karpathy 在原文中介绍了基本 RNN 的递归更新。实际实验采用更复杂的 LSTM 更新，而本章为便于从零实现而使用基本 RNN。

> 核对注：原笔记用“四个门代替一个 tanh”来概括 LSTM，不够准确。常见 LSTM 有输入门、遗忘门、输出门三个门，以及候选细胞状态计算；不能把四个计算分支都叫作门。

---

<a id="the-title"></a>

## 标题的含义

标题呼应 Eugene Wigner 于 1960 年发表的《数学在自然科学中令人惊叹的有效性》（The Unreasonable Effectiveness of Mathematics in the Natural Sciences）：数学在描述物理世界时为何如此有效？

Karpathy 的类比是，字符级预测在学习语言结构时也有效得令人惊讶。任务只是预测一个字符，模型却能学到拼写、语法、排版和领域习惯。

---

<a id="what-the-post-gets-right"></a>

## 原文的优点

- 以演示为主：莎士比亚、Linux 和 LaTeX 输出比抽象宣称更直观。
- 如实展示局限：检查是否记忆训练数据，指出代码不可编译，并承认语义不连贯。
- 隐藏状态可视化提供了研究线索，而不仅是效果展示。
- 温度说明具有实践依据，便于操作。

<a id="what-the-post-doesnt-cover"></a>

## 原文没有深入讨论的内容

- 为什么 LSTM 比基本 RNN 更有效；这留给 Colah 的文章，也就是第 2 天。
- 对隐藏状态究竟计算什么的形式化分析。
- 困惑度等定量指标或标准基准结果。
- 与 n-gram 基线的比较；Yoav Goldberg 后来做了比较，指出某些生成效果也可由 n-gram 达到。
- 与词级模型的联系；原文只简要提及词级模型效果更好，并认为这可能是暂时现象。
- 规模定律或对未来模型的预测。

---

<a id="looking-back-our-retrospective-not-in-the-post"></a>

## 回顾：本教程的后见之明，并非原文结论

十年后回看，从字符 RNN 到现代大语言模型的发展脉络更为清楚：

| Karpathy 的设置（2015） | 现代大语言模型（2024，原笔记的概括） |
|---|---|
| 字符级 | 子词词元（BPE） |
| LSTM | Transformer |
| 隐藏状态 | 注意力与 KV 缓存 |
| 约 10M 参数 | 约 100B 及以上参数 |
| 单张 GPU、数天 | 数千张 GPU、数周 |
| 生成形式，未必有意义 | 原笔记认为通常能兼顾形式和含义 |

> 核对注：此表是教学性概括，不是所有现代模型的共同配置，也不证明模型“理解”了含义。隐藏状态与 KV 缓存并不构成严格的一一对应。

**通过预测下一个词元来学习**这一核心思想仍然存在，但这条联系是回顾性的。Karpathy 并没有在文中预测 GPT，而是在展示 RNN 进行字符预测时的意外能力。

---

<a id="questions-worth-thinking-about"></a>

## 值得思考的问题

以下问题来自原文或其展示的实验：

1. 模型为何能学会打开和关闭括号，却无法始终跟踪配对的具体类型？例如 LaTeX 中的 `\begin{proof}` → `\end{lemma}`。
2. 约 5% 的可视化单元呈现出可解释功能，其余 95% 在做什么？
3. 《战争与和平》训练中，模型先学短词，再学长词，最后出现主题，这说明序列结构如何被编码？
4. 对某些任务，n-gram 模型也能生成外观相似的文本。那么 RNN 学到了哪些 n-gram 不容易捕捉的东西？

---

**下一步：** [第 2 天——理解 LSTM 网络](../02_Understanding_LSTM/)，学习如何让循环网络记住更长的序列。
