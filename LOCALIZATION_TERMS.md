# 简体中文术语与翻译约定

本文随本地化工作持续维护。译名表示本项目的选择，不声称是论文作者认可的官方中文名称。正文首次出现的重要概念保留英文；代码标识符、公式、数据集原始文本与模型生成样本保持原样。

| 英文 | 首选中文 | 语境与约束 | 核验依据 |
|---|---|---|---|
| recurrent neural network (RNN) | 循环神经网络 | vanilla RNN 译为基本循环神经网络；不能与 LSTM 混为一谈 | Karpathy，2015，原文 Getting fancy |
| long short-term memory (LSTM) | 长短期记忆 | 是 RNN 的一种；通常区分输入门、遗忘门、输出门及候选状态 | 原论文与 Day 2 来源待核验 |
| hidden state | 隐藏状态 | 与 LSTM 的 cell state（细胞状态）区分 | Karpathy，2015，递归更新公式 |
| backpropagation through time (BPTT) | 随时间反向传播 | truncated BPTT 为截断随时间反向传播 | Karpathy，2015，训练设置 |
| one-hot encoding | 独热编码 | 不改动词表索引与编码维度 | Karpathy，2015，hello 示例 |
| logits | 未归一化分数（logits） | 不是概率；经过 softmax 才得到概率分布 | Day 1 forward 与 softmax 实现 |
| cross-entropy loss | 交叉熵损失 | 区分逐词元平均损失与序列总损失 | Day 1 loss 实现 |
| gradient clipping | 梯度裁剪 | 本项目 Day 1 按元素裁剪；不称为按范数裁剪 | Day 1 backward 实现 |
| temperature | 温度 | 控制采样分布；低温不等于严格贪心解码 | Karpathy，2015，Temperature |
| token | 词元 | 可指字符、子词等，不一概译成“单词” | 按各章节实现核对 |
| embedding | 嵌入 | 区分嵌入向量与嵌入层 | 按各章节实现核对 |
| attention / self-attention | 注意力／自注意力 | scaled dot-product attention 为缩放点积注意力 | Vaswani 等，2017，原文待核验 |
| alignment | 对齐 | 翻译模型中可指源序列与目标序列间的软对齐 | Bahdanau 等，原文待核验 |
| regularization | 正则化 | 区分 normalization（归一化） | 按原论文语境核对 |
| residual connection | 残差连接 | skip connection 为跳跃连接；具体关系按结构说明 | He 等，原文待核验 |
| dropout | 随机失活（dropout） | 保留 dropout 便于检索；区分丢弃率与保留率 | Srivastava 等，原文待核验 |
| minimum description length (MDL) | 最小描述长度 | 不与最小编码长度的具体形式不加区分地互换 | MDL 章节原始来源待核验 |
| posterior collapse | 后验坍塌 | 不误译为数值溢出或训练崩溃 | VLAE 原文待核验 |
| policy / value function | 策略／价值函数 | PPO 和 RLHF 语境；reward 为奖励 | 对应原论文待核验 |
| micro-batch / mini-batch | 微批次／小批次 | GPipe 中两者不可混用 | GPipe 原文待核验 |

## 原始来源核验记录

- Andrej Karpathy，2015-05-21，*The Unreasonable Effectiveness of Recurrent Neural Networks*：本项目译为《循环神经网络令人惊叹的有效性》。这是技术博客，不是研究论文。
- 原始发布地址：https://karpathy.github.io/2015/05/21/rnn-effectiveness/
- 已读取作者网站对应的源文件：https://raw.githubusercontent.com/karpathy/karpathy.github.io/master/_posts/2015-05-21-rnn-effectiveness.markdown
- 核对内容：所有展示实验使用 LSTM；Paul Graham 实验采用 2 层、每层 512 个隐藏单元、dropout 0.5；莎士比亚语料 4.4MB；Linux 语料 474MB；低温降低采样多样性；约 5% 的状态单元呈现作者可解释的行为。
- 仓库 Day 1 使用基本 RNN，是教学简化，不能把原文的 LSTM 实验结果归到此实现名下。
- 上述读取对应当前访问到的作者网站源文件，未固定该外部仓库的提交版本；没有完成其他来源的核验前，不把它们记为“已核实”。

## 章节标题对照（核验状态见验收报告）

中文标题为本项目译名；博客、课程、学位论文与研究论文保持类型区别。此表用于一致性管理，不代表已逐篇核验外部原文。

| 天数 | 中文 | 英文 |
|---|---|---|
| 1 | 循环神经网络令人惊叹的有效性 | The Unreasonable Effectiveness of Recurrent Neural Networks |
| 2 | 理解 LSTM 网络 | Understanding LSTM Networks |
| 3 | 循环神经网络正则化 | Recurrent Neural Network Regularization |
| 4 | 通过最小化权重的描述长度保持神经网络简洁 | Keeping Neural Networks Simple by Minimizing the Description Length of the Weights |
| 5 | 最小描述长度原理入门教程 | A Tutorial Introduction to the Minimum Description Length Principle |
| 6 | 复杂性动力学第一定律 | The First Law of Complexodynamics |
| 7 | 量化封闭系统中复杂性的兴衰：咖啡自动机 | Quantifying the Rise and Fall of Complexity in Closed Systems: The Coffee Automaton |
| 8 | 使用深度卷积神经网络进行 ImageNet 分类 | ImageNet Classification with Deep Convolutional Neural Networks |
| 9 | 用于图像识别的深度残差学习 | Deep Residual Learning for Image Recognition |
| 10 | 深度残差网络中的恒等映射 | Identity Mappings in Deep Residual Networks |
| 11 | 利用空洞卷积进行多尺度上下文聚合 | Multi-Scale Context Aggregation by Dilated Convolutions |
| 12 | 随机失活：防止神经网络过拟合的简单方法 | Dropout: A Simple Way to Prevent Neural Networks from Overfitting |
| 13 | 注意力就是你所需要的一切 | Attention Is All You Need |
| 14 | 带注释的 Transformer | The Annotated Transformer |
| 15 | 通过联合学习对齐与翻译进行神经机器翻译 | Neural Machine Translation by Jointly Learning to Align and Translate |
| 16 | 顺序很重要：面向集合的序列到序列建模 | Order Matters: Sequence to Sequence for Sets |
| 17 | 神经图灵机 | Neural Turing Machines |
| 18 | 指针网络 | Pointer Networks |
| 19 | 用于关系推理的简单神经网络模块 | A Simple Neural Network Module for Relational Reasoning |
| 20 | 关系循环神经网络 | Relational Recurrent Neural Networks |
| 21 | 用于量子化学的神经消息传递 | Neural Message Passing for Quantum Chemistry |
| 22 | Deep Speech 2：端到端语音识别 | Deep Speech 2: End-to-End Speech Recognition |
| 23 | 变分有损自编码器 | Variational Lossy Autoencoder |
| 24 | GPipe：利用流水线并行高效训练巨型神经网络 | GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism |
| 25 | 神经语言模型的规模定律 | Scaling Laws for Neural Language Models |
| 26 | 柯尔莫哥洛夫复杂度与算法随机性 | Kolmogorov Complexity and Algorithmic Randomness |
| 27 | 机器超级智能 | Machine Super Intelligence |
| 28 | CS231n：用于视觉识别的卷积神经网络 | CS231n: Convolutional Neural Networks for Visual Recognition |
| 29 | 近端策略优化算法 | Proximal Policy Optimization Algorithms |
| 30 | 从人类偏好中学习深度强化学习 | Deep Reinforcement Learning from Human Preferences |
