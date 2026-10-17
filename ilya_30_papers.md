# Ilya Sutskever 的 30 篇基础文献阅读清单

*“如果你真正学会了这些内容，就会掌握当今重要知识的 90%。”*


> 中文标题是本项目译名；博客、课程、学位论文与研究论文分别按其实际类型理解。作者和年份主要保留原仓库记录；逐项来源核验状态见 [验收报告](LOCALIZATION_REPORT.md)，不能把保留书目信息视为已经全部核实。

---

## 单元 1：基础篇（序列与循环）

**1. 循环神经网络令人惊叹的有效性（The Unreasonable Effectiveness of Recurrent Neural Networks）**  
*Andrej Karpathy (2015)*  
帮助理解“预测下一个词元”与智能之间联系的直观材料。  
https://karpathy.github.io/2015/05/21/rnn-effectiveness/

**2. 理解 LSTM 网络（Understanding LSTM Networks）**  
*Christopher Olah (2015)*  
理解记忆机制。  
https://colah.github.io/posts/2015-08-Understanding-LSTMs/

**3. 循环神经网络正则化（Recurrent Neural Network Regularization）**  
*Zaremba, Sutskever et al. (2014)*  
如何让 RNN 有效训练，并防止过拟合。  
https://arxiv.org/abs/1409.2329

**4. 通过最小化权重的描述长度保持神经网络简洁（Keeping Neural Networks Simple by Minimizing the Description Length of the Weights）**  
*Hinton, Van Camp (1993)*  
探索“压缩与智能”联系的理论来源。  
https://www.cs.toronto.edu/~hinton/absps/colt93.pdf

**5. 最小描述长度原理入门教程（A Tutorial Introduction to the Minimum Description Length Principle）**  
*Grünwald (2004)*  
用数学形式描述压缩原理。  
https://arxiv.org/abs/math/0406077

**6. 复杂性动力学第一定律（The First Law of Complexodynamics）**  
*Scott Aaronson (2011)*  
从理论物理视角思考复杂性。  
https://scottaaronson.blog/?p=762

**7. 量化封闭系统中复杂性的兴衰：咖啡自动机（Quantifying the Rise and Fall of Complexity in Closed Systems: The Coffee Automaton）**  
*Aaronson et al. (2014)*  
建模复杂性先增长、后消退的过程，并讨论生命与智能为何可能出现在宇宙的“中间状态”。  
https://arxiv.org/abs/1405.6903

---

## 单元 2：深度学习的兴起（视觉与架构）

**8. 使用深度卷积神经网络进行 ImageNet 分类（ImageNet Classification with Deep Convolutional Neural Networks）**  
*Krizhevsky, Sutskever, Hinton (2012)*  
推动深度学习浪潮的标志性工作。  
https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html

https://arxiv.org/abs/1211.5063


**9. 用于图像识别的深度残差学习（Deep Residual Learning for Image Recognition）**  
*He et al. (2015)*  
如何训练更深的网络，并扩展网络深度。  
https://arxiv.org/abs/1512.03385

**10. 深度残差网络中的恒等映射（Identity Mappings in Deep Residual Networks）**  
*He et al. (2016)*  
改进信号传播。  
https://arxiv.org/abs/1603.05027

**11. 利用空洞卷积进行多尺度上下文聚合（Multi-Scale Context Aggregation by Dilated Convolutions）**  
*Yu, Koltun (2015)*  
如何在保留分辨率的同时获取更大范围的上下文。  
https://arxiv.org/abs/1511.07122

**12. 随机失活：防止神经网络过拟合的简单方法（Dropout: A Simple Way to Prevent Neural Networks from Overfitting）**  
*Srivastava et al. (2014)*  
常用的正则化方法。  
https://jmlr.org/papers/v15/srivastava14a.html

---

## 单元 3：Transformer 与推理

**13. 注意力就是你所需要的一切（Attention Is All You Need）**  
*Vaswani et al. (2017)*  
影响广泛的网络架构。  
https://arxiv.org/abs/1706.03762

**14. 带注释的 Transformer（The Annotated Transformer）**  
*Alexander Rush (2018)*  
在代码层面理解 Transformer。  
https://nlp.seas.harvard.edu/annotated-transformer/

**15. 通过联合学习对齐与翻译进行神经机器翻译（Neural Machine Translation by Jointly Learning to Align and Translate）**  
*Bahdanau, Cho, Bengio (2014)*  
Transformer 之前的经典注意力机制。  
https://arxiv.org/abs/1409.0473

**16. 顺序很重要：面向集合的序列到序列建模（Order Matters: Sequence to Sequence for Sets）**  
*Vinyals et al. (2015)*  
让网络理解某些任务中输入顺序并不重要，区分集合与序列。  
https://arxiv.org/abs/1511.06391

---

## 单元 4：专门化架构（记忆与图）

**17. 神经图灵机（Neural Turing Machines）**  
*Graves, Wayne, Danihelka (2014)*  
为神经网络配备可读写的外部记忆。  
https://arxiv.org/abs/1410.5401

**18. 指针网络（Pointer Networks）**  
*Vinyals, Fortunato, Jaitly (2015)*  
能够“指向”输入的网络，适用于组合问题。  
https://arxiv.org/abs/1506.03134

**19. 用于关系推理的简单神经网络模块（A Simple Neural Network Module for Relational Reasoning）**  
*Santoro et al. (2017)*  
https://arxiv.org/abs/1706.01427

**20. 关系循环神经网络（Relational Recurrent Neural Networks）**  
*Santoro et al. (2018)*  
https://arxiv.org/abs/1806.01822

**21. 用于量子化学的神经消息传递（Neural Message Passing for Quantum Chemistry）**  
*Gilmer et al. (2017)*  
原简介将图网络概括为“智能”的通用形式，并称其同样适用于生物、化学和语言。译注：该论文提出消息传递框架并研究量子化学预测，这不等于证明了智能的普适性或各领域效果相同。  
https://arxiv.org/abs/1704.01212

**22. Deep Speech 2：端到端语音识别（Deep Speech 2: End-to-End Speech Recognition）**  
*Amodei et al. (2015)*  
将端到端深度学习应用于音频。  
https://arxiv.org/abs/1512.02595

---

## 单元 5：生成模型与规模扩展

**23. 变分有损自编码器（Variational Lossy Autoencoder）**  
*Chen et al. (2016)*  
将信息论用于生成建模。  
https://arxiv.org/abs/1611.02731

**24. GPipe：利用流水线并行高效训练巨型神经网络（GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism）**  
*Huang et al. (2019)*  
如何在多张 GPU 上训练超大模型。  
https://arxiv.org/abs/1811.06965

**25. 神经语言模型的规模定律（Scaling Laws for Neural Language Models）**  
*Kaplan et al., OpenAI (2020)*  
研究 AI 的“物理规律”，在训练前预测性能。  
https://arxiv.org/abs/2001.08361

**26. 柯尔莫哥洛夫复杂度与算法随机性（Kolmogorov Complexity and Algorithmic Randomness）**  
*Shen, Uspensky, Vereshchagin*  
压缩的纯数学基础。  
https://www.lirmm.fr/~ashen/kolmbook-eng-scan.pdf

**27. 机器超级智能（Machine Super Intelligence）**  
*Shane Legg (2008)*  
从 DeepMind 联合创始人的视角讨论智能、对齐与安全。  
https://www.vetta.org/documents/Machine_Super_Intelligence.pdf

**28. CS231n：用于视觉识别的卷积神经网络（CS231n: Convolutional Neural Networks for Visual Recognition）**  
*斯坦福课程（Karpathy、李飞飞）*  
用于巩固相关知识的斯坦福课程。  
https://cs231n.github.io/

---

## 单元 6：强化学习与人类反馈

**29. 近端策略优化算法（Proximal Policy Optimization Algorithms）**  
*Schulman et al., OpenAI (2017)*  
ChatGPT 所用 RLHF 流程涉及的算法之一。  
https://arxiv.org/abs/1707.06347

**30. 从人类偏好中学习深度强化学习（Deep Reinforcement Learning from Human Preferences）**  
*Christiano et al., OpenAI (2017)*  
基于人类偏好学习的早期代表工作。译注：原列表将其简称为 Human Feedback；链接论文的标题是 Deep Reinforcement Learning from Human Preferences。  
https://arxiv.org/abs/1706.03741

---

## 🌟 附加论文：语言模型革命

*以下条目不在原清单中，是原项目补充的、产生广泛影响的后续工作。*

**附加 1：BERT：用于语言理解的深层双向 Transformer 预训练（BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding）**  
*Devlin et al., Google (2018)*  
双向预训练推动了自然语言处理的发展，侧重语言理解任务。  
https://arxiv.org/abs/1810.04805

**附加 2：语言模型是无监督多任务学习器（GPT-2）（Language Models are Unsupervised Multitask Learners (GPT-2)）**  
*Radford et al., OpenAI (2019)*  
原简介称语言模型能够完成“任何”任务，零样本能力随之出现。译注：这不是对任意任务都有效的证明；应结合论文的具体任务与评估结果理解。  
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

**附加 3：语言模型是少样本学习器（GPT-3）（Language Models are Few-Shot Learners (GPT-3)）**  
*Brown et al., OpenAI (2020)*  
以规模扩展为核心：175B 参数、涌现能力与上下文学习。  
https://arxiv.org/abs/2005.14165

**附加 4：训练计算最优的大语言模型（Chinchilla）（Training Compute-Optimal Large Language Models (Chinchilla)）**  
*Hoffmann et al., DeepMind (2022)*  
寻找模型规模与训练数据量之间合适的平衡。  
https://arxiv.org/abs/2203.15556

**附加 5：超大规模神经网络：稀疏门控混合专家层（MoE）（Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (MoE)）**  
*Shazeer et al. (2017)*  
现代稀疏激活模型的重要基础。  
https://arxiv.org/abs/1701.06538

---

## 其他资源

**事后经验回放（Hindsight Experience Replay）**  
*Andrychowicz et al., OpenAI (2017)*  
让智能体从失败经验中学习。  
https://arxiv.org/abs/1707.01495

**神经常微分方程（Neural Ordinary Differential Equations）**  
https://arxiv.org/abs/1806.07366

**WaveNet：原始音频生成模型（WaveNet: A Generative Model for Raw Audio）**  
https://arxiv.org/abs/1609.03499

**世界模型（World Models）**  
https://arxiv.org/abs/1803.10122
