<a id="30u30-master-ilyas-30-foundational-ai-papers-in-30-days"></a>

# 30u30：用 30 天学习 Ilya 推荐的 30 篇 AI 基础文献

[![许可证：CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg?style=for-the-badge)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![GitHub 星标](https://img.shields.io/github/stars/Ojhaharsh/30u30?style=for-the-badge&logo=github)](https://github.com/Ojhaharsh/30u30)
[![项目网站](https://img.shields.io/badge/Website-30u30-blue?style=for-the-badge&logo=googlechrome&logoColor=white)](https://ojhaharsh.github.io/30u30/)

> “如果你真正学会了这些内容，就会掌握当今重要知识的 90%。”——Ilya Sutskever
>
> 译注：这是原项目引用的推荐语，不是本项目验收或学习效果的量化保证。

> 完整实现 • 交互式 Notebook • 面向初学者 • 免费学习

**30u30 是一份开源学习指南，围绕 Ilya Sutskever 推荐的 30 篇 AI 基础文献展开，建议每天学习一篇。**
每章包含从零实现、详细笔记、带参考解答的交互练习和可视化。可以把它看作深度学习基础领域的 Rustlings：阅读原文、理解数学，再亲手实现。

> 本地化状态、核验依据及已知问题见 [验收报告](LOCALIZATION_REPORT.md)；术语与中英标题对照见 [术语表](LOCALIZATION_TERMS.md)。原项目的“完成／上线”指章节已提供，不代表所有运行入口和本地化均已验收。部分条目是博客、课程或学位论文，并非全部都是研究论文。

---

<a id="-all-units-are-complete-bonus-papers-are-coming-soon"></a>

## 🚀 原项目的全部单元已提供，附加论文即将推出

<a id="-unit-1-the-foundations-days-1-7---complete"></a>

### ✅ 单元 1：基础篇（第 1–7 天）——已提供

**第 1 天：《循环神经网络令人惊叹的有效性》（The Unreasonable Effectiveness of Recurrent Neural Networks）**
- 从零实现字符级 RNN（纯 NumPy）
- 带可视化的交互式 Jupyter Notebook
- 5 个循序渐进的练习及参考解答
**[开始第 1 天 →](papers/01_Unreasonable_Effectiveness/)**

**第 2 天：《理解 LSTM 网络》（Understanding LSTM Networks）**
- 从零实现包含 4 个门控／状态计算分支的完整 LSTM
- 门激活分析与可视化
- 5 个练习，包括 LSTM 与 GRU 的比较
**[开始第 2 天 →](papers/02_Understanding_LSTM/)**

**第 3 天：《循环神经网络正则化》（Recurrent Neural Network Regularization）**
- 从零实现随机失活、层归一化、权重衰减与早停
- 完整的正则化流程
- 5 个防止过拟合的练习
**[开始第 3 天 →](papers/03_RNN_Regularization/)**

**第 4 天：《通过最小化权重的描述长度保持神经网络简洁》（Keeping Neural Networks Simple by Minimizing the Description Length of the Weights）**
- 贝叶斯／噪声权重网络与 MDL 直觉
- 不确定性包络、压缩分析与帕累托前沿
- 5 个练习，涉及差距分析、beta 调节与蒙特卡洛推断
**[开始第 4 天 →](papers/04_Minimizing_Description_Length/)**

**第 5 天：《最小描述长度原理入门教程》（A Tutorial Introduction to the Minimum Description Length Principle）**
- 两部编码、序贯 MDL 与 NML 复杂度
- MDL、AIC、BIC 的比较及压缩分析
- 5 个练习，从基本 MDL 到模型选择比较
**[开始第 5 天 →](papers/05_MDL_Principle/)**

**第 6 天：《复杂性动力学第一定律》（The First Law of Complexodynamics）**
- 信息均衡、信道容量与演化动力学
- 包含 7 种可视化的复杂性演化模拟器
- 5 个练习，从香农熵到真实基因组分析
**[开始第 6 天 →](papers/06_Complexodynamics/)**

**第 7 天：《量化封闭系统中复杂性的兴衰：咖啡自动机》（Quantifying the Rise and Fall of Complexity in Closed Systems: The Coffee Automaton）**
- 细胞自动机的复杂性、热扩散与涌现行为
- 混沌理论、李雅普诺夫指数与信息流分析
- 5 个练习，涉及混沌边缘、模式分类与神经网络初始化
**[开始第 7 天 →](papers/07_coffee_automaton/)**

<a id="-unit-2-deep-learning-explosion-vision--architectures---complete"></a>

### ✅ 单元 2：深度学习的兴起（第 8–12 天）——已提供

**第 8 天：《使用深度卷积神经网络进行 ImageNet 分类》（ImageNet Classification with Deep Convolutional Neural Networks）**
- 推动深度学习浪潮的论文
- GPU 加速训练、ReLU 激活与随机失活正则化
- 5 个练习，涉及 GPU 影响分析、激活函数与数据增强
**[开始第 8 天 →](papers/08_alexnet/)**

**第 9 天：《用于图像识别的深度残差学习》（Deep Residual Learning for Image Recognition）**
- 通过跳跃连接支持训练超过 100 层的网络
- 恒等映射、残差块与梯度传播通路
- 5 个练习，涉及梯度消失、跳跃连接消融与深度分析
**[开始第 9 天 →](papers/09_resnet/)**

**第 10 天：《深度残差网络中的恒等映射》（Identity Mappings in Deep Residual Networks）**
- 预激活设计：BN → ReLU → Conv
- 理解操作顺序为何对超过 1000 层的网络至关重要
- 5 个练习，涉及预激活与后激活比较、信息流和极深网络
**[开始第 10 天 →](papers/10_resnet_v2/)**

**第 11 天：《利用空洞卷积进行多尺度上下文聚合》（Multi-Scale Context Aggregation by Dilated Convolutions）**
- 无需池化即可使感受野指数级扩大
- 密集预测、语义分割与 WaveNet 的基础
- 5 个练习，涉及感受野分析、空洞模式与上下文模块
**[开始第 11 天 →](papers/11_dilated_convolutions/)**

**第 12 天：《随机失活：防止神经网络过拟合的简单方法》（Dropout: A Simple Way to Prevent Neural Networks from Overfitting）**
- 常用神经网络正则化方法
- 反向缩放随机失活、用于不确定性估计的蒙特卡洛随机失活与集成解释
- 5 个练习，涉及随机失活实现、丢弃率扫描、空间随机失活与 MC 不确定性
**[开始第 12 天 →](papers/12_dropout/)**

<a id="-unit-3-the-transformer-era-days-13-16---complete"></a>

### ✅ 单元 3：Transformer 时代（第 13–16 天）——已提供

**第 13 天：《注意力就是你所需要的一切》（Attention Is All You Need）**
- 推动自然语言处理及其他领域变革的 Transformer
- 自注意力、多头注意力与位置编码
- 5 个练习，从缩放点积注意力到完整 Transformer，并含交互式可视化
**[开始第 13 天 →](papers/13_attention/)**

**第 14 天：《带注释的 Transformer》（The Annotated Transformer）**
- 从数学到 PyTorch，在代码层面理解 Transformer
- 包含训练基础设施的完整实现
- 5 个练习，涉及注意力、多头注意力、编码器、训练与推断
**[开始第 14 天 →](papers/14_annotated_transformer/)**

**第 15 天：《通过联合学习对齐与翻译进行神经机器翻译》（Neural Machine Translation by Jointly Learning to Align and Translate）**
- Transformer 出现之前的经典注意力机制
- Bahdanau 加性注意力、双向编码器与对齐可视化
- 5 个练习，涉及从零实现注意力、编码器—解码器、束搜索与可视化
**[开始第 15 天 →](papers/15_bahdanau_attention/)**

**第 16 天：《顺序很重要：面向集合的序列到序列建模》（Order Matters: Sequence to Sequence for Sets）**
- 处理集合，并通过指向输入位置输出序列
- 顺序无关编码与读取—处理—写入框架
- 5 个练习，涉及指针注意力、集合编码器、排序、凸包与旅行商问题
**[开始第 16 天 →](papers/16_order_matters/)**

<a id="-unit-4-specialized-architectures-days-17-22---complete"></a>

### ✅ 单元 4：专门化架构（第 17–22 天）——已提供

**第 17 天：《神经图灵机》（Neural Turing Machines）**
- 可微外部记忆
- 寻址机制：基于内容的寻址、插值、移位与锐化
- 5 个练习，涉及寻址逻辑、循环卷积与记忆更新
**[开始第 17 天 →](papers/17_neural_turing_machines/)**

**第 18 天：《指针网络》（Pointer Networks）**
- 可以“指向”输入的网络，适用于组合问题
- 聚焦输入的指针注意力、无放回采样与组合优化
- 5 个练习，涉及指针注意力、凸包格式与 TSP 代价分析
**[开始第 18 天 →](papers/18_pointer_networks/)**

**第 19 天：《用于关系推理的简单神经网络模块》（A Simple Neural Network Module for Relational Reasoning）**
- 通过成对处理对象进行视觉问答与物理推理
- g_theta 与 f_phi 模块、基于集合的归纳偏置
- 5 个练习，涉及对象对生成、Sort-of-CLEVR 逻辑与掩码
**[开始第 19 天 →](papers/19_relational_reasoning/)**

**第 20 天：《关系循环神经网络》（Relational Recurrent Neural Networks）**
- 在循环单元内部使用多头点积注意力（MHDPA）
- 关系记忆核心：每个时间步让记忆槽通过自注意力交互
- 5 个练习，涉及记忆注意力、槽间交互与序列建模
**[开始第 20 天 →](papers/20_Relational_RNNs/)**

**第 21 天：《用于量子化学的神经消息传递》（Neural Message Passing for Quantum Chemistry）**
- 统一图神经网络的消息、更新与读出框架
- 边网络、GRU 更新、Set2Set 读出与 QM9 基准
- 5 个练习，涉及消息函数、图构建与属性预测
**[开始第 21 天 →](papers/21_Neural_Message_Passing/)**

**第 22 天：《Deep Speech 2：端到端语音识别》（Deep Speech 2: End-to-End Speech Recognition）**
- 以端到端语音识别替代传统自动语音识别流水线
- 卷积、双向 GRU、CTC 损失、序列级批归一化与 SortaGrad
- 5 个练习，涉及声谱图特征、CTC 解码、RNN 批归一化、课程学习与完整流程
**[开始第 22 天 →](papers/22_deep_speech_2/)**

<a id="-unit-5-generative-models--scaling-days-23-28---complete"></a>

### ✅ 单元 5：生成模型与规模扩展（第 23–28 天）——已提供

**第 23 天：《变分有损自编码器》（Variational Lossy Autoencoder）**
- 研究如何缓解配备强大解码器的 VAE 中的后验坍塌
- 受限感受野（PixelCNN）与逆自回归流（IAF）
- 5 个练习，从掩码卷积到完整流先验
**[开始第 23 天 →](papers/23_variational_lossy_autoencoder/)**

**第 24 天：《GPipe：利用流水线并行高效训练巨型神经网络》（GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism）**
- 流水线并行、微批次与激活检查点
- 在有限硬件上训练超过 6B 参数的巨型模型
- 5 个练习，从微批次到完整流水线集成
**[开始第 24 天 →](papers/24_gpipe/)**

**第 25 天：《神经语言模型的规模定律》（Scaling Laws for Neural Language Models）**
- 模型规模、计算量与数据集规模之间的幂律关系
- 在模型规模与数据集规模之间分配计算预算
- 5 个练习，涉及规模定律计算、计算最优训练与数据集扩展
**[开始第 25 天 →](papers/25_scaling_laws/)**

**第 26 天：《柯尔莫哥洛夫复杂度与算法随机性》（Kolmogorov Complexity and Algorithmic Randomness）**
- 信息论的数学基础，以及“压缩与智能”的联系
- 从零实现霍夫曼编码与算术编码
- 5 个练习，涉及熵比较、NCD 相似性聚类与不可压缩性
**[开始第 26 天 →](papers/26_kolmogorov_complexity/)**

**第 27 天：《机器超级智能》（Machine Super Intelligence）**
- 通用智能（Υ）、柯尔莫哥洛夫复杂度的代理指标与智能体—环境循环
- 对随机智能体、强化学习智能体和预测型智能体进行形式化基准比较
- 5 个练习，涉及 Upsilon 计算、环境设计与复杂度不变性
**[开始第 27 天 →](papers/27_MSI/)**

**第 28 天：《CS231n：用于视觉识别的卷积神经网络》（CS231n: Convolutional Neural Networks for Visual Recognition）**
- 用 NumPy 从零实现卷积层（朴素版本与 im2col）、池化、ReLU 和全连接层
- VGGNet-16 参数分析、空间维度变化与架构案例研究
- 5 个练习，涉及卷积前向、池化反向传播、输出尺寸、参数计数与特征可视化
**[开始第 28 天 →](papers/28_CS231n/)**

<a id="-unit-6-deep-reinforcement-learning--alignment-days-29-30---complete"></a>

### ✅ 单元 6：深度强化学习与对齐（第 29–30 天）——已提供

**第 29 天：《近端策略优化算法》（Proximal Policy Optimization Algorithms）**
- ChatGPT 所用 RLHF 流程涉及的算法之一
- 从零实现截断替代目标与广义优势估计（GAE）
- 5 个策略约束与稳定性练习
**[开始第 29 天 →](papers/29_PPO/)**

**第 30 天：《从人类偏好中学习深度强化学习》（Deep Reinforcement Learning from Human Preferences）**
- 使 AI 行为更符合人类偏好
- 奖励建模、偏好损失与 PPO 集成
- 从零实现合成偏好评判器与完整训练循环
**[开始第 30 天 →](papers/30_RLHF/)**

---

<a id="-mission"></a>

## 🎯 项目使命

原项目希望提供全面、适合初学者的开源学习路径，带你学习塑造现代 AI 的基础文献。没有付费墙，也不设置知识门槛，专注于知识本身。

无论你准备转向 AI、正在求学，还是纯粹出于好奇，都可以将这里作为路线图。

<a id="-what-youll-find-here"></a>

## 📚 这里有什么

每篇文献都配有：

- **📖 深入讲解的 README**：结合生活类比进行完整说明。
- **💡 通俗笔记**：用易于理解的方式解释概念（ELI5）。
- **💻 实现**：清晰、带注释、便于在 CPU 上学习的代码。
- **🎨 可视化**：把抽象概念画出来。
- **🏋️ 练习**：亲自动手，并附参考解答。
- **📓 Notebook**：交互式 Jupyter 教程。
- **⚡ 快速开始**：精简训练脚本，便于短时间试运行。

<a id="-the-journey"></a>

## 🗺️ 学习路线

<a id="unit-1-the-foundations-days-1-7----complete"></a>

### 单元 1：基础篇（第 1–7 天）——✅ 已提供

*序列与循环*

| 天数 | 文献／章节 | 状态 | 核心概念 |
|---|---|---|---|
| 1 | [字符级 RNN](papers/01_Unreasonable_Effectiveness/) | 🚀 已上线 | 通过预测学习结构 |
| 2 | [理解 LSTM 网络](papers/02_Understanding_LSTM/) | 🚀 已上线 | 记忆机制 |
| 3 | [RNN 正则化](papers/03_RNN_Regularization/) | 🚀 已上线 | 提升 RNN 的泛化能力 |
| 4 | [最小化描述长度](papers/04_Minimizing_Description_Length/) | 🚀 已上线 | 压缩与智能的联系 |
| 5 | [MDL 原理教程](papers/05_MDL_Principle/) | 🚀 已上线 | 压缩的数学原理 |
| 6 | [复杂性动力学第一定律](papers/06_Complexodynamics/) | 🚀 已上线 | 复杂性的物理视角 |
| 7 | [咖啡自动机](papers/07_coffee_automaton/) | 🚀 已上线 | 复杂性的兴衰与智能 |

<a id="unit-2-deep-learning-explosion-days-8-12----complete"></a>

### 单元 2：深度学习的兴起（第 8–12 天）——✅ 已提供

*视觉、深度与改变领域的技术*

| 天数 | 文献／章节 | 状态 | 核心概念 |
|---|---|---|---|
| 8 | [AlexNet](papers/08_alexnet/) | 🚀 已上线 | 深度学习革命 |
| 9 | [ResNet](papers/09_resnet/) | 🚀 已上线 | 残差与跳跃连接 |
| 10 | [ResNet v2 的恒等映射](papers/10_resnet_v2/) | 🚀 已上线 | 预激活设计 |
| 11 | [空洞卷积](papers/11_dilated_convolutions/) | 🚀 已上线 | 空洞卷积 |
| 12 | [随机失活（dropout）](papers/12_dropout/) | 🚀 已上线 | 防止过拟合 |

<a id="unit-3-the-transformer-era-days-13-16----complete"></a>

### 单元 3：Transformer 时代（第 13–16 天）——✅ 已提供

*影响广泛的网络架构*

| 天数 | 文献／章节 | 状态 | 核心概念 |
|---|---|---|---|
| 13 | [注意力与 Transformer](papers/13_attention/) | 🚀 已上线 | 自注意力与 Transformer |
| 14 | [带注释的 Transformer](papers/14_annotated_transformer/) | 🚀 已上线 | 代码层面的 Transformer |
| 15 | [Bahdanau 注意力](papers/15_bahdanau_attention/) | 🚀 已上线 | 经典注意力机制 |
| 16 | [面向集合的序列建模](papers/16_order_matters/) | 🚀 已上线 | 集合到序列问题 |

<a id="-unit-4-specialized-architectures-days-17-22---complete"></a>

### 单元 4：专门化架构（第 17–22 天）——✅ 已提供

*记忆、图与推理*

| 天数 | 文献／章节 | 状态 | 核心概念 |
|---|---|---|---|
| 17 | [神经图灵机](papers/17_neural_turing_machines/) | 🚀 已上线 | 可微外部记忆 |
| 18 | [指针网络](papers/18_pointer_networks/) | 🚀 已上线 | 通过注意力选择输入位置 |
| 19 | [关系推理](papers/19_relational_reasoning/) | 🚀 已上线 | 对象之间的成对关系及 g_theta、f_phi 模块 |
| 20 | [关系 RNN](papers/20_Relational_RNNs/) | 🚀 已上线 | 循环结构内部的自注意力 |
| 21 | [神经消息传递](papers/21_Neural_Message_Passing/) | 🚀 已上线 | 图神经网络的 MPNN 框架 |
| 22 | [Deep Speech 2](papers/22_deep_speech_2/) | 🚀 已上线 | 基于 CTC 的端到端语音识别 |

<a id="-unit-5-generative-models--scaling-days-23-28---complete"></a>

### 单元 5：生成模型与规模扩展（第 23–28 天）——✅ 已提供

*从理论走向大规模模型*

| 天数 | 文献／章节 | 状态 | 核心概念 |
|---|---|---|---|
| 23 | [变分有损自编码器](papers/23_variational_lossy_autoencoder/) | 🚀 已上线 | 通过受限解码器及 IAF 研究后验坍塌 |
| 24 | [GPipe](papers/24_gpipe/) | 🚀 已上线 | 流水线并行 |
| 25 | [神经语言模型的规模定律](papers/25_scaling_laws/) | 🚀 已上线 | AI 规模扩展的规律 |
| 26 | [柯尔莫哥洛夫复杂度](papers/26_kolmogorov_complexity/) | 🚀 已上线 | 压缩与随机性的数学原理 |
| 27 | [机器超级智能](papers/27_MSI/) | 🚀 已上线 | 安全与智能的定义 |
| 28 | [CS231n 卷积神经网络](papers/28_CS231n/) | 🚀 已上线 | 从零构建 CNN 层并分析 VGGNet |

<a id="-unit-6-deep-reinforcement-learning--alignment-days-29-30---complete"></a>

### 单元 6：深度强化学习与对齐（第 29–30 天）——✅ 已提供

*从策略梯度到 RLHF*

| 天数 | 文献／章节 | 状态 | 核心概念 |
|---|---|---|---|
| 29 | [近端策略优化（PPO）](papers/29_PPO/) | 🚀 已上线 | ChatGPT 的 RLHF 流程所涉及的算法 |
| 30 | [基于人类反馈的强化学习（RLHF）](papers/30_RLHF/) | 🚀 已上线 | 人类偏好学习与早期 RLHF |

<a id="-bonus-papers-the-language-model-revolution-coming-soon"></a>

### 💻⏭️ 附加论文：语言模型革命（即将推出）

*大语言模型的现代发展。下列目录是原项目的计划入口，当前仓库尚未提供。*

| 论文 | 状态 |
|---|---|
| [BERT：用于语言理解的深层双向 Transformer 预训练](papers/bonus_BERT/) | 即将推出 |
| [GPT-2：语言模型是无监督多任务学习器](papers/bonus_GPT2/) | 即将推出 |
| [GPT-3：语言模型是少样本学习器](papers/bonus_GPT3/) | 即将推出 |
| [Chinchilla：训练计算最优的大语言模型](papers/bonus_Chinchilla/) | 即将推出 |

**[完整文献列表与原文链接 →](ilya_30_papers.md)**

<a id="-quick-start"></a>

## ⚡ 快速开始

```bash
# 克隆本仓库
 git clone https://github.com/zhang990527/30u30-zh.git
cd 30u30-zh

# 从第 1 天开始
cd papers/01_Unreasonable_Effectiveness
```

<a id="prerequisites"></a>

### 先修知识

**初学者：** 掌握 Python 基础即可，其余内容随章节学习。

**已有经验的读者：** 可以直接选择感兴趣的文献。

<a id="-how-to-use-this-repo"></a>

## 🎓 如何使用本仓库

**🎯 30 天挑战：**

- 每天学习一篇文献。
- 阅读 README。
- 运行代码。
- 完成练习。
- 使用 #30u30 分享学习进度。

**🔀 选择适合自己的路线：**

- **理论优先：** README → 笔记 → 代码。
- **代码优先：** Notebook → 实现 → README。
- **实践优先：** 练习 → 参考解答 → 深入讲解。

<a id="-why-this-project"></a>

## 🌟 为什么创建这个项目

网上有许多论文摘要，原项目希望进一步提供：

1. **亲手实现：** 不依赖一个虚构的 `import magic_ai_library` 掩盖细节。
2. **多条学习路线：** 可以从理论、代码或交互实践出发。
3. **可运行、可学习的实现：** 原项目以实用代码质量为目标；实际可用性以验收记录为准。
4. **适合初学者：** 将生活类比与严谨数学相结合。
5. **社区共同改进：** 反馈会影响后续内容。

**目标：** 成为优秀的免费 AI 基础学习资源。

如果项目对你有帮助，欢迎 ⭐ 点亮星标并分享给其他学习者！

---

<a id="-contributing"></a>

## 🤝 参与贡献

欢迎一起改进项目。

- 🐛 **发现问题？** [提交 Issue](https://github.com/zhang990527/30u30-zh/issues)。
- 💡 **有新想法？** [提交 Issue](https://github.com/zhang990527/30u30-zh/issues)，附上 `enhancement` 标签，或[发起讨论](https://github.com/zhang990527/30u30-zh/discussions)。
- 📝 **希望贡献代码？** 阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

每一份贡献都能帮助其他学习者。

<a id="-license"></a>

## 📜 许可证

CC BY-NC-ND 4.0：可免费阅读、学习，并在署名条件下分享原作品；不可商用，不允许传播改编作品。具体条款见 [LICENSE](LICENSE)，本地译文不改变原许可。

<a id="-acknowledgments"></a>

## 🙏 致谢

- **Ilya Sutskever**：提供原始阅读清单。
- **所有原文作者**：推动领域进步。
- **你**：参与这段学习旅程。

---

<a id="-stay-connected"></a>

## 💬 保持联系

- 🐦 **Twitter：** 用 **#30u30** 分享进度。
- 📧 **Issue：** [报告问题或提出功能请求](https://github.com/zhang990527/30u30-zh/issues)。
- 💬 **讨论区：** [参与交流](https://github.com/zhang990527/30u30-zh/discussions)。
- ⭐ **点亮仓库星标**，关注后续更新。

---

**准备开始了吗？**

→ **[第 1 天：字符级 RNN](papers/01_Unreasonable_Effectiveness/)**  
→ **[第 2 天：理解 LSTM 网络](papers/02_Understanding_LSTM/)**  
→ **[第 3 天：RNN 正则化](papers/03_RNN_Regularization/)**  
→ **[第 4 天：最小化描述长度](papers/04_Minimizing_Description_Length/)**  
→ **[第 5 天：MDL 原理教程](papers/05_MDL_Principle/)**  
→ **[第 6 天：复杂性动力学第一定律](papers/06_Complexodynamics/)**  
→ **[第 7 天：咖啡自动机](papers/07_coffee_automaton/)**  
→ **[第 8 天：AlexNet](papers/08_alexnet/)**  
→ **[第 9 天：ResNet](papers/09_resnet/)**  
→ **[第 10 天：ResNet v2 的恒等映射](papers/10_resnet_v2/)**  
→ **[第 11 天：空洞卷积](papers/11_dilated_convolutions/)**  
→ **[第 12 天：随机失活（dropout）](papers/12_dropout/)**  
→ **[第 13 天：注意力与 Transformer](papers/13_attention/)**  
→ **[第 14 天：带注释的 Transformer](papers/14_annotated_transformer/)**  
→ **[第 15 天：Bahdanau 注意力](papers/15_bahdanau_attention/)**  
→ **[第 16 天：面向集合的序列建模](papers/16_order_matters/)**  
→ **[第 17 天：神经图灵机](papers/17_neural_turing_machines/)**  
→ **[第 18 天：指针网络](papers/18_pointer_networks/)**  
→ **[第 19 天：关系推理](papers/19_relational_reasoning/)**  
→ **[第 20 天：关系 RNN](papers/20_Relational_RNNs/)**  
→ **[第 21 天：神经消息传递](papers/21_Neural_Message_Passing/)**  
→ **[第 22 天：Deep Speech 2](papers/22_deep_speech_2/)**  
→ **[第 23 天：变分有损自编码器](papers/23_variational_lossy_autoencoder/)**  
→ **[第 24 天：GPipe](papers/24_gpipe/)**  
→ **[第 25 天：神经语言模型的规模定律](papers/25_scaling_laws/)**  
→ **[第 26 天：柯尔莫哥洛夫复杂度](papers/26_kolmogorov_complexity/)**  
→ **[第 27 天：机器超级智能](papers/27_MSI/)**  
→ **[第 28 天：CS231n 卷积神经网络](papers/28_CS231n/)**  
→ **[第 29 天：近端策略优化（PPO）](papers/29_PPO/)**  
→ **[第 30 天：基于人类反馈的强化学习（RLHF）](papers/30_RLHF/)** 🆕 **← 也可从这里开始！**  

一起动手构建吧！🚀
