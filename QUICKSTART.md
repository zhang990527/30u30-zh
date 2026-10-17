<a id="quick-start-guide"></a>

# 快速入门指南

欢迎来到 **30u30——30 天学习 30 篇文献**！🚀

<a id="installation"></a>

## 安装

```bash
# 克隆仓库
 git clone https://github.com/zhang990527/30u30-zh.git
cd 30u30-zh

# 安装依赖
pip install -r requirements.txt
```

入门示例可在 CPU 上运行，无需 GPU。个别后续章节可能有额外依赖，应查看对应章节说明。

> 云环境说明：已配置的虚拟环境可通过 `source /workspace/.30u30-runtime/activate.sh` 激活。根依赖清单中的可选 `jupyter_contrib_nbextensions` 与 Notebook 7 不兼容；云环境安装时跳过它，未更改依赖声明。运行记录见 [验收报告](LOCALIZATION_REPORT.md)。

---

<a id="day-1-get-started"></a>

## 第 1 天：开始学习

可以选择三种方式。

<a id="1-interactive-notebook-recommended-for-beginners"></a>

### 1️⃣ 交互式 Notebook（推荐初学者使用）

```bash
cd papers/01_Unreasonable_Effectiveness
jupyter notebook notebook.ipynb
```

逐个执行单元，跟随其中的解释学习。

<a id="2-train-from-command-line"></a>

### 2️⃣ 从命令行训练

```bash
cd papers/01_Unreasonable_Effectiveness
python train_minimal.py --data data/tiny_shakespeare.txt --epochs 200
```

> 实现核对：本脚本的 `--epochs` 实际指定更新次数，并非完整遍历语料的轮数。保留原接口名称，避免误读训练规模。

<a id="3-build-from-scratch"></a>

### 3️⃣ 从零实现

完成练习：

```bash
cd papers/01_Unreasonable_Effectiveness/exercises
python exercise_01_build_rnn.py
```

练习含待完成的占位代码；直接运行练习不等于已验证其算法实现正确。

---

<a id="folder-structure"></a>

## 目录结构

每篇文献对应的典型目录如下，具体文件以各章为准：

```
papers/XX_Paper_Name/
├── README.md              # 理论与示例的完整指南
├── paper_notes.md         # 通俗笔记
├── implementation.py      # 带详细注释的完整实现
├── train_minimal.py       # 精简训练脚本
├── visualization.py       # 绘图函数
├── notebook.ipynb         # 交互式 Jupyter Notebook
├── exercises/             # 练习
│   ├── README.md
│   └── solutions/         # 参考解答
└── data/                  # 示例数据
```

---

<a id="learning-path"></a>

## 学习路线

<a id="if-youre-new-to-deep-learning"></a>

### 深度学习初学者

1. 阅读 `README.md`，建立整体认识。
2. 阅读 `paper_notes.md`，理解通俗解释。
3. 打开 `notebook.ipynb`，执行单元。
4. 准备好后开始 `exercises/` 中的练习。

<a id="if-you-have-ml-experience"></a>

### 已有机器学习经验

1. 浏览 `README.md`，掌握核心思想。
2. 研究 `implementation.py`。
3. 用自己的数据运行 `train_minimal.py`。
4. 直接进入练习。

<a id="if-youre-advanced"></a>

### 进阶读者

1. 阅读 README 中链接的原始文献。
2. 检查 `implementation.py` 的细节。
3. 完成附加练习。
4. 继续下一天的内容。

---

<a id="tips-for-success"></a>

## 学习建议

✅ **不要赶进度：** 每次专注一天的内容。  
✅ **跟着敲代码：** 亲手输入，而不只是阅读。  
✅ **多做实验：** 改变超参数，尝试新数据。  
✅ **分享进度：** 使用 #30u30 记录学习过程。  
✅ **主动提问：** 参与 GitHub 讨论。

❌ **不要跳过基础：** 后续章节会依赖前面的知识。  
❌ **不要只复制粘贴：** 理解每行代码。  
❌ **不要因困难而灰心：** 有些文献确实更难。

---

<a id="troubleshooting"></a>

## 常见问题

<a id="issue-modulenotfounderror-no-module-named-numpy"></a>

### 提示 “ModuleNotFoundError: No module named 'numpy'”

安装依赖：

```bash
pip install -r requirements.txt
```

<a id="issue-training-is-slow"></a>

### 训练速度慢

教学实现主要以便于理解为目标，CPU 训练较慢是正常现象。可以：

- 减小隐藏层大小。
- 缩短序列长度。
- 使用更小的数据集。

<a id="issue-loss-is-not-decreasing"></a>

### 损失没有下降

- 检查学习率，尝试 0.01、0.1、0.5。
- 确认已启用梯度裁剪。
- 确认数据加载正确。

<a id="issue-generated-text-is-gibberish"></a>

### 生成文本没有意义

- 增加训练迭代次数。
- 增大隐藏层大小。
- 使用更多训练数据。
- 尝试不同温度，例如 0.5–1.0。

---

<a id="daily-routine"></a>

## 每日安排

建议安排：

- **30–45 分钟：** 阅读 README 与 paper_notes。
- **45–60 分钟：** 运行 Notebook 或训练模型。
- **30–45 分钟：** 完成一个练习。
- **15 分钟：** 分享结果并规划下一天。

**总计：约 2–3 小时／天。**

如果时间不够，可以暂时跳过练习、用两天学习一篇文献，或只选择最相关的章节。

---

<a id="contributing"></a>

## 参与贡献

发现问题，希望增加练习或改进解释？请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

<a id="resources"></a>

## 资源

- 📖 [完整阅读清单](ilya_30_papers.md)
- 💬 [GitHub 讨论区](https://github.com/zhang990527/30u30-zh/discussions)：提问与交流
- 🐛 [Issue](https://github.com/zhang990527/30u30-zh/issues)：报告问题
- 📊 [项目看板](https://github.com/zhang990527/30u30-zh/projects)：跟踪进度

---

<a id="whats-next"></a>

## 接下来学什么

学完第 1 天后，你应该理解：

- ✅ RNN 如何处理序列。
- ✅ 前向传播与随时间反向传播。
- ✅ 简单模型为什么也能学到复杂模式。

**下一天：** 学习 LSTM 如何缓解梯度消失问题。

准备好了吗？进入第 1 天：

```bash
cd papers/01_Unreasonable_Effectiveness
jupyter notebook notebook.ipynb
```

开始动手吧！🔥
