<a id="contributing-to-30u30"></a>

# 为 30u30 做贡献

感谢你帮助改进 Ilya 的 30 篇基础文献学习资源！🙏 本指南介绍如何有效参与。

---

<a id="ways-to-contribute"></a>

## 参与方式

<a id="-report-bugs"></a>

### 🐛 报告问题

代码有错误，或文档中有笔误？

1. 在 [Issue](https://github.com/zhang990527/30u30-zh/issues) 中确认是否已经有人报告。
2. 如果没有，请创建新 Issue，提供清晰标题、复现步骤、预期与实际行为，以及操作系统和 Python 版本。

<a id="-suggest-improvements"></a>

### 💡 提出改进建议

更好的解释、额外练习、更多可视化或代码优化，都可以在[讨论区](https://github.com/zhang990527/30u30-zh/discussions)或 [Issue](https://github.com/zhang990527/30u30-zh/issues) 中提出。

<a id="-improve-documentation"></a>

### 📝 改进文档

帮助修正笔误、补充示例、完善类比或翻译为其他语言，让解释更加清楚。

<a id="-contribute-code"></a>

### 💻 贡献代码

欢迎增加实现、练习和测试。

---

<a id="code-contribution-guidelines"></a>

## 代码贡献指南

<a id="before-you-start"></a>

### 开始之前

1. **检查已有 Issue：** 避免与其他贡献者重复工作。
2. **先讨论重大变更：** 提交 Issue 或发起讨论。
3. **每个 PR 聚焦一个问题：** 保持改动集中。

<a id="setting-up-development-environment"></a>

### 准备开发环境

```bash
# 在 GitHub 上派生仓库，再克隆自己的副本
 git clone https://github.com/YOUR_USERNAME/30u30.git
cd 30u30

# 创建分支
 git checkout -b feature/your-feature-name

# 安装依赖
pip install -r requirements.txt

# 进行修改
# ...

# 在对应目录测试修改，XX_Paper 是占位目录名
python papers/XX_Paper/implementation.py
```

<a id="code-style"></a>

### 代码风格

遵循以下原则。

<a id="1-clarity-over-cleverness"></a>

#### 1. 清晰优先，避免炫技

```python
# ✅ 推荐：清晰、易读
for i in range(len(data)):
    loss += compute_loss(data[i])

# ❌ 不推荐：这里使用了没有必要的复杂写法
loss = sum(map(lambda x: compute_loss(x), data))
```

<a id="2-heavy-comments-for-educational-code"></a>

#### 2. 教学代码提供充分注释

```python
# ✅ 推荐：解释为什么这样做
# 裁剪梯度，防止反向传播时梯度爆炸
# 不裁剪时，梯度可能沿时间指数级增大
np.clip(grad, -5, 5, out=grad)

# ❌ 不推荐：只重复代码做了什么
# 裁剪梯度
np.clip(grad, -5, 5, out=grad)
```

<a id="3-use-numpy-not-pytorchtensorflow"></a>

#### 3. 核心教学实现优先使用 NumPy

原项目指南主张尽可能采用便于 CPU 学习的代码：

```python
# ✅ 原指南推荐的从零实现方式
h = np.tanh(Wxh @ x + Whh @ h_prev + bh)

# ❌ 对于要求从零实现的核心运算，不要直接交给框架隐藏细节
h = torch.tanh(torch.mm(Wxh, x) + torch.mm(Whh, h_prev) + bh)
```

> 译注：原指南表述为“使用 NumPy，而非 PyTorch／TensorFlow”。当前仓库部分后续章节实际采用 PyTorch；贡献时应遵循所在章节的技术路线，不能把此条理解为整个仓库都不允许使用框架。

<a id="4-descriptive-variable-names"></a>

#### 4. 使用含义清楚的变量名

```python
# ✅ 推荐
hidden_size = 100
learning_rate = 0.01
char_to_idx = {...}

# ❌ 对教学解释不够直观
n = 100
lr = 0.01
c2i = {...}
```

<a id="5-format-code-consistently"></a>

#### 5. 保持格式一致

原项目使用 `black`：

```bash
pip install black
black papers/XX_Paper/*.py
```

仅做翻译时避免顺带大范围格式化，以便审查译文与行为变化。

<a id="file-organization"></a>

### 文件组织

新增文献章节时，参考：

```
papers/XX_Paper_Name/
├── README.md              # 完整指南
├── paper_notes.md         # 通俗笔记
├── implementation.py      # 核心实现
├── train_minimal.py       # 训练脚本
├── visualization.py       # 绘图工具
├── notebook.ipynb         # Jupyter Notebook
├── exercises/
│   ├── README.md
│   ├── exercise_01_*.py
│   └── solutions/
│       └── exercise_01_solution.py
└── data/
    └── sample_data.txt
```

<a id="documentation-standards"></a>

### 文档规范

<a id="readmemd-structure"></a>

#### README.md 的结构

1. 标题与原文链接。
2. 核心思想，用 2–3 句话概括。
3. 重要性与历史背景。
4. 先修知识。
5. 直觉解释与类比。
6. 带解释的数学推导。
7. 实现讲解。
8. 可尝试的实验。
9. 5 个实践练习。
10. 进阶主题。
11. 视频、博客等资源。

<a id="paper_notesmd-structure"></a>

#### paper_notes.md 的结构

1. 面向零基础读者的通俗概述（ELI5）。
2. 生活中的类比。
3. 简化解释的关键概念。
4. 原文为什么具有影响力。
5. 与现代 AI 的联系。

<a id="pull-request-process"></a>

### Pull Request 流程

1. **使用清楚的标题。** 例如：

   ```
   为第 1 天增加练习 3：自定义数据集训练
   修复 LSTM 实现中的梯度裁剪错误
   改进第 13 天的注意力机制解释
   ```

2. **说明改了什么、为什么改，以及审阅者如何验证。**
3. **关联相关 Issue。** GitHub 自动关联语法保持英文：

   ```
   Closes #42
   Relates to #38
   ```

4. **确认检查通过：** 代码运行无误、输出合理，并且没有破坏兼容性。
5. **等待审查：** 审查需要时间；请耐心接受反馈，并及时完成必要修改。

---

<a id="writing-guidelines"></a>

## 写作指南

<a id="tone"></a>

### 语气

- **专业且易懂：** 专业术语需要解释。
- **鼓励学习：** 让读者感到目标可以逐步达到。
- **注重实践：** 围绕理解与亲手实现展开。

<a id="examples-of-good-vs-bad-writing"></a>

### 写作示例

<a id="-good---clear-and-educational"></a>

#### ✅ 清楚且适合教学

```markdown
## 理解随时间反向传播

想象你在教别人投球。如果没有命中目标，你不会只说“再试一次”，
而会指出“投低一点”或“少用一点力”。

BPTT 对神经网络做的也是类似的事：计算哪些权重需要调整，以及调整多少。
“随时间”表示沿序列向前面的时间步追溯误差。
```

<a id="-bad---too-technical-no-intuition"></a>

#### ❌ 只有技术表述，缺乏直觉

```markdown
## 随时间反向传播

BPTT 通过沿计算图中的时间依赖递归应用链式法则来计算梯度。
```

<a id="-good---math-with-context"></a>

#### ✅ 数学公式配有解释

```markdown
隐藏状态更新公式：
$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

逐项解释：
- $W_{xh} x_t$：当前输入的贡献。
- $W_{hh} h_{t-1}$：前一时刻记忆的贡献。
- $b_h$：偏置项，即可学习的偏移量。
- $\tanh$：将数值压缩到 [-1, 1] 范围内。
```

<a id="-bad---math-without-context"></a>

#### ❌ 只有公式，没有解释

```markdown
$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
```

<a id="use-analogies"></a>

### 运用类比

从读者熟悉的概念出发，例如：

- RNN 隐藏状态：短期记忆。
- 梯度裁剪：给更新速度设限。
- 采样温度：控制多样性的旋钮。
- 注意力：聚焦的探照灯。
- 批归一化：对考试成绩进行标准化。

类比用于建立直觉，不能代替数学定义。

---

<a id="adding-exercises"></a>

## 增加练习

好的练习应当：

1. **帮助建立理解，** 而不仅是练习语法。
2. **难度递进，** 从简单逐步深入。
3. **目标明确，** 说明学生将学到什么。
4. **附带解答，** 并解释完整过程。

<a id="exercise-template"></a>

### 练习模板

```python
"""
练习 X：[清晰、具体的标题]

目标：[学生将学到什么？]

难度：⏱️ 快速 / ⏱️⏱️ 中等 / ⏱️⏱️⏱️ 项目

步骤：
1. [步骤 1]
2. [步骤 2]
...

预期结果：
[正确实现后应出现什么？]

提示：
- [提示 1]
- [提示 2]

常见错误：
- [易错点 1]
- [易错点 2]
"""

# 含待完成部分的起始代码
def exercise_function():
    # TODO：在此实现
    pass

if __name__ == "__main__":
    # 检查代码
    pass
```

---

<a id="testing"></a>

## 验证

提交前确认以下事项。

<a id="1-code-runs-without-errors"></a>

### 1. 代码运行无误

在相应章节目录执行；示例数据路径应换成实际存在的文件：

```bash
python implementation.py
python train_minimal.py --data data/sample.txt --epochs 10
```

<a id="2-outputs-are-reasonable"></a>

### 2. 输出合理

- 训练损失下降。
- 生成样本随训练改善。
- 可视化正确显示。

<a id="3-notebook-cells-execute-in-order"></a>

### 3. Notebook 单元依次执行

```bash
jupyter nbconvert --to notebook --execute notebook.ipynb
```

翻译检查请将执行产物写入独立目录，避免覆盖原文件或混入历史输出。

<a id="4-documentation-is-accurate"></a>

### 4. 文档准确

- 代码与解释一致。
- 示例行为符合描述。
- 链接可用。

---

<a id="recognition"></a>

## 贡献者署名

原项目承诺在 README、发布说明和提交历史中记录贡献者并表示感谢。

<a id="questions"></a>

## 有疑问？

- 💬 [GitHub 讨论区](https://github.com/zhang990527/30u30-zh/discussions)：一般问题。
- 📧 邮箱：`[your-email]`（原文占位符，尚未提供实际地址）。
- 🐦 Twitter：#30u30。

---

<a id="code-of-conduct"></a>

## 行为准则

<a id="be-respectful"></a>

### 尊重他人

- 每个人都在学习。
- 鼓励提问。
- 批评应具有建设性。

<a id="be-patient"></a>

### 保持耐心

- 审查需要时间。
- 维护者是志愿者。
- 并非每条建议都会被采纳。

<a id="be-helpful"></a>

### 乐于帮助

- 力所能及时回答问题。
- 分享学习收获。
- 帮助其他人调试。

---

感谢你让 30u30 变得更好！🎉 无论贡献大小，都能帮助更多人学习。
