<a id="day-1-cheat-sheet-character-level-rnn"></a>

# 第 1 天速查表：字符级 RNN

训练和使用 RNN 时的快速参考。

---

<a id="the-big-idea-30-seconds"></a>

## 核心思想（30 秒）

仅通过字符级预测，一个带隐藏状态的循环网络就可能学到复杂结构，例如莎士比亚风格文本、LaTeX 与 C 代码。Karpathy 强调，RNN 在理论上具有图灵完备性，在实践中也能在没有显式规则的情况下发现引号配对、括号和缩进等模式。

> 核对注：理论上的图灵完备性需要相应模型和精度假设，不能推导出本章有限精度、有限容量的基本 RNN 能可靠学习任意程序。原博文展示实验采用 LSTM；本章单层基本 RNN 是教学实现。

---

<a id="quick-start"></a>

## 快速开始

```bash
# 在示例数据上训练
python train_minimal.py --data data/tiny_shakespeare.txt --epochs 200

# 自定义设置
python train_minimal.py --data your_data.txt --epochs 500 --hidden-size 128 --seq-length 50 --learning-rate 0.01

# 从已训练的模型生成文本
python train_minimal.py --generate --checkpoint model.pkl --gen-length 500 --temperature 0.8
```

> 文档修正：原速查表使用 `--lr` 和 `--length`，实际解析器只提供 `--learning-rate` 和 `--gen-length`，上面已按现有接口更正。`--epochs` 在当前脚本中实际表示更新次数。

---

<a id="key-hyperparameters"></a>

## 关键超参数

| 参数 | 常见范围 | 作用 | 建议 |
|---|---|---|---|
| `hidden_size` | 50–200 | RNN 的记忆容量 | 越大容量越高，但速度越慢 |
| `seq_length` | 10–100 | 每段处理的字符数 | 越长则展开上下文越长，但速度越慢 |
| `learning_rate` | 0.01–0.5 | 参数更新步长 | 从 0.1 开始，不稳定时降低 |
| `temperature` | 0.2–2.0 | 采样随机性 | 0.8 较均衡，2.0 更多样 |

---

<a id="common-issues--fixes"></a>

## 常见问题与处理

<a id="loss-explodes-nan"></a>

### 损失爆炸或出现 NaN

```python
# 梯度裁剪，参考实现中已包含
np.clip(grad, -5, 5, out=grad)
# 也可降低学习率
```

<a id="loss-plateaus-too-high"></a>

### 损失停留在过高水平

```bash
# 增大 hidden_size 或延长训练
python train_minimal.py --hidden-size 200 --epochs 1000
```

<a id="generated-text-is-gibberish"></a>

### 生成文本杂乱无章

```bash
# 增加训练次数，或降低采样温度
python train_minimal.py --generate --temperature 0.5
```

<a id="training-is-too-slow"></a>

### 训练太慢

```bash
# 减小 hidden_size 或 seq_length
python train_minimal.py --hidden-size 50 --seq-length 25
```

---

<a id="the-math-copy-paste-ready"></a>

## 数学与代码速查

<a id="forward-pass"></a>

### 前向传播

```python
# 隐藏状态，h_prev 表示前一时间步的状态
h_t = np.tanh(Wxh @ x_t + Whh @ h_prev + bh)

# 输出分数
y_t = Why @ h_t + by

# 概率
p_t = softmax(y_t)
```

> 文档修正：原代码块把数学记号 `h_{t-1}` 写进 Python 表达式，不能执行；这里用 `h_prev` 表示同一量。片段仍需所在上下文提供矩阵和 softmax 函数。

<a id="backward-pass-bptt"></a>

### 反向传播（BPTT）

```python
# 输出梯度
dy = p - y_true

# 隐藏状态梯度
dh = Why.T @ dy + dh_next
dh_raw = (1 - h**2) * dh  # 经过 tanh 的导数

# 裁剪梯度
np.clip(grad, -5, 5, out=grad)
```

---

<a id="visualization-examples"></a>

## 可视化示例

```python
from visualization import *

# 训练损失曲线
plot_loss_curve(losses, save_path='loss.png')

# 隐藏状态可视化
plot_hidden_states(hidden_states, "hello world", save_path='states.png')

# 比较采样温度
samples = {0.5: "...", 1.0: "...", 1.5: "..."}
plot_temperature_comparison(samples, save_path='temps.png')
```

---

<a id="experiment-ideas"></a>

## 实验建议

<a id="easy"></a>

### 入门

- 使用不同文本训练，例如歌曲、代码或推文。
- 尝试 0.1 到 3.0 的不同温度。
- 改变 hidden_size，例如 25、50、100、200。

<a id="medium"></a>

### 中等

- 实现学习率衰减。
- 监测梯度范数。
- 跟踪字符级准确率。

<a id="advanced"></a>

### 进阶

- 构建多层 RNN，将多个循环层堆叠。
- 研究双向 RNN，同时处理正向与反向信息。
- 用束搜索替代简单解码方式。

> 核对注：原文把当前采样称为“贪心采样”，但参考实现实际从 softmax 分布随机采样。双向模型可以用于序列表示，不能直接读取尚未生成的未来字符来做自回归生成。

---

<a id="file-reference"></a>

## 文件参考

| 文件 | 用途 |
|---|---|
| `README.md` | 完整教程 |
| `paper_notes.md` | 快速复习 |
| `implementation.py` | 参考代码 |
| `train_minimal.py` | 训练 |
| `visualization.py` | 绘图 |
| `notebook.ipynb` | 交互学习 |
| `exercises/` | 练习 |

---

<a id="debugging-checklist"></a>

## 调试清单

- [ ] 数据文件存在并可读取？
- [ ] 词表大小 < 100？越大通常越难训练。
- [ ] 数据量 > 100KB？较少数据可能学不到丰富模式。
- [ ] 初始损失是否接近 `-log(1/vocab_size) * seq_length`？
- [ ] 前 100 次迭代损失是否下降？
- [ ] 是否裁剪梯度，损失中没有 NaN？
- [ ] 训练超过 500 次迭代后，生成样本是否改善？

> 核对注：上述初始损失公式对应序列总交叉熵；`CharRNN.loss` 返回逐字符平均交叉熵，均匀预测时应接近 `-log(1/vocab_size)`。训练脚本的平滑损失初始化与实际损失尺度不一致，是原实现问题，不在翻译中修改算法。词表和数据量阈值是经验建议，不是正确性判据。

---

<a id="pro-tips"></a>

## 实践建议

1. **从小样例开始：** 先用微型数据验证。
2. **监控损失：** 观察是否总体平稳下降。
3. **定期采样：** 例如每 100 次迭代生成文本。
4. **保存检查点：** 避免丢失训练进度。
5. **尝试不同数据：** 代码、歌词和对话各有不同结构。
6. **可视化：** 绘制损失、隐藏状态与梯度。
7. **关注温度：** 0.8 通常可作为起点。
8. **保持耐心：** 较好的效果可能需要超过 1000 次迭代。

---

<a id="success-criteria"></a>

## 效果参考

原教程建议观察：

- 损失 < 1.5，具体取决于数据和损失定义。
- 生成文本中出现真实单词。
- 具备基本语法或结构。
- 风格接近训练数据。
- 不陷入重复循环。

这些是定性教学目标，不是模型正确性或学习完成的统一验收阈值。

---

<a id="next-day-2"></a>

## 下一步：第 2 天

理解 RNN 后，继续学习 **LSTM**：

- 缓解梯度消失。
- 改善长程记忆。
- 了解 Transformer 流行之前广泛采用的序列模型。

---

**有疑问？** 查看 [README.md](README.md)，或到 [GitHub 讨论区](https://github.com/zhang990527/30u30-zh/discussions)提问。

**分享学习成果：** #30u30
