# 第 16 天速查表：面向集合的序列到序列学习

指针网络训练与部署速查。

---

## 快速开始

```bash
# 训练排序任务（默认）
python train.py --task sort --set-size 10

# 训练凸包任务
python train.py --task hull --set-size 20

# 可视化结果
# python visualization.py --checkpoint checkpoints/model.pt
```

---

## 关键超参数（原仓库标注为 Vinyals 等，2015）

| 参数 | 典型值 | 作用 | 原仓库所列论文依据（待一手核验） |
|-----------|---------------|--------------|-------------------------|
| `embedding_dim`| 128 | 集合元素的向量维度 | 足以表示坐标／标量 |
| `hidden_dim` | 256 | LSTM 状态维度 | 捕捉集合的全局上下文 |
| `processing_steps` | 3-5 | “思考”步数 | 第 3 节：改善组合搜索 |
| `learning_rate`| 1e-3 | Adam/RMSProp 学习率 | 第 4 节：指针实验的常见设置 |
| `grad_clip` | 2.0 | 梯度阈值 | 对序列训练稳定性很重要 |

---

## 常见问题与处理方法

### 模型重复选择同一元素
- **原因：**注意力机制没有学会忽略已经选择的索引。
- **处理：**实现**硬屏蔽（hard masking）**。在 `implementation.py` 的 softmax 步骤之前，将已选索引的注意力分数设为 `-inf`。

### 集合变大后性能下降
- **原因：**模型对训练中固定的序列长度过拟合。
- **处理：**采用**可变长度训练**。在一系列集合规模上训练（例如每批采样 $N \in [5, 15]$），促使模型学习选择规则，而不是固定模式。

### 损失变成 NaN
- **原因：**注意力分数计算中除以零，或 LSTM 发生梯度爆炸。
- **处理：**用 `log_softmax` 计算负对数似然（NLL）损失，并确保启用了 `torch.nn.utils.clip_grad_norm_`。

---

## 核心公式

1. **指针注意力分数：**$u_i = v^T \tanh(W_1 e_i + W_2 d)$
2. **概率分布：**$P(p | e_1, \dots, e_n, d) = \text{Softmax}(u)$
3. **读取操作（集合到集合）：**$h = \text{SelfAttention}(X)$（原仓库将其等同于“读取”阶段；参见 README 的译者核验注）
