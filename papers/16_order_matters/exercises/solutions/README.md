# 第 16 天参考解答：顺序很重要（指针网络）

全部 5 个练习的完整参考解答，附详细说明。

---

## 解答文件

每份解答都提供可直接运行的完整实现：

- `solution_1.py`——基本指针注意力
- `solution_2.py`——集合编码器  
- `solution_3.py`——使用指针网络排序
- `solution_4.py`——凸包
- `solution_5.py`——旅行商问题

---

## 如何使用参考解答

### 何时查看参考解答

**适合查看的情况：**
- 已尝试超过 30 分钟，仍无进展
- 想核对自己的实现
- 想了解其他方法
- 已完成练习，希望比较实现

**请避免：**
- 尚未尝试就查看解答
- 不理解就复制代码
- 跳过学习过程

### 如何从解答中学习

1. **先尝试**——认真花时间完成练习
2. **比较方法**——即使你的实现能运行，也可以研究参考解答
3. **理解差异**——参考解答为何采用另一种做法？
4. **两者都运行**——比较你的版本与参考解答
5. **动手实验**——修改解答，观察出错，再修复它。

---

## 涵盖的关键概念

### 解答 1：指针注意力
- 注意力分数计算
- 屏蔽机制
- 用 softmax 得到概率分布
- 用 argmax 进行选择

### 解答 2：集合编码器
- 不使用位置编码的自注意力
- 置换不变性
- 层归一化
- 残差连接

### 解答 3：排序
- 训练循环实现
- 教师强制
- 负对数似然损失
- 完全匹配准确率

### 解答 4：凸包
- 使用神经网络进行几何推理
- 可变长度输出
- 预测结果可视化
- 几何任务的评估指标

### 解答 5：TSP
- 组合优化
- 束搜索（可选）
- 巡回路径质量指标
- 与经典算法比较

---

## 预期性能

| 任务 | 小集合 | 中等集合 | 大集合 | 说明 |
|------|-----------|------------|-----------|-------|
| **排序** | 100% | 100% | 98% | 最简单的任务 |
| **凸包** | 99% | 95% | 90% | 几何任务 |
| **TSP** | 90% | 80% | 70% | NP 难问题 |

*小集合 = 5-10 个元素，中等集合 = 20-30 个元素，大集合 = 50 个及以上元素。上述数值保留自原仓库，未附可复现实验依据，不作为本地化验收结果。*

---

## 常见问题与处理方法

### 问题 1：损失为 NaN
**现象：**几次迭代后损失变成 NaN

**处理方法：**
```python
# 检查屏蔽逻辑：在 softmax 之前，将被屏蔽的值设为 -inf
scores = scores.masked_fill(mask == 0, float('-inf'))
probs = F.softmax(scores, dim=-1)  # 在仍有有效位置时，可正确处理 -inf

# 同时加入梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
```

### 问题 2：模型学不动
**现象：**准确率停留在随机水平（原仓库给出的 10 个元素约 10% 指单步选择，并非整条序列完全匹配率）

**处理方法：**
- 从较小集合（5 个元素）开始
- 增大模型容量（hidden_dim=128 或 256）
- 训练时使用教师强制
- 检查学习率（尝试 1e-3 或 1e-4）
- 确认损失在下降

### 问题 3：内存不足
**现象：**CUDA 显存不足错误

**处理方法：**
```python
# 减小批大小
batch_size = 32  # 或 16

# 减小模型规模
hidden_dim = 64  # 原先为 128

# 使用梯度累积
for i, batch in enumerate(dataloader):
    loss = compute_loss(...)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 问题 4：训练缓慢
**现象：**每轮训练耗时很长

**处理方法：**
- 有 GPU 时使用 GPU
- 测试时减小数据集规模
- 使用 `num_workers > 0` 的 DataLoader
- 分析代码性能，找出瓶颈

---

## 解答要点

### 简洁的注意力实现（解答 1）

```python
# 简洁的向量化注意力计算
keys_proj = self.W_key(keys)  # [batch, seq_len, hidden]
query_proj = self.W_query(query).unsqueeze(1)  # [batch, 1, hidden]
combined = torch.tanh(keys_proj + query_proj)  # 使用广播
scores = self.v(combined).squeeze(-1)  # [batch, seq_len]
```

### 置换不变性测试（解答 2）

```python
# 检查聚合表示是否不依赖顺序（应在 eval 模式下比较）
points_shuffled = points[:, torch.randperm(set_size)]
enc1 = encoder(points)
enc2 = encoder(points_shuffled)
# enc1.mean(dim=1) 应与 enc2.mean(dim=1) 在数值误差内相等
```

### 教师强制策略（解答 3）

```python
# 训练时使用真实目标
if training and random.random() < teacher_forcing_ratio:
    next_input = inputs[target_pointer]  # 真实目标
else:
    next_input = inputs[predicted_pointer]  # 模型预测
```

---

## 进一步探索

### 可以尝试的扩展

1. **束搜索**——保留得分最高的 k 个候选，而非直接贪心解码
2. **注意力可视化**——绘制注意力权重热力图
3. **更大规模的问题**——测试对 50 个以上元素的泛化能力
4. **其他任务**——尝试其他集合到序列问题
5. **课程学习**——从小规模开始，逐步提高难度

### 研究方向

- **集合 Transformer**——使用完整注意力机制的后续扩展
- **图神经网络**——带有边／关系的集合
- **槽注意力（Slot Attention）**——以对象为中心的表示
- **神经组合优化**——改进 TSP 求解器

---

## 补充阅读

- 原论文：https://arxiv.org/abs/1511.06391
- 指针网络：https://arxiv.org/abs/1506.03134
- 集合 Transformer：https://arxiv.org/abs/1810.00825
- 《注意力即你所需》：https://arxiv.org/abs/1706.03762

---

## 解答检查表

用此表记录已阅读的解答：

- [ ] 解答 1：指针注意力
- [ ] 解答 2：集合编码器
- [ ] 解答 3：排序
- [ ] 解答 4：凸包
- [ ] 解答 5：TSP

---

**参考解答位于上面列出的文件中。**

*请记住：目标是理解，而不只是得到正确答案。*
