# 数据目录

本目录说明指针网络训练所用的数据集与数据生成工具。

---

## 可用数据集

### 1. 排序数据集

**用途：**训练数字排序模型

**生成方式：**
```python
from torch.utils.data import Dataset
import torch

class SortingDataset(Dataset):
    def __init__(self, num_samples=10000, set_size=10):
        self.num_samples = num_samples
        self.set_size = set_size
    
    def __getitem__(self, idx):
        values = torch.rand(self.set_size)
        _, sorted_indices = torch.sort(values)
        return values.unsqueeze(1), sorted_indices
```

**用法：**
```bash
python train.py --task sort --set-size 10 --train-samples 10000
```

**数据集统计：**
- **默认规模：**10,000 个训练样本、1,000 个验证样本
- **元素范围：**[0.0, 1.0]（均匀随机）
- **集合规模：**5-50 个元素（可配置）

---

### 2. 凸包数据集

**用途：**学习几何推理——寻找边界点

**生成方式：**
```python
from scipy.spatial import ConvexHull
import torch

# 生成随机二维点
points = torch.rand(num_points, 2)

# 计算真实凸包
hull = ConvexHull(points.numpy())
hull_indices = hull.vertices  # 边界点索引
```

**用法：**
```bash
python train.py --task convex_hull --set-size 20 --train-samples 5000
```

**数据集统计：**
- **默认规模：**5,000 个训练样本、500 个验证样本
- **点分布：**在 [0, 1]^2 内均匀分布
- **凸包规模：**可变（原仓库称通常占全部点的 30-40%；该比例依赖点数和分布）

---

### 3. TSP 数据集

**用途：**旅行商问题——寻找高质量巡回路径

**生成方式：**
```python
# 生成城市坐标
cities = torch.rand(num_cities, 2)

# 计算贪心最近邻巡回路径（监督信号）
def greedy_tour(cities):
    current = 0
    unvisited = set(range(1, len(cities)))
    tour = [current]
    
    while unvisited:
        nearest = min(unvisited, 
                     key=lambda c: torch.norm(cities[current] - cities[c]))
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return tour
```

**用法：**
```bash
python train.py --task tsp --set-size 15 --train-samples 10000
```

**数据集统计：**
- **默认规模：**10,000 个训练样本、1,000 个验证样本
- **城市分布：**在 [0, 1]^2 内均匀分布
- **巡回路径质量：**贪心启发式（原仓库标为约最优值的 1.25 倍；未附验证依据）

---

## 动态数据生成

所有数据集都在训练过程中**即时生成**：

**优点：**
- 无需存储数据（按需生成）  
- 理论上可持续采样，不受固定样本数限制；原仓库“永不重复”的表述不构成严格保证  
- 易于调整难度（改变集合规模）  
- 无需下载

**工作方式：**
```python
# 在数据加载器中
for epoch in range(num_epochs):
    for batch in dataloader:
        # 每次迭代生成新的随机实例
        inputs, targets = batch
```

---

## 预生成数据集（可选）

为便于复现，可以生成固定数据集：

### 生成并保存

```python
import torch
from train import get_dataset

# 生成数据集
dataset = get_dataset(task='sort', num_samples=10000, set_size=10)

# 保存到磁盘
torch.save({
    'data': [dataset[i] for i in range(len(dataset))],
    'config': {
        'task': 'sort',
        'set_size': 10,
        'num_samples': 10000
    }
}, 'sorting_dataset_10k.pt')
```

### 加载预生成数据集

```python
import torch

# 加载数据集
data = torch.load('sorting_dataset_10k.pt')
samples = data['data']
config = data['config']

print(f"已加载任务 {config['task']} 的 {len(samples)} 个样本")
```

---

## 数据集难度等级

### 简单（适合初始训练）
- **排序：**5-10 个元素
- **凸包：**10-15 个点
- **TSP：**5-10 个城市

### 中等（常规训练）
- **排序：**15-30 个元素
- **凸包：**20-30 个点  
- **TSP：**15-20 个城市

### 困难（泛化测试）
- **排序：**50-100 个元素
- **凸包：**50 个以上的点
- **TSP：**30-50 个城市

---

## 数据集统计

### 排序

| 集合规模 | 收敛时间 | 准确率 |
|----------|------------------|----------|
| 5 | 约 10 轮 | 100% |
| 10 | 约 20 轮 | 100% |
| 20 | 约 50 轮 | 99% |
| 50 | 约 100 轮 | 95% |

### 凸包

| 点数 | 凸包大小（平均） | 收敛时间 | 准确率 |
|------------|-----------------|-------------|----------|
| 10 | 4-5 个点 | 约 50 轮 | 98% |
| 20 | 6-8 个点 | 约 100 轮 | 95% |
| 50 | 10-15 个点 | 约 200 轮 | 90% |

### TSP

| 城市数 | 巡回路径长度（相对贪心算法） | 收敛时间 |
|------------|-------------------------|-------------|
| 10 | 改善 5-10% | 约 100 轮 |
| 20 | 改善 3-8% | 约 200 轮 |
| 50 | 改善 0-5% | 约 500 轮 |

---

## 自定义数据集

想尝试自己的集合到序列问题吗？

### 模板

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """自定义的集合到序列问题。"""
    
    def __init__(self, num_samples, set_size):
        self.num_samples = num_samples
        self.set_size = set_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 第 1 步：生成输入集合（顺序无关）
        inputs = self.generate_input_set(self.set_size)
        
        # 第 2 步：计算真实输出序列（顺序很重要）
        targets = self.compute_target_sequence(inputs)
        
        return inputs, targets
    
    def generate_input_set(self, size):
        # 在此实现输入生成逻辑
        raise NotImplementedError
    
    def compute_target_sequence(self, inputs):
        # 在此实现目标计算逻辑
        raise NotImplementedError
```

### 示例：按绝对值排序

```python
class AbsValueSortDataset(Dataset):
    def __getitem__(self, idx):
        # 生成数字（可以为负）
        values = torch.randn(self.set_size)
        
        # 按绝对值排序
        abs_values = torch.abs(values)
        _, sorted_indices = torch.sort(abs_values)
        
        return values.unsqueeze(1), sorted_indices
```

---

## 数据增强

### 改善泛化能力

```python
# 随机缩放
values = values * random.uniform(0.5, 2.0)

# 随机偏移
values = values + random.uniform(-1.0, 1.0)

# 随机子集（用于可变规模训练）
actual_size = random.randint(5, max_size)
values = values[:actual_size]
```

---

## 快速开始

```bash
# 训练排序任务（最简单）
python train.py --task sort --set-size 10 --epochs 50

# 训练凸包任务（中等）
python train.py --task convex_hull --set-size 20 --epochs 100

# 训练 TSP 任务（最难）
python train.py --task tsp --set-size 15 --epochs 200
```

---

## 资源

- **训练脚本：**`../train.py`
- **数据集生成器：**`../train.py`（SortingDataset、ConvexHullDataset、TSPDataset）
- **论文：**https://arxiv.org/abs/1511.06391

---

**提示：**从小集合（5-10 个元素）开始，逐渐增大规模。

> **译者核验注：**本页统计数字与预期性能保留自原仓库，未附实验日志、随机种子或评估协议，不能视为复现结果。实际 `train.py` 未提供 `get_dataset` 或命令行训练入口；其中数据集定义与本页声称的默认值、即时生成行为应逐项核对。本地化未补写这些接口。
