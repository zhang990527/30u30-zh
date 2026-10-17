# 第 8 天：AlexNet——练习

深入理解赢得 ImageNet 2012 的革命性深度学习架构！

---

## 练习 1：GPU 影响模拟器 ⭐⭐⭐

理解 AlexNet 的 GPU 加速为何具有革命性。

**题目：** AlexNet 无法放入单块 GPU 的 3GB 显存，因此拆分到 2 块 GPU。分析 GPU 并行如何提高训练速度。

**起始代码：**
```python
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

class AlexNetSimplified(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def benchmark_gpu_vs_cpu():
    """Benchmark training on GPU vs CPU"""
    
    model = AlexNetSimplified()
    criterion = nn.CrossEntropyLoss()
    
    batch_sizes = [32, 64, 128, 256]
    gpu_times = []
    cpu_times = []
    
    # TODO: Time forward/backward passes on GPU and CPU
    # For each batch size, run 10 training iterations
    # Measure elapsed time using time.time()
    
    return batch_sizes, gpu_times, cpu_times

results = benchmark_gpu_vs_cpu()
# Plot comparison
```

**预期行为：**
- 大批次下 GPU 应比 CPU 快 10-50 倍
- 加速比随批大小增加而提高（GPU 利用率更高）
- 批大小为 256 时，GPU 训练应快约 30-50 倍

**提示：**
1. 用 `.to(device)` 将模型和数据移到目标设备
2. 计时前预热 GPU，以获得准确测量
3. 用 `torch.cuda.synchronize()` 确保 GPU 操作完成
4. 测量 10 次迭代的总时间（前向 + 反向传播）
5. 比较 GPU 与 CPU 的吞吐量（图像／秒）

**评分标准：**
- ⭐⭐⭐（7/10）：正确测量 GPU 与 CPU 耗时，展示加速比
- ⭐⭐⭐⭐（9/10）：同时分析内存使用量
- ⭐⭐⭐⭐⭐（10/10）：分析扩展效率及实践意义

---

## 练习 2：ReLU 与 Sigmoid ⭐⭐⭐⭐

AlexNet 的关键创新：使用 ReLU 激活代替 sigmoid／tanh。

**题目：** 比较不同激活函数如何影响训练动态，测量收敛速度、梯度消失和表示能力。

**起始代码：**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

class NetworkWithActivation(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.act1 = activation_fn
        self.fc2 = nn.Linear(512, 256)
        self.act2 = activation_fn
        self.fc3 = nn.Linear(256, 128)
        self.act3 = activation_fn
        self.fc4 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)
        return x

def compare_activations():
    """Compare ReLU, Sigmoid, and Tanh activations"""
    
    activations = {
        'ReLU': nn.ReLU(),
        'Sigmoid': nn.Sigmoid(),
        'Tanh': nn.Tanh()
    }
    
    # Load MNIST
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST('data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    
    results = {}
    
    for act_name, act_fn in activations.items():
        model = NetworkWithActivation(act_fn)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        grad_magnitudes = []
        
        # TODO: Train for 5 epochs
        # - Track training loss per batch
        # - Track average gradient magnitude in first layer
        # - Measure training time
        
        results[act_name] = {
            'losses': losses,
            'gradients': grad_magnitudes,
            'final_loss': losses[-1] if losses else None
        }
    
    return results

results = compare_activations()
# Visualize results
```

**预期发现：**
- ReLU：收敛快、梯度大、训练稳定
- Sigmoid／Tanh：收敛慢、梯度消失、训练停滞
- 在 MNIST 上，ReLU 的收敛速度快 3-5 倍

**提示：**
1. 用 `model.fc1.weight.grad` 访问梯度
2. 用 `grad.norm().item()` 计算梯度大小
3. 在训练初期追踪梯度，此时最有参考价值
4. 用对数刻度绘制损失曲线，观察收敛模式

**评分标准：**
- ⭐⭐⭐⭐（8/10）：用损失曲线正确比较激活函数
- ⭐⭐⭐⭐（9/10）：分析梯度大小，展示梯度消失
- ⭐⭐⭐⭐⭐（10/10）：同时分析 ReLU 死亡问题并提出解决方法

---

## 练习 3：学习率调度 ⭐⭐⭐⭐

AlexNet 采用手动学习率退火训练，这对收敛至关重要。

**题目：** 实现并测试不同学习率调度，找出收敛最快的方案。

**起始代码：**
```python
import torch.optim.lr_scheduler as lr_scheduler

class LRScheduleComparison:
    def __init__(self):
        self.schedules = {
            'constant': lambda opt: None,
            'step': lambda opt: lr_scheduler.StepLR(opt, step_size=10, gamma=0.1),
            'exponential': lambda opt: lr_scheduler.ExponentialLR(opt, gamma=0.95),
            'cosine': lambda opt: lr_scheduler.CosineAnnealingLR(opt, T_max=50),
            'warmup_cosine': lambda opt: self._warmup_cosine(opt)
        }
    
    def _warmup_cosine(self, optimizer):
        # TODO: Implement warmup followed by cosine annealing
        # Warmup: linearly increase LR from 0 to base_lr over 5 epochs
        # Then: cosine annealing for remaining epochs
        pass
    
    def compare_schedules(self):
        # TODO: Train AlexNet with each schedule
        # Track: loss, validation accuracy, convergence speed
        # Measure: total time to reach 95% accuracy
        
        results = {}
        
        return results

comparison = LRScheduleComparison()
results = comparison.compare_schedules()
```

**预期结果：**
- 恒定学习率：训练约在第 20 轮停滞
- 阶梯衰减：有所改善，至第 40 轮收敛
- 指数衰减：平稳改善，至第 35 轮收敛
- 余弦退火：收敛最好，至第 30 轮达到目标
- 预热 + 余弦：初期进展最快，训练稳定

**提示：**
1. 每轮结束后调用 `scheduler.step()`
2. 预热阶段：前 N 轮手动调整优化器学习率
3. 用 `optimizer.param_groups[0]['lr']` 设置自定义学习率
4. 绘制学习率随轮次变化的曲线，可视化所有方案

**评分标准：**
- ⭐⭐⭐⭐（8/10）：实现 3+ 种调度并合理比较
- ⭐⭐⭐⭐（9/10）：包含预热调度及分析
- ⭐⭐⭐⭐⭐（10/10）：发现预热 + 余弦对本任务最优

---

## 练习 4：数据增强的重要性 ⭐⭐⭐⭐

AlexNet 用较强数据增强来正则化训练。

**题目：** 使用不同强度的数据增强训练模型，测量对过拟合和泛化的影响。

**起始代码：**
```python
from torchvision import transforms

# Define augmentation strategies
augmentations = {
    'none': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    
    'basic': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    
    'strong': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # TODO: Add more augmentations similar to AlexNet
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    
    'alexnet_style': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # TODO: Add AlexNet-style augmentations:
        # - Random crops to 224x224
        # - PCA color augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
}

def measure_augmentation_impact():
    # TODO: Train with each augmentation level
    # Track: training loss, validation loss, train/test gap (overfitting)
    # Measure: final test accuracy for each strategy
    
    results = {}
    
    return results

results = measure_augmentation_impact()
```

**预期结果：**
- 无增强：训练损失最好，测试准确率最差（差距大 = 过拟合）
- 基础增强：减少过拟合，泛化更好
- 强增强：与基础增强相似，性能趋于平稳
- AlexNet 风格：测试准确率最好，训练／测试差距最小

**提示：**
1. 使用相同架构和超参数，确保公平比较
2. 绘制训练损失与验证损失，观察过拟合
3. 计算“泛化差距” = test_loss - train_loss
4. AlexNet 在测试时使用多裁剪评估，以提高准确率

**评分标准：**
- ⭐⭐⭐⭐（8/10）：使用合适指标测试 3+ 种增强策略
- ⭐⭐⭐⭐（9/10）：实现 AlexNet 风格增强（裁剪、颜色扰动）
- ⭐⭐⭐⭐⭐（10/10）：展示数据增强对现代深度学习的重要性

---

## 练习 5：回看批归一化 ⭐⭐⭐⭐⭐

AlexNet 早于批归一化（2015 年提出），比较如今可以做出的改进。

**题目：** 用批归一化重新实现 AlexNet，测量训练稳定性和收敛速度的改善。

**起始代码：**
```python
class AlexNetWithoutBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # ... rest of network
        )

class AlexNetWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),  # ADD BN
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),  # ADD BN
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # ... rest of network
        )

def compare_with_without_bn():
    # TODO: Train both models on CIFAR-10
    # Track: training loss, validation loss, training stability
    # Measure: time to convergence, final accuracy
    
    models = {
        'AlexNet (no BN)': AlexNetWithoutBN(),
        'AlexNet (with BN)': AlexNetWithBN()
    }
    
    results = {}
    
    return results

results = compare_with_without_bn()
# BN should show ~30-50% faster training!
```

**预期结果：**
- 不使用 BN：训练不稳定，需要仔细调整学习率
- 使用 BN：训练稳定，收敛快 2-5 倍，最终准确率更高
- BN 减少内部协变量偏移，允许使用更高学习率

**提示：**
1. 批归一化通常放在激活之前
2. 两者使用相同超参数（学习率、批大小）
3. 追踪梯度范数，观察内部协变量偏移
4. BN 的 momentum 参数（默认 0.1）影响运行统计量

**评分标准：**
- ⭐⭐⭐⭐（9/10）：正确实现两版本，展示 BN 改善收敛
- ⭐⭐⭐⭐⭐（10/10）：同时分析梯度流，解释 BN 为何有效

---

## 附加挑战：多 GPU 训练 🚀

实现多 GPU 数据并行，或用 CPU 模拟。

```python
# TODO: Use nn.DataParallel to split batch across GPUs
# - Train AlexNet on CIFAR-10 with data parallelism
# - Measure speedup with 1 GPU vs 2 GPUs vs 4 GPUs (if available)
# - Show that speedup saturates due to communication overhead

# This mirrors the original AlexNet paper's approach!
```

**关键见解：** 原始 AlexNet 拆分到 2 块 GPU，因为单块 GPU 的 3GB 显存无法容纳整个网络。现代 8GB+ 显存的 GPU 不再需要这样做，但分布式训练对大规模模型仍然重要！

---

## 总结

**掌握的概念：**
- ✅ GPU 加速对深度学习的影响
- ✅ ReLU 相对于 sigmoid／tanh 的优势
- ✅ 用学习率调度促进收敛
- ✅ 用数据增强实现正则化
- ✅ 批归一化带来的改善
- ✅ 多 GPU 并行

**恍然大悟的时刻：**
1. GPU 加速使深度学习走向实用
2. ReLU 解决梯度消失问题
3. 学习率调度比恒定学习率快 2-3 倍
4. 数据增强是正则化的具体实践
5. 批归一化为稳定训练提供开箱即用的支持
6. 今天训练 AlexNet 的方式会完全不同！

这些实践见解可以直接迁移到现代架构！🔥
