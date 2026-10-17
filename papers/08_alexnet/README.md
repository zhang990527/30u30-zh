# 第 8 天：AlexNet——深度学习革命

> *《使用深度卷积神经网络进行 ImageNet 分类》（ImageNet Classification with Deep Convolutional Neural Networks）*——Alex Krizhevsky、Ilya Sutskever、Geoffrey E. Hinton（2012）

**原论文：** https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html

**完成用时：** 4-5 小时

**学习内容：**
- AlexNet 如何展示深度学习的能力
- ReLU 激活为何能提高训练效率
- 随机失活（dropout）的正则化效果
- GPU 并行技术
- 如今仍然有效的数据增强策略

---

## 🧠 核心思想

**一句话概括：** AlexNet 表明，在 GPU 上训练并使用 ReLU 激活和随机失活的深度卷积神经网络，能够显著超越传统计算机视觉方法，由此开启了深度学习时代。

### ImageNet 时刻

2012 年 9 月，ImageNet 大规模视觉识别挑战赛（ILSVRC）公布结果：

**传统计算机视觉方法（第 2 名）：** 错误率 26.2%  
**AlexNet（第 1 名）：** 错误率 15.3%

**相差 10.9 个百分点。** 计算机视觉竞赛的进步通常只有零点几个百分点。这种变化堪称一场**革命**。

### 范式转变

AlexNet 出现之前：
- ❌ 计算机视觉依赖人工设计特征（SIFT、HOG、SURF）
- ❌ 深度网络被认为“太难训练”
- ❌ 常见看法是“深度学习不适用于视觉”
- ❌ 最好的系统采用浅层处理流程和人工特征工程

AlexNet 出现之后：
- ✅ 端到端学习取代特征工程
- ✅ “越深越好”成为流行观点
- ✅ GPU 成为 AI 研究的重要工具
- ✅ 深度学习在各项视觉基准上占据主导地位

这篇论文不仅赢得竞赛，还**开启了一场革命**，推动了自动驾驶汽车、医学诊断 AI、人脸识别，以及我们今天熟悉的现代计算机视觉。

---

## 🤔 AlexNet 为何改变了一切

赢得 ImageNet 的意义还在于**验证一种范式**：

**核心见解：** 给定足够的数据和计算资源，神经网络能够学到比人工设计更好的特征。

### 历史背景

**2012 年以前的计算机视觉流程：**
```
Raw Image → Hand-Designed Features → Classifier → Prediction
            (SIFT, HOG, etc.)       (SVM, etc.)
```

每一步都需要：
- 专业领域知识
- 人工调参
- 对每种特征进行多年的研究
- 为每项任务设计独立流程

**AlexNet 的流程：**
```
Raw Image → Deep CNN → Prediction
            (learns features automatically!)
```

一个架构，端到端学习，不针对特定任务手工设计。

### 它为何能在其他方法失败之处成功

AlexNet 将五项关键创新结合起来，取得了其他方法未能实现的成功：

1. **ReLU 激活**——解决梯度消失问题
2. **随机失活**——防止过拟合  
3. **GPU 训练**——使深度网络成为可行方案
4. **数据增强**——人为扩充训练数据
5. **局部响应归一化（LRN）**——改善泛化

这些技术都并非完全全新，但**大规模地将它们结合起来**具有革命性。

---

## 🌍 现实类比

### 流水线类比

可以将传统计算机视觉想象为一家**定制作坊**：

**AlexNet 之前（工匠）：**
- 专家：“我要手工设计用于角点的 SIFT 特征。”
- 专家：“我要为边缘设计 HOG 特征。”
- 专家：“我要为纹理建立颜色直方图。”
- 专家：“现在仔细调参，将它们组合起来……”
- **结果：** 精巧，但速度慢，而且依赖专业知识

**AlexNet 之后（流水线）：**
- 工程师：“这里有 1.2 百万张标注图像。”
- AlexNet：“我自己来学习特征。”
- GPU：*发出大规模计算的嗡嗡声*
- **结果：** 效果更好，速度更快，可扩展到各种任务

### 语言学习类比

**传统视觉（语法规则）：**
- 学习规则：“边缘连接形成形状。”
- 学习规则：“形状组合形成物体。”
- 尝试用规则识别新物体
- **问题：** 不可能穷尽全部规则！

**AlexNet（沉浸学习）：**
- 观察数百万个示例
- 大脑自然地发现模式
- 泛化到新情境
- **优势：** 学到人类难以明确表述的知识！

正如孩子通过接触语言而非语法规则学习语言，AlexNet 通过示例学习视觉，而非依赖人工编写的特征。

---

## 网络架构

### 8 层结构

```
Input: 224×224×3 RGB Image
         ↓
Layer 1: Conv(11×11×96, stride=4) → ReLU → MaxPool(3×3, stride=2) → LRN
         [55×55×96 → 27×27×96]
         ↓
Layer 2: Conv(5×5×256) → ReLU → MaxPool(3×3, stride=2) → LRN
         [27×27×256 → 13×13×256]
         ↓
Layer 3: Conv(3×3×384) → ReLU
         [13×13×384]
         ↓
Layer 4: Conv(3×3×384) → ReLU
         [13×13×384]
         ↓
Layer 5: Conv(3×3×256) → ReLU → MaxPool(3×3, stride=2)
         [13×13×256 → 6×6×256]
         ↓
Layer 6: FC(4096) → ReLU → Dropout(0.5)
         [9216 → 4096]
         ↓
Layer 7: FC(4096) → ReLU → Dropout(0.5)
         [4096 → 4096]
         ↓
Layer 8: FC(1000) → Softmax
         [4096 → 1000]
```

**总参数量：** 约 60 百万
**所需内存：** 约 240 MB（使用 float32）

### 关键创新详解

#### 1. ReLU 激活函数

**改变局面的技术：**
$$f(x) = \max(0, x)$$

**它为何改善深度学习：**

ReLU 之前（sigmoid／tanh）：
- 输入过大或过小时，梯度消失
- 在 0 和 1（或 -1 和 1）处饱和
- 计算较慢，需要指数运算

使用 ReLU 后：
- 对正值**不出现梯度消失**
- **稀疏激活**（许多神经元的输出 = 0）
- **计算高效**（简单的 max 操作）
- **具有生物学上的合理性**（神经元发放或不发放）

**对训练速度的影响：**
AlexNet 论文展示，ReLU 网络的训练速度是 tanh 网络的 **6 倍**！

```python
# Old way (slow, vanishing gradients)
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# AlexNet way (fast, healthy gradients)
def relu(x):
    return np.maximum(0, x)
```

#### 2. 随机失活正则化

**抑制过拟合：**

训练时：
$$y = x \odot \text{Bernoulli}(p)$$

推理时：
$$y = p \cdot x$$

**为何有效：**

传统正则化：
- L2 惩罚：$\lambda \sum w^2$
- 提前停止
- 在大型网络上的效果有限

随机失活：
- 训练包含 $2^n$ 个子网络的**集成**
- 防止**共适应**（神经元不能依赖特定的其他神经元）
- 迫使网络学习**稳健特征**（缺少某些神经元时仍须有效）
- 测试时隐式进行**模型平均**

**关键见解：** 每次前向传播训练一个不同的随机子网络！

```python
class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training:
            # Random mask: keep neuron with probability p
            mask = torch.bernoulli(torch.ones_like(x) * self.p)
            return x * mask / self.p  # Scale to maintain expected value
        else:
            return x  # Use all neurons at test time
```

#### 3. 局部响应归一化（Local Response Normalization，LRN）

受神经科学**侧抑制**启发：

$$b_{x,y}^i = \frac{a_{x,y}^i}{\left(k + \alpha \sum_{j=\max(0,i-n/2)}^{\min(N-1,i+n/2)} (a_{x,y}^j)^2\right)^\beta}$$

其中：
- $a_{x,y}^i$ = 特征图 $i$ 中位置 $(x,y)$ 的神经元活性
- $n$ = 邻域大小（AlexNet 使用 $n=5$）
- $k=2, \alpha=10^{-4}, \beta=0.75$（超参数）

**生物学启发：** 活跃的神经元抑制附近神经元，形成竞争。

**注：** 现代网络通常改用批归一化（Batch Normalization），但 LRN 具有历史意义。

#### 4. 重叠池化

**小而有效的调整：**

传统池化：步幅 = 池化窗口大小（无重叠）
- 2×2 池化、步幅 2：无重叠

AlexNet 池化：步幅 < 池化窗口大小（有重叠！）
- 3×3 池化、步幅 2：区域有重叠

**优点：**
- 准确率略有提高（约 0.4%）
- 特征更稳健
- 更难过拟合

---

## 实现指南

### 从零构建 AlexNet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Layer 1: Conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            
            # Layer 2: Conv2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            
            # Layer 3: Conv3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 4: Conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 5: Conv5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten for FC layers
        x = self.classifier(x)
        return x

# Create model
model = AlexNet(num_classes=1000)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 数据增强（AlexNet 成功的关键）

AlexNet 使用大量数据增强，有效扩充了数据集规模：

```python
import torchvision.transforms as transforms
from PIL import Image

# AlexNet's augmentation strategy
train_transform = transforms.Compose([
    # 1. Random crops
    transforms.RandomResizedCrop(224),
    
    # 2. Random horizontal flips
    transforms.RandomHorizontalFlip(p=0.5),
    
    # 3. Color jittering (AlexNet's PCA color augmentation)
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    ),
    
    # 4. Normalize (ImageNet statistics)
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Test-time augmentation (deterministic)
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

**AlexNet 的 PCA 颜色增强：**

```python
def pca_color_augmentation(image, alpha_std=0.1):
    """
    AlexNet's fancy color augmentation using PCA.
    Adds multiples of principal components of RGB pixel values.
    """
    # Reshape image to (3, H*W)
    orig_shape = image.shape
    image = image.reshape(3, -1)
    
    # Compute covariance matrix
    image_cov = np.cov(image)
    
    # Eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eigh(image_cov)
    
    # Generate random scaling factors
    alpha = np.random.normal(0, alpha_std, 3)
    
    # Add weighted principal components
    rgb_shift = (eig_vecs @ (alpha * eig_vals))
    
    # Apply to image
    augmented = image + rgb_shift.reshape(3, 1)
    
    return augmented.reshape(orig_shape)
```

---

## 🎯 训练建议

### 1. **学习率调度**

AlexNet 使用**手动学习率衰减**：

```python
# Initial learning rate
lr = 0.01

# Paper's approach: reduce LR by 10x when validation error plateaus
# LR was reduced 3 times during ~90 epochs of training (0.01 → 0.001 → 0.0001 → 0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5
)

# Training loop
for epoch in range(90):
    train_one_epoch(model, train_loader, optimizer)
    val_error = validate(model, val_loader)
    scheduler.step(val_error)  # Reduce LR when val error plateaus
```

**现代替代方案（余弦退火）：**

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=0
)
```

### 2. **权重初始化**

AlexNet 使用特定初始化方式：

```python
def init_weights(model):
    """AlexNet's initialization strategy (Section 5).
    
    Paper uses bias=1 for conv2, conv4, conv5 and all FC layers
    to ensure ReLUs start in the active region. Bias=0 for conv1, conv3.
    """
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                # Bias=1 for conv layers 2, 4, 5; bias=0 for conv layers 1, 3
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 1)

    # Paper-specific per-layer bias override (conv2, conv4, conv5 → bias=1)
    # In our nn.Sequential: features[3]=conv2, features[6]=conv4, features[8]=conv5
    for idx in [3, 6, 8]:  # conv2, conv4, conv5 indices
        if hasattr(model, 'features') and isinstance(model.features[idx], nn.Conv2d):
            nn.init.constant_(model.features[idx].bias, 1)
```

**为何某些层使用 bias=1？** 论文将 conv2、conv4、conv5 以及全部全连接层的偏置初始化为 1，使 ReLU 神经元从“激活”区域开始，以加速初期训练。

### 3. **批大小与动量**

```python
# AlexNet's training setup
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,      # Heavy momentum
    weight_decay=5e-4  # L2 regularization
)

# Batch size: 128 (across 2 GPUs = 64 per GPU)
batch_size = 128
```

### 4. **GPU 训练（必需）**

AlexNet 开创性地在 2 块 GPU 之间采用**模型并行**：

```python
class AlexNetParallel(nn.Module):
    """AlexNet with model parallelism like the original"""
    def __init__(self):
        super().__init__()
        
        # First GPU: layers 1-2
        self.features_gpu1 = nn.Sequential(
            nn.Conv2d(3, 48, 11, 4, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.LocalResponseNorm(5),
            
            nn.Conv2d(48, 128, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.LocalResponseNorm(5),
        ).cuda(0)
        
        # Second GPU: layers 3-5
        self.features_gpu2 = nn.Sequential(
            nn.Conv2d(128, 192, 3, 1, 1),
            nn.ReLU(),
            # ... rest of layers
        ).cuda(1)
    
    def forward(self, x):
        x = x.cuda(0)
        x = self.features_gpu1(x)
        x = x.cuda(1)  # Transfer to GPU 2
        x = self.features_gpu2(x)
        return x
```

**现代做法：** 改用 DataParallel 或 DistributedDataParallel。

---

## 📈 可视化

### 1. 特征图可视化

```python
def visualize_feature_maps(model, image, layer_idx=0):
    """Visualize what AlexNet sees at different layers"""
    import matplotlib.pyplot as plt
    
    # Hook to capture activations
    activations = []
    def hook_fn(module, input, output):
        activations.append(output)
    
    # Register hook
    handle = list(model.features.children())[layer_idx].register_forward_hook(hook_fn)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(image.unsqueeze(0))
    
    # Remove hook
    handle.remove()
    
    # Visualize first 64 feature maps
    features = activations[0][0]  # Shape: (C, H, W)
    
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    for idx, ax in enumerate(axes.flat):
        if idx < features.shape[0]:
            ax.imshow(features[idx].cpu(), cmap='viridis')
        ax.axis('off')
    
    plt.suptitle(f'Feature Maps at Layer {layer_idx}', fontsize=16)
    plt.tight_layout()
    plt.show()
```

### 2. 卷积核可视化

```python
def visualize_conv_filters(model, layer_idx=0):
    """Visualize learned convolutional filters"""
    # Get first conv layer weights
    conv_layer = list(model.features.children())[layer_idx]
    filters = conv_layer.weight.data.cpu()
    
    # Normalize for visualization
    filters = (filters - filters.min()) / (filters.max() - filters.min())
    
    # Plot first 96 filters
    fig, axes = plt.subplots(8, 12, figsize=(20, 14))
    for idx, ax in enumerate(axes.flat):
        if idx < filters.shape[0]:
            # Convert to RGB (take first 3 channels)
            filter_rgb = filters[idx, :3].permute(1, 2, 0)
            ax.imshow(filter_rgb)
        ax.axis('off')
    
    plt.suptitle('Learned Convolutional Filters (Layer 1)', fontsize=16)
    plt.tight_layout()
    plt.show()
```

### 3. 训练曲线

```python
def plot_training_curves(train_losses, val_losses, val_accs):
    """Plot AlexNet training dynamics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('AlexNet Training Loss', fontsize=14, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs, val_accs, 'g-', linewidth=2, marker='o')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Mark the 84.7% top-5 accuracy milestone
    ax2.axhline(y=84.7, color='r', linestyle='--', 
                label='AlexNet Top-5 Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

---

## 练习

### 练习 1：从零构建 AlexNet（中等）
不参考现成代码，用 PyTorch 实现 AlexNet。在较小的 CIFAR-10 数据集上训练，并与浅层基线比较。

**学习重点：** 理解各层的作用，以及网络深度如何带来帮助。

### 练习 2：消融研究（困难）
每次移除一项创新，测量其影响：
- 将 ReLU 替换为 sigmoid
- 移除随机失活
- 移除数据增强
- 使用更小的网络

哪项创新最重要？

### 练习 3：特征可视化（简单）
提取并可视化各层特征，观察：
- 第 1 层：边缘检测器
- 第 2 层：纹理模式
- 第 3-5 层：物体部件
- 全连接层：高级概念

### 练习 4：迁移学习（中等）
用预训练 AlexNet 处理新任务：
1. 加载 ImageNet 权重
2. 冻结前面的层
3. 在新数据集上微调，如花卉或汽车

与从零训练比较。

### 练习 5：现代改进（困难）
用现代技术升级 AlexNet：
- 用批归一化替换 LRN
- 添加残差连接
- 使用更好的优化器（Adam、AdamW）
- 实现学习率预热

准确率能提高多少？

---

## 进一步探索

### AlexNet 的影响

**直接影响：**
- VGGNet（2014）：使用更小卷积核构建更深网络
- GoogLeNet（2014）：Inception 模块
- ResNet（2015）：跳跃连接使 100+ 层网络成为可能

**长期影响：**
- 确立 GPU 训练的标准地位
- 验证端到端学习的有效性
- 开启“ImageNet 预训练”时代
- 使深度学习成为主流

### 现代背景

**AlexNet 做对的事情（今天仍在使用）：**
- ReLU 激活
- 随机失活正则化
- 数据增强
- GPU 加速
- 端到端学习

**后来的改进：**
- 批归一化 > LRN
- 残差连接支持更深的网络
- 更好的优化器（Adam 系列）
- 更丰富的数据增强（MixUp、CutMix）
- 注意力机制

### 研究前沿

1. **视觉 Transformer（ViT）**
   - 用自注意力替代卷积
   - 但仍使用 AlexNet 的许多原则！

2. **高效架构**
   - MobileNet、EfficientNet 系列
   - 受 AlexNet 启发，但面向移动设备优化

3. **自监督学习**
   - 不依赖标签学习
   - 仍使用卷积网络骨干

---

## 资源

### 必读资料
- [原论文](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)——AlexNet（2012）
- [ImageNet](http://www.image-net.org/)——改变一切的数据集
- [随机失活论文](https://jmlr.org/papers/v15/srivastava14a.html)——Hinton 等（2014）

### 实现
- [PyTorch AlexNet](https://pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html)——官方实现
- [TensorFlow AlexNet](https://github.com/tensorflow/models/tree/master/research/slim/nets)——TF-Slim 版本
- [原始代码](https://code.google.com/archive/p/cuda-convnet/)——Krizhevsky 的 CUDA 实现

### 可视化
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)——交互式 CNN 可视化
- [Distill：特征可视化](https://distill.pub/2017/feature-visualization/)——理解 CNN 学到了什么
- [Andrej Karpathy 的讲座](https://www.youtube.com/watch?v=NfnWJUyUJYU)——用于视觉识别的 CNN

### 历史背景
- [ImageNet 时刻](https://qz.com/1034972/the-data-that-changed-the-direction-of-ai-research-and-possibly-the-world/)——深度学习占据主导地位之时
- [LeNet-5](http://yann.lecun.com/exdb/lenet/)——前身（Yann LeCun，1998）

---

## 要点

1. **端到端学习胜过人工特征**——让网络自己学习！
2. **ReLU 解决梯度消失问题**——简单而有革命性
3. **随机失活防止过拟合**——类似训练集成模型
4. **数据增强至关重要**——有效扩充数据集规模
5. **GPU 使深度学习成为可能**——硬件与算法同样重要
6. **深度产生层次结构**——前层学习边缘，后层学习概念

### 历史意义

AlexNet 不仅赢得 ImageNet，还验证了一种范式：

**之前：**“神经网络不能处理真实视觉任务。”
**之后：**“神经网络是视觉任务的首选方法。”

仅这一篇论文就：
- 开启深度学习革命
- 使计算机视觉成为 AI 问题，而不只是工程问题
- 确立 ImageNet 挑战赛作为核心基准的地位
- 证明规模（数据 + 计算）很重要

### 它为何仍然重要

虽然今天有了更好的架构（ResNet、视觉 Transformer），AlexNet 确立的原则仍然是基础：

- 深层次的分层特征学习
- 可端到端训练的系统
- GPU 加速
- 通过随机失活和数据增强实现正则化

每个现代视觉系统都受益于 AlexNet。

---

**完成第 8 天了吗？** 接下来学习 **[第 9 天：ResNet](../09_resnet/)**，跳跃连接将使网络深度达到 AlexNet 的 10 倍！

**有疑问？** 查看 [notebook.ipynb](notebook.ipynb)，获取交互式代码与可视化。

---

*“2012 年是重大突破之年。那时，我们让一个大型深度学习系统取得了出色效果；它表现得如此之好，让所有人——包括我们自己——都感到意外。”*——Geoffrey Hinton 回顾 AlexNet
