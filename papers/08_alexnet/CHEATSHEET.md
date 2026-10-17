# AlexNet 速查表

《使用深度卷积神经网络进行 ImageNet 分类》（ImageNet Classification with Deep Convolutional Neural Networks）速查参考

---

## 核心思想（30 秒）

AlexNet 证明了**更大网络 + 更多数据 + GPU = 突破性表现**。它以巨大优势赢得 ImageNet 2012，开启深度学习革命。可以这样理解：
- **AlexNet 之前** = 人工特征 + 浅层学习
- **AlexNet** = 用深度网络直接从原始像素学习特征
- **影响** = 永久改变了计算机视觉

---

## 架构：8 层先驱

```
Input: 224×224×3 RGB image

CONV1: 96 filters, 11×11, stride 4 → 55×55×96
POOL1: 3×3 max pooling, stride 2 → 27×27×96

CONV2: 256 filters, 5×5, pad 2 → 27×27×256  
POOL2: 3×3 max pooling, stride 2 → 13×13×256

CONV3: 384 filters, 3×3, pad 1 → 13×13×384
CONV4: 384 filters, 3×3, pad 1 → 13×13×384
CONV5: 256 filters, 3×3, pad 1 → 13×13×256
POOL5: 3×3 max pooling, stride 2 → 6×6×256

FC6: 4096 units (Dropout 0.5)
FC7: 4096 units (Dropout 0.5)  
FC8: 1000 units (ImageNet classes)
```

**关键见解：** 大卷积核（11×11、5×5）捕捉更多空间上下文！

---

## 快速开始

### 训练
```bash
# Train on ImageNet subset
python train_minimal.py --data imagenet_subset --epochs 90 --batch-size 128

# Custom dataset
python train_minimal.py --data custom_dataset --lr 0.01 --momentum 0.9
```

### 推理
```bash
# Classify single image
python train_minimal.py --predict --image cat.jpg --checkpoint alexnet_model.pth
```

### 在 Python 中使用
```python
from implementation import AlexNet

# Create model
model = AlexNet(num_classes=1000)

# Load pretrained weights
model.load_state_dict(torch.load('alexnet.pth'))

# Classify image
prediction = model(image_tensor)
```

---

## 革命性技术

### 1. ReLU 激活
```python
# Instead of sigmoid/tanh
output = torch.relu(input)  # Much faster training!
```

### 2. 随机失活正则化  
```python
# Prevent overfitting in fully connected layers
x = F.dropout(x, p=0.5, training=self.training)
```

### 3. 数据增强
```python
# Horizontal flips + random crops + color jittering
transforms.RandomHorizontalFlip(),
transforms.RandomResizedCrop(224),
transforms.ColorJitter(0.4, 0.4, 0.4)
```

### 4. GPU 并行
```python
# Split model across 2 GPUs (original approach)
conv1_2 = nn.DataParallel(conv_layers[:2])
conv3_5 = nn.DataParallel(conv_layers[2:])
```

---

## 训练配方

### 超参数
```python
lr = 0.01                    # Learning rate
momentum = 0.9              # SGD momentum  
weight_decay = 0.0005       # L2 regularization
batch_size = 128            # Mini-batch size
epochs = 90                 # Training epochs
lr_decay_factor = 0.1       # Divide LR by 10 when val error plateaus
lr_reductions = 3            # Paper reduced LR 3 times total (0.01 → 1e-5)
```

### 学习率调度
```python
# Paper's approach: reduce LR by 10x when validation error stops improving
# (NOT at fixed epochs — manually monitored during training)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5
)
```

### 数据增强流程
```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

---

## 实现建议

### 内存优化
```python
# Use gradient checkpointing for large images
model = torch.utils.checkpoint.checkpoint_sequential(model, 2, input)

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)
```

### 特征提取
```python
# Use pretrained AlexNet as feature extractor
model = torchvision.models.alexnet(pretrained=True)
model.classifier = model.classifier[:-1]  # Remove last layer

# Extract features
features = model(images)  # 4096-dimensional features
```

### 微调
```python
# Freeze convolutional layers
for param in model.features.parameters():
    param.requires_grad = False

# Only train classifier
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.001)
```

---

## 可视化工具

### 卷积核可视化
```python
from visualization import AlexNetVisualizer

viz = AlexNetVisualizer(model)

# Visualize learned filters
viz.plot_conv_filters(layer='conv1', num_filters=16)

# Feature maps for input image
viz.plot_feature_maps(image, layer='conv2')
```

### 激活模式
```python
# Hook to capture intermediate activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.features[0].register_forward_hook(get_activation('conv1'))
```

---

## 常见问题与解决方法

### 训练问题
```python
# Problem: Exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# Problem: Slow convergence  
# Solution: Proper weight initialization
nn.init.normal_(m.weight, mean=0, std=0.01)

# Problem: Overfitting
# Solution: More dropout, data augmentation
dropout_rate = 0.7  # Increase from 0.5
```

### 内存问题
```python
# Reduce batch size for limited GPU memory
batch_size = 64  # Instead of 128

# Use gradient accumulation
accumulation_steps = 2
loss = loss / accumulation_steps
loss.backward()

if (i + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

---

## 性能基准

### 原始结果
- **ILSVRC-2010**（单个 CNN，10 裁剪）：Top-1 错误率 37.5%，Top-5 错误率 17.0%
- **ILSVRC-2012**（5 个 CNN 的集成）：Top-5 错误率 **15.3%**（领先第 2 名 10.9%）
- **训练用时：** 在 2 块 GTX 580 3GB GPU 上训练 5-6 天

### 现代改进
```python
# Better initialization
nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

# Better optimization
optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)

# Better augmentation
transforms.RandAugment(num_ops=2, magnitude=9)
```

---

## 架构变体

### AlexNet-Mini（用于较小数据集）
```python
class AlexNetMini(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 11, 4, 2),  # Fewer filters
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # ... reduced architecture
        )
```

### AlexNet + 批归一化
```python
# Add batch norm for faster training
nn.Conv2d(3, 96, 11, 4, 2),
nn.BatchNorm2d(96),  # Add after conv
nn.ReLU(inplace=True),
```

---

## 调试命令

### 模型概览
```python
from torchsummary import summary
summary(model, (3, 224, 224))  # Print architecture

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # ~61M for AlexNet
```

### 训练监控
```python
# Check gradient norms
total_norm = 0
for p in model.parameters():
    param_norm = p.grad.data.norm(2)
    total_norm += param_norm.item() ** 2
total_norm = total_norm ** (1. / 2)
print(f"Gradient norm: {total_norm}")
```

### 激活统计
```python
# Monitor activation distributions
def print_activation_stats(module, input, output):
    print(f"{module.__class__.__name__}:")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std:  {output.std().item():.4f}")
    print(f"  Dead: {(output == 0).float().mean().item():.4f}")
```

---

## 历史背景

### AlexNet 为何重要
- **首个深度 CNN：** 证明深度对图像识别很重要
- **GPU 革命：** 展示专用硬件如何支持更大模型  
- **对数据的渴求：** 展示对大型数据集（ImageNet）的需求
- **端到端学习：** 特征通过学习获得，而非人工设计

### 它开启了什么
```
2012: AlexNet → Deep learning renaissance
2014: VGGNet → Deeper networks  
2015: ResNet → Even deeper with skip connections
2016: DenseNet → Maximum information flow
Today: Vision Transformers → Attention-based vision
```

*AlexNet 不仅赢得竞赛，还开启了深度学习革命！*
