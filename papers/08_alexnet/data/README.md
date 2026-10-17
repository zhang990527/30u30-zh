# 数据目录——第 8 天：AlexNet

此目录保存 AlexNet 训练的数据集和模型检查点。

## 内容

### 数据集
将数据集放在这里：

- `imagenet/`——ImageNet 数据集（从 http://www.image-net.org/ 下载）
- `cifar10/`——CIFAR-10 数据集（由 torchvision 自动下载）
- `custom/`——自定义图像分类数据集

### 模型检查点
- `checkpoints/`——已保存的模型权重
- `pretrained/`——预训练模型权重

### 日志
- `tensorboard/`——TensorBoard 日志文件
- `wandb/`——Weights & Biases 日志

## 下载数据集

### CIFAR-10（自动）
```python
from torchvision import datasets

trainset = datasets.CIFAR10(
    root='./data/cifar10',
    train=True,
    download=True
)
```

### ImageNet（手动）
1. 在 http://www.image-net.org/ 注册
2. 下载 ILSVRC2012 数据集
3. 解压到 `data/imagenet/`
4. 目录结构：
   ```
   imagenet/
   ├── train/
   │   ├── n01440764/
   │   ├── n01443537/
   │   └── ...
   └── val/
       ├── n01440764/
       ├── n01443537/
       └── ...
   ```

## 说明

- CIFAR-10：约 170 MB
- ImageNet：约 150 GB
- 模型检查点：每个模型约 200 MB
