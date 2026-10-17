"""
练习 3：训练指针网络进行排序
==================================================

建立训练循环，让指针网络学会执行
数字升序排序。

任务：将随机列表（例如 [5.2, 1.8, 9.3]）映射到指针索引
      （例如 [1, 0, 2]）。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('..')
from implementation import PointerNetwork


class SortingDataset(Dataset):
    """数字排序任务的数据集。"""
    
    def __init__(self, num_samples, set_size, min_val=0.0, max_val=10.0):
        self.num_samples = num_samples
        self.set_size = set_size
        self.min_val = min_val
        self.max_val = max_val
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # TODO: 在 [min, max] 范围内生成随机数
        values = None 
        
        # TODO: 获取排序索引（真实目标）
        _, target = None, None 
        
        # 整理为 [set_size, 1] 形状
        inputs = None 
        
        return inputs, target


def compute_loss(log_probs, targets):
    """
    计算负对数似然损失。
    
    参数：
        log_probs: [batch, seq_len]
        targets: [batch, seq_len]
    """
    # TODO: 收集目标索引的对数概率
    target_log_probs = None 
    
    # TODO: 计算均值并取负
    loss = None 
    
    return loss


def train_epoch(model, dataloader, optimizer, device):
    """单轮训练的标准循环。"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size, set_size = inputs.size(0), inputs.size(1)
        
        # TODO: 前向传播（使用教师强制）
        lengths = torch.full((batch_size,), set_size, device=device)
        pointers, log_probs, _ = None, None, None 
        
        # TODO: 计算损失并反向传播
        loss = None 
        
        total_loss += loss.item()
        correct += (pointers == targets).all(dim=1).sum().item()
        total += batch_size
    
    return total_loss / len(dataloader), correct / total


def demo_sorting():
    print('训练指针网络进行排序')
    print("-" * 30)
    
    set_size = 5
    hidden_dim = 64
    num_epochs = 20
    batch_size = 32
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备：{device}")
    
    # TODO: 初始化数据集、加载器、模型与优化器
    train_loader = None 
    model = None 
    optimizer = None 
    
    for epoch in range(1, num_epochs + 1):
        # TODO: 实现单轮训练
        train_loss, train_acc = 0.0, 0.0
        print(f"轮次 {epoch:2d} | 损失：{train_loss:.4f} | 训练准确率：{train_acc:.3f}")


if __name__ == "__main__":
    demo_sorting()
    
    print('练习 3 小结')
    print("-" * 30)
    print('\n涵盖的关键概念：\n1. 算法的监督学习：从数据中学习逻辑操作。\n2. 教师强制：在逐步解码时提供真实目标。\n3. 完全匹配准确率：衡量整个序列是否正确排序。\n    ')
