"""
指针网络最小训练脚本
按 30u30 第 18 天的格式组织

在排序任务上训练指针网络，学习按输入数字递增顺序
输出它们的索引。

用法：
    python train_minimal.py --epochs 20 --seq_len 5
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import time
from implementation import PointerNetwork

class SortingDataset(Dataset):
    """
    随机数序列及其排序索引的数据集。
    
    参数：
        num_samples（int）：生成的序列总数。
        seq_len（int）：每个序列的长度。
    """
    def __init__(self, num_samples=5000, seq_len=5):
        self.samples = []
        for _ in range(num_samples):
            # 生成 [0, 1) 内的随机数
            nums = np.random.rand(seq_len)
            # 目标为能够将数组排序的索引
            indices = np.argsort(nums)
            self.samples.append((nums, indices))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        nums, indices = self.samples[idx]
        return torch.tensor(nums).float().unsqueeze(-1), torch.tensor(indices).long()

def train(epochs=20, seq_len=5, batch_size=64, hidden_size=64, lr=0.001):
    """
    指针网络的标准训练逻辑。
    
    参数：
        epochs（int）：遍历训练数据的次数。
        seq_len（int）：输入序列长度。
        batch_size（int）：每次更新的样本数。
        hidden_size（int）：模型容量。
        lr（float）：Adam 学习率。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # === 第 1 步：准备数据 ===
    train_data = SortingDataset(10000, seq_len=seq_len)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # === 第 2 步：初始化模型 ===
    model = PointerNetwork(input_size=1, hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    
    print(f"\n正在以下任务上训练指针网络：'{device}'")
    print(f"序列长度：{seq_len} | 隐藏状态大小：{hidden_size}")
    print("=" * 60)
    
    start_time = time.time()
    
    # === 第 3 步：训练循环 ===
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(x)
            
            # 计算损失（展平批次和时间维）
            loss = criterion(output.view(-1, seq_len), y.view(-1))
            
            # 反向传播
            loss.backward()
            # Ptr-Net 通常通过梯度裁剪稳定 RNN 训练
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        
        # 每 5 轮打印进度
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"轮次 {epoch:2d}/{epochs} | 损失：{avg_loss:.4f} | 用时：{elapsed:.1f}s")

    print("=" * 60)
    print(f"训练完成！最终损失：{avg_loss:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "model_weights.pth")
    print('[OK] 模型已保存至 model_weights.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='指针网络训练')
    parser.add_argument("--epochs", type=int, default=20, help='训练轮数')
    parser.add_argument("--seq_len", type=int, default=5, help='排序序列长度')
    parser.add_argument("--hidden_size", type=int, default=64, help='LSTM 隐藏状态大小')
    parser.add_argument("--lr", type=float, default=0.001, help='学习率')
    args = parser.parse_args()
    
    train(
        epochs=args.epochs, 
        seq_len=args.seq_len, 
        hidden_size=args.hidden_size, 
        lr=args.lr
    )
