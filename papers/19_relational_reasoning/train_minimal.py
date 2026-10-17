"""
train_minimal.py：关系推理训练与泛化程序

本脚本提供用于训练关系网络的命令行接口。
该版本增加“集合大小泛化”评估。

主要功能：
1. 多聚合器比较：使用 sum、mean 或 max 训练。
2. 集合大小泛化：在 N 个对象上训练，在 M 个对象上测试。
3. 注入坐标：测试空间感知的影响。
4. 性能指标：跟踪不同变体的损失和准确率。

用法：
    # 使用不同聚合器训练
    python train_minimal.py --aggregator mean --epochs 20
    
    # 运行泛化测试（训练 N=5，测试 N=15）
    python train_minimal.py --train-n 5 --test-n 15 --epochs 30

原仓库引用：Santoro 等（2017）第 4.2 节（泛化）；节号待核验
作者：30u30 项目
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import time
from typing import Tuple, Dict

from implementation import RelationNetwork, add_coordinates

# ============================================================================
# 关系任务数据集
# ============================================================================

class RelationalDataset(Dataset):
    """
    关系推理任务的数据集。
    支持 furthest（空间任务）和 count（基数任务）。
    """
    def __init__(
        self, 
        mode: str = 'furthest', 
        num_samples: int = 5000, 
        num_objects: int = 10,
        use_coords: bool = False
    ):
        self.mode = mode
        self.use_coords = use_coords
        self.samples = []
        
        for _ in range(num_samples):
            # 对象为 [-1, 1] 内的二维坐标
            points = np.random.uniform(-1, 1, (num_objects, 2))
            
            if mode == 'furthest':
                # 标签：离原点最远的点的索引
                # 作为 CLEVR 中前后空间关系任务的简化替代（原仓库引用第 4.1 节）。
                distances = np.linalg.norm(points, axis=1)
                target = np.argmax(distances)
            elif mode == 'count':
                # 标签：右上象限中的点数
                # 作为 CLEVR 中“有多少”计数任务的简化替代（原仓库引用第 4.1 节）。
                # 显式测试 sum 与 mean 的归纳偏置。
                target = np.sum((points[:, 0] > 0) & (points[:, 1] > 0))
            else:
                raise ValueError(f"未知模式：{mode}")
                
            self.samples.append((points, target))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        points, target = self.samples[idx]
        points_tensor = torch.tensor(points).float()
        
        # 如果模型启用坐标感知，则使用相应工具函数
        if self.use_coords:
            points_tensor = add_coordinates(points_tensor.unsqueeze(0)).squeeze(0)
            
        return points_tensor, torch.tensor(target).long()


# ============================================================================
# 训练逻辑
# ============================================================================

def run_evaluation(model, loader, device):
    """计算给定数据加载器上准确率的工具函数。"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def run_experiment(args):
    """包含训练与泛化评估的主实验循环。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- 第 19 天实验：[任务：{args.mode}] ---")
    print(f"聚合器：{args.aggregator} | 训练 N：{args.train_n} | 测试 N：{args.test_n}")
    
    # 1. 准备训练数据（N 个对象）
    train_dataset = RelationalDataset(
        mode=args.mode, 
        num_samples=args.train_samples, 
        num_objects=args.train_n,
        use_coords=args.use_coords
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. 准备测试数据（M 个对象，用于泛化测试）
    # 若 test_n != train_n，检查 RN 逻辑能否泛化到不同大小集合。
    test_dataset = RelationalDataset(
        mode=args.mode, 
        num_samples=1000, 
        num_objects=args.test_n,
        use_coords=args.use_coords
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 3. 构建模型
    # 确定输出维度
    if args.mode == 'furthest':
        # 训练标签为 0 到 N-1，测试标签可能为 0 到 M-1。
        # 一个关键限制是：最终 f_phi MLP 的输出大小固定。
        # 跨集合大小泛化时，任务应使用固定输出空间，
        # 例如计数或二元比较，并覆盖所需标签范围。
        if args.train_n != args.test_n:
            print('[WARNING] furthest 任务的集合大小泛化受到 f_phi 输出神经元数量限制。')
        num_classes = max(args.train_n, args.test_n)
    else:
        # 计数任务：输出空间为 0 到 num_objects
        num_classes = args.test_n + 1
        
    model = RelationNetwork(
        object_dim=4 if args.use_coords else 2, 
        relation_dim=args.hidden_dim, 
        output_dim=num_classes,
        aggregator=args.aggregator
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 4. 训练循环
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            # 计数任务的标签是从 0 到计数上界的索引。
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 5 == 0 or epoch == 1:
            train_acc = run_evaluation(model, train_loader, device)
            test_acc = run_evaluation(model, test_loader, device)
            print(f"轮次 {epoch:2d}/{args.epochs} | 损失：{total_loss/len(train_loader):.4f} | "
                  f"训练准确率（N={args.train_n}): {train_acc:.3f} | 测试准确率（N={args.test_n}): {test_acc:.3f}")

    print("-" * 50)
    print(f"实验完成。泛化差距：{abs(train_acc - test_acc):.3f}")
    
    if args.save_model:
        torch.save(model.state_dict(), f"rn_{args.mode}_{args.aggregator}.pt")
        
    return model

# ============================================================================
# 命令行配置
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='第 19 天：关系网络泛化程序')
    
    # 通用参数
    parser.add_argument('--mode', type=str, default='count', choices=['furthest', 'count'],
                        help='任务模式（计数更适合泛化测试）')
    parser.add_argument('--aggregator', type=str, default='sum', choices=['sum', 'mean', 'max'],
                        help='关系池化策略（标准选择为 sum）')
    parser.add_argument('--use-coords', action='store_true', help='向对象注入 (x, y) 坐标')
    
    # 集合大小参数
    parser.add_argument('--train-n', type=int, default=10, help='训练时的集合大小')
    parser.add_argument('--test-n', type=int, default=10, help='测试时的集合大小（泛化）')
    
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--train-samples', type=int, default=10000)
    
    parser.add_argument('--save-model', action='store_true')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
