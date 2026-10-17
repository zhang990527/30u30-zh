"""
练习 5：旅行商问题（TSP）
============================================

使用指针网络近似求解组合优化问题。

任务：给定城市坐标，找出访问所有城市且
      总欧氏距离尽可能短的巡回路径。

概念：TSP 是 NP 难问题。此处用神经网络学习效果较好的
         近似启发式方法。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
sys.path.append('..')
from implementation import ReadProcessWrite


class TSPDataset(Dataset):
    """
    TSP 实例，其监督目标由最近邻贪心启发式方法提供。
    """
    def __init__(self, num_samples, num_cities):
        self.num_samples = num_samples
        self.num_cities = num_cities
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        cities = torch.rand(self.num_cities, 2)
        tour = self._greedy_tour(cities)
        return cities, tour
    
    def _greedy_tour(self, cities):
        n = len(cities)
        unvisited = set(range(n))
        current = 0
        tour = [current]
        unvisited.remove(current)
        
        while unvisited:
            nearest = min(unvisited, 
                         key=lambda c: torch.norm(cities[current] - cities[c]))
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        return torch.tensor(tour, dtype=torch.long)


def compute_tour_length(tour, cities):
    tour_cities = cities[tour]
    distances = torch.norm(tour_cities[1:] - tour_cities[:-1], dim=1)
    return_dist = torch.norm(tour_cities[-1] - tour_cities[0])
    return (distances.sum() + return_dist).item()


def demo_tsp():
    print('训练指针网络近似求解 TSP')
    print("-" * 30)
    
    num_cities = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ReadProcessWrite(
        input_dim=2, 
        hidden_dim=128, 
        use_set_encoder=True,
        num_layers=3
    ).to(device)
    
    print('TSP 模型已初始化。')


if __name__ == "__main__":
    demo_tsp()
    
    print('练习 5 小结')
    print("-" * 30)
    print('\n涵盖的关键概念：\n1. 组合优化：从数据中学习求解搜索问题。\n2. 启发式学习：用贪心基线输出训练，学习可泛化的规则。\n3. 路径一致性：确保模型生成合法置换（不重复访问）。\n    ')
