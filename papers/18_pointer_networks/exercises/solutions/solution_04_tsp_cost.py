"""
解答 4：TSP 巡回路径代价计算
按 30u30 第 18 天的格式组织

依据：Vinyals 等（2015）《指针网络》（Pointer Networks）
"""

import torch
import numpy as np

def calculate_tour_cost(points, pointers):
    """
    计算 TSP 巡回路径总长度。
    """
    cost = 0.0
    num_cities = len(pointers)
    
    for i in range(num_cities):
        # 当前城市索引
        c1 = pointers[i]
        # 下一城市索引（末尾回到 0）
        c2 = pointers[(i + 1) % num_cities]
        
        # 欧氏距离：sqrt((x2-x1)^2 + (y2-y1)^2)
        p1, p2 = points[c1], points[c2]
        dist = torch.sqrt(torch.sum((p2 - p1)**2))
        cost += dist.item()
        
    return cost
