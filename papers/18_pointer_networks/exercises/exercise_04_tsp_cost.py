"""
练习 4：TSP 巡回路径代价计算
按 30u30 第 18 天的格式组织

任务：根据指针（索引）序列，计算 TSP 巡回路径的
欧氏距离总和。

本练习帮助理解：
- 指针与原始坐标空间的关系。
- 为什么最小化路径长度是 TSP 中 Ptr-Net 的目标。
"""

import torch
import numpy as np

def calculate_tour_cost(points, pointers):
    """
    计算 TSP 巡回路径总长度。
    
    参数：
        points（Tensor）：城市坐标 (N, 2)。
        pointers（整数列表）：城市索引序列，即巡回路径。
        
    返回：
        cost（float）：欧氏距离总和。
    """
    # TODO：
    # 1. 遍历指针。
    # 2. 累加相邻城市间距离，即 points[p1] 到 points[p2]。
    # 3. 别忘记闭合路径：最后一个城市返回第一个。
    
    pass

if __name__ == "__main__":
    print('练习 4：TSP 代价分析')
    print("=" * 50)
    
    # 测试城市：1x1 单位正方形
    cities = torch.tensor([
        [0.0, 0.0], # 0
        [1.0, 0.0], # 1
        [1.0, 1.0], # 2
        [0.0, 1.0]  # 3
    ])
    
    # 巡回路径：0 -> 1 -> 2 -> 3 -> 0
    pointers = [0, 1, 2, 3]
    
    try:
        cost = calculate_tour_cost(cities, pointers)
        print(f"城市：\n{cities}")
        print(f"巡回路径：{pointers}")
        print(f"你计算的代价：{cost:.4f}")
        
        # 验证：正方形周长 = 1+1+1+1 = 4.0
        if abs(cost - 4.0) < 1e-4:
            print('\n[OK] 代价计算正确！')
        else:
            print(f"\n[FAIL] 预期代价为 4.0，实际得到 {cost}")
            
    except Exception as e:
        print(f"\n[FAIL] 错误：{e}")
