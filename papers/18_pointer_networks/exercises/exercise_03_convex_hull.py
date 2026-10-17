"""
练习 3：凸包数据格式化
按 30u30 第 18 天的格式组织

任务：实现函数，计算凸包任务的“指针”目标。
给定点集，找出组成凸包的点的索引，
按逆时针顺序输出。

这是 Ptr-Net 论文中的经典组合任务（第 3.1 节，结果见第 4.2 节）。
"""

import numpy as np
import torch

def get_convex_hull_pointers(points):
    """
    计算构成凸包的点的索引。
    
    参数：
        points（ndarray）：大小为 (N, 2) 的 (x, y) 坐标数组。
        
    返回：
        indices（整数列表）：按逆时针凸包顺序排列的输入点索引。
    """
    # TODO：实现简单凸包算法，例如 Graham 扫描或单调链，
    # 或使用 scipy.spatial.ConvexHull 等库。
    # 本练习假定有 N 个点，返回相应索引。
    
    # 提示：scipy.spatial.ConvexHull(points).vertices 按顺序返回索引。
    pass

if __name__ == "__main__":
    print('练习 3：凸包格式化')
    print("=" * 50)
    
    # 测试点：一个正方形与一个内部点
    points = np.array([
        [0, 0],   # 0
        [1, 0],   # 1
        [1, 1],   # 2
        [0, 1],   # 3
        [0.5, 0.5] # 4（内部）
    ])
    
    try:
        from scipy.spatial import ConvexHull
        # 逻辑检查：
        hull_indices = get_convex_hull_pointers(points)
        
        print(f"输入点：\n{points}")
        print(f"你的凸包索引：{hull_indices}")
        
        # 验证
        expected = ConvexHull(points).vertices.tolist()
        # 逆时针顺序可能从不同顶点开始，应同时检查点集和顺序
        if set(hull_indices) == set(expected) and len(hull_indices) == len(expected):
            print('\n[OK] 正确选择了凸包顶点！')
        else:
            print(f"\n[FAIL] 预期索引类似 {expected}，实际得到 {hull_indices}")
            
    except ImportError:
        print('\n[SKIP] 未找到 Scipy。请安装后再运行验证。')
    except Exception as e:
        print(f"\n[FAIL] 错误：{e}")
