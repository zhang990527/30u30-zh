"""
解答 3：凸包数据格式化
按 30u30 第 18 天的格式组织

依据：Vinyals 等（2015）《指针网络》（Pointer Networks）
"""

import numpy as np
from scipy.spatial import ConvexHull

def get_convex_hull_pointers(points):
    """
    计算构成凸包的点的索引。
    """
    # 直接用 scipy 优化的 ConvexHull，按顺序返回顶点索引。
    hull = ConvexHull(points)
    return hull.vertices.tolist()
