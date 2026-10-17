import torch
from implementation import add_coordinates

def test_spatial_logic():
    """
    TODO：验证坐标注入工具函数。
    第 3.1 节的原仓库引文：“对于 CLEVR 任务，我们发现 [...] 将每个对象的
    绝对 (x, y) 坐标拼接到其特征向量很有用。”
    
    1. 创建随机对象（B=1、N=4、D=8）。
    2. 使用 add_coordinates 添加坐标。
    3. 验证新维度为 D+2。
    4. 验证第一个对象的 x 坐标为 -1.0。
    """
    # 在此填写代码
    pass

if __name__ == "__main__":
    print("-" * 50)
    print('练习 5：空间／坐标逻辑')
    print("-" * 50)
    test_spatial_logic()
    print('练习 5：[COMPLETE]')
