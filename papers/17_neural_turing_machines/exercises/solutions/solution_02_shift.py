"""
解答 2：卷积移位
===============================

练习 2 的完整参考解答。
"""

import torch

def circular_shift(w, s):
    """
    参数：
        w：权重向量 [batch_size, N]
        s：移位向量 [batch_size, 3]（索引对应移位 -1、0、1）
        
    返回：
        w_s：移位后的权重 [batch_size, N]
    """
    # 用 torch.roll 高效实现循环移位
    # s[:, 0] 对应移位 -1（左），s[:, 1] 对应 0，s[:, 2] 对应 1（右）；译注：原注释与下方实现的方向存在冲突。
    # 在 torch.roll 中，正数移位向右，负数移位向左。
    
    # 初始化 w_s
    w_s = torch.zeros_like(w)
    N = w.size(1)
    
    # 移位 -1（索引向右移动相当于 torch.roll 的正数 1）；此处沿用原实现约定
    # 原注释称：论文中的 s(i) 在移位 +1 时使位置 i 的权重移动到 i+1。
    # 此处遵循 implementation.py 的约定：
    # s[0] -> 移位 -1
    # s[1] -> 移位 0
    # s[2] -> 移位 +1
    
    w_s = (
        s[:, 0] * torch.roll(w, shifts=1, dims=1) +
        s[:, 1] * w +
        s[:, 2] * torch.roll(w, shifts=-1, dims=1)
    )
    
    return w_s

if __name__ == "__main__":
    # 测试
    batch, N = 1, 10
    w = torch.zeros(batch, N)
    w[0, 5] = 1.0 
    
    # 移位 +1（s 的索引 2 对应 +1）；注意上述方向冲突
    s = torch.tensor([[0.0, 0.0, 1.0]])
    
    w_s = circular_shift(w, s)
    print('移位后的权重：', w_s)
    assert w_s[0, 6] == 1.0
    print('成功')
