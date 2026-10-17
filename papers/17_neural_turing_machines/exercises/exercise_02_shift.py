"""
练习 2：卷积移位
===============================

NTM 通过循环卷积实现基于位置的寻址
（Graves 等，2014，第 3.3.1 节，公式 8）。

公式：
w_s(i) = sum_{j=0}^{N-1} w_g(j) s(i - j)

移位范围大小为 3（索引 0、1、2 对应移位 -1、0、1）时，
公式 8 简化为：
w_s(i) = s(0)*w_g(i+1) + s(1)*w_g(i) + s(2)*w_g(i-1)
"""

import torch

def circular_shift(w, s):
    """
    参数：
        w：门控后的权重向量 [batch_size, N]
        s：移位权重 [batch_size, 3]（对应移位 -1、0、+1）
        
    返回：
        w_s：移位后的权重 [batch_size, N]
    """
    # TODO：用 torch.roll 实现循环移位（公式 8）
    pass

if __name__ == "__main__":
    # 测试你的实现
    batch, N = 1, 10
    w = torch.zeros(batch, N)
    w[0, 5] = 1.0 # 关注槽位 5
    
    # 移位 +1（s 的索引 2）；具体方向按本文件公式约定
    s = torch.tensor([[0.0, 0.0, 1.0]])
    
    w_s = circular_shift(w, s)
    
    if w_s is not None:
        print(f"移位后的关注索引：{w_s[0].argmax().item()}")
        assert w_s[0, 6] == 1.0, '关注位置应恰好移动到索引 6'
