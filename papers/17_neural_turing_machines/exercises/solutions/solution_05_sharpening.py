"""
解答 5：锐化机制
================================

练习 5 的完整参考解答。
"""

import torch

def sharpening(w, gamma):
    """
    参数：
        w：模糊权重向量 [batch_size, N]
        gamma：锐化因子 [batch_size, 1]
        
    返回：
        w_sharp：锐化后的权重 [batch_size, N]
    """
    # 1. 对 w 求 gamma 次幂
    # gamma: [batch, 1]
    # w: [batch, N]
    w_pow = w ** gamma
    
    # 2. 重新归一化
    w_sharp = w_pow / (w_pow.sum(dim=1, keepdim=True) + 1e-8)
    
    return w_sharp

if __name__ == "__main__":
    # 测试
    batch, N = 1, 5
    w = torch.tensor([[0.2, 0.3, 0.2, 0.1, 0.2]])
    gamma = torch.tensor([[5.0]])
    
    w_sharp = sharpening(w, gamma)
    print('原始 w：', w)
    print('锐化后的 w：', w_sharp)
    assert w_sharp[0, 1] > w[0, 1]
    assert torch.allclose(w_sharp.sum(), torch.tensor(1.0))
    print('成功')
