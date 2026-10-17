"""
练习 5：锐化机制
================================

NTM 的寻址权重常因循环卷积（公式 8）变得“模糊”。
锐化使权重分布更集中，防止关注扩散。

公式（Graves 等，2014，第 3.3.1 节，公式 9）：
w_sharp[i] = w[i]^gamma / sum w[j]^gamma

说明：
- gamma（锐化因子）应 >= 1。
- gamma 增大时，权重分布更加“尖锐”，峰值更突出。
"""

import torch

def sharpening(w, gamma):
    """
    参数：
        w：权重向量 [batch_size, N]
        gamma：锐化因子 [batch_size, 1]
        
    返回：
        w_sharp：锐化后的权重 [batch_size, N]
    """
    # TODO：实现公式 9
    # 1. 对 w 求 gamma 次幂
    # 2. 沿 dim 1 重新归一化
    pass

if __name__ == "__main__":
    # 测试你的实现
    batch, N = 1, 5
    w = torch.tensor([[0.2, 0.3, 0.2, 0.1, 0.2]])
    gamma = torch.tensor([[5.0]])
    
    w_sharp = sharpening(w, gamma)
    
    if w_sharp is not None:
        print(f"锐化前最大权重：{w.max().item():.2f}")
        print(f"锐化后最大权重：{w_sharp.max().item():.2f}")
        assert w_sharp[0, 1] > w[0, 1], '锐化后峰值应增大'
        assert torch.allclose(w_sharp.sum(), torch.tensor(1.0)), '权重总和必须为 1.0'
