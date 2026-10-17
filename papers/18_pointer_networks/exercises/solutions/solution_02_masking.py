"""
解答 2：实现注意力掩码
按 30u30 第 18 天的格式组织

依据：Vinyals 等（2015）《指针网络》（Pointer Networks）
"""

import torch
import torch.nn.functional as F

def apply_mask(scores, mask):
    """
    对注意力分数应用布尔掩码，返回 log_probs。
    """
    # 将屏蔽位置置为 -inf，使 softmax 后概率为 0
    masked_scores = scores.masked_fill(mask, float('-inf'))
    return F.log_softmax(masked_scores, dim=-1)
