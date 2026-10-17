"""
练习 2：实现注意力掩码
按 30u30 第 18 天的格式组织

任务：实现掩码逻辑，避免模型重复选择同一项。
依据：Vinyals 等（2015）《指针网络》（Pointer Networks）
"""

import torch
import torch.nn.functional as F

def apply_mask(scores, mask):
    """
    TODO：实现掩码逻辑。
    
    参数：
        scores（Tensor）：原始注意力分数 (batch, seq_len)
        mask（Tensor）：布尔掩码 (batch, seq_len)，True 表示要屏蔽的索引
        
    返回：
        log_probs（Tensor）：对数概率；被屏蔽位置对应概率为零，对数概率为 -inf
    """
    # TODO：用 scores.masked_fill_ 将屏蔽位置替换为 -inf
    # 再应用 log_softmax
    pass

if __name__ == "__main__":
    print('练习 2：注意力掩码')
    print("=" * 50)
    
    # 测试用例
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[False, True, False, False]]) # 屏蔽索引 1（数值 2.0）
    
    try:
        result = apply_mask(scores, mask)
        probs = torch.exp(result)
        
        print(f"原始分数：{scores.tolist()}")
        print(f"掩码：{mask.tolist()}")
        print(f"应用掩码后的概率：{probs.tolist()}")
        
        if probs[0, 1] < 1e-6:
            print('\n[OK] 掩码成功！索引 1 已被忽略。')
        else:
            print('\n[FAIL] 索引 1 未被忽略。请检查 masked_fill 逻辑。')
            
    except Exception as e:
        print(f"\n[FAIL] 发生错误：{e}")
