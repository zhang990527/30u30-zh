"""
练习 5：贪心解码循环
按 30u30 第 18 天的格式组织

任务：完成指针网络的自回归解码循环。
每步生成一个指针，更新解码器输入，
并将其用于下一步。

原仓库将此称为“读取—处理—写入”框架的写入阶段；这里是解码练习。
"""

import torch
import torch.nn as nn

def greedy_decode(decoder, pointer_head, encoder_outputs, batch_size, seq_len, hidden_size):
    """
    执行贪心解码以生成指针。
    
    参数：
        decoder（RNN）：解码器 LSTM／GRU。
        pointer_head（Module）：指针注意力模块。
        encoder_outputs (Tensor): (batch, seq_len, hidden_size)
    """
    device = encoder_outputs.device
    
    # 初始解码器输入为零
    decoder_input = torch.zeros(batch_size, 1, hidden_size, device=device)
    # 初始隐藏状态（本练习假设为零）
    decoder_hidden = None 
    
    pointers = []
    
    # TODO：循环 seq_len 步
    # 1. 将 decoder_input 传入 decoder
    # 2. 用 decoder_output 与 encoder_outputs 调用 pointer_head 得到 log_probs
    # 3. 选择最高概率的索引（argmax）
    # 4. 将该索引处的 encoder_output 用作下一步 decoder_input
    # 5. 将索引追加到 pointers 列表
    
    # return torch.tensor(pointers)
    pass

if __name__ == "__main__":
    print('练习 5：贪心解码')
    print("=" * 50)
    
    # 模拟组件
    batch_size, seq_len, hidden_size = 1, 3, 4
    encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)
    
    # 返回有效输出的模拟解码器和指针头
    class MockDecoder:
        def __call__(self, x, h):
            return torch.randn(batch_size, 1, hidden_size), None
            
    class MockPointer:
        def __call__(self, d, e, mask=None):
            # 直接返回偏向下一个顺序索引的分数
            scores = torch.zeros(batch_size, seq_len)
            return scores
            
    try:
        results = greedy_decode(MockDecoder(), MockPointer(), encoder_outputs, batch_size, seq_len, hidden_size)
        print(f"解码指针：{results}")
        
        if results is not None and len(results) == seq_len:
            print('\n[OK] 解码循环产生了正确数量的指针！')
        else:
            print('\n[FAIL] 解码循环未完成，或返回形状错误。')
            
    except Exception as e:
        print(f"\n[FAIL] 错误：{e}")
