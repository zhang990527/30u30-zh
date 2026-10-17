"""
解答 5：贪心解码循环
按 30u30 第 18 天的格式组织

依据：Vinyals 等（2015）《指针网络》（Pointer Networks）
"""

import torch

def greedy_decode(decoder, pointer_head, encoder_outputs, batch_size, seq_len, hidden_size):
    """
    执行贪心解码以生成指针。
    """
    device = encoder_outputs.device
    decoder_input = torch.zeros(batch_size, 1, hidden_size, device=device)
    decoder_hidden = None 
    
    pointers = []
    
    for _ in range(seq_len):
        # 1. 解码器单步更新
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        
        # 2. 指针注意力分数
        log_probs = pointer_head(decoder_output.squeeze(1), encoder_outputs)
        
        # 3. 贪心选择
        idx = log_probs.argmax(dim=-1)
        pointers.append(idx.item())
        
        # 4. 用选中元素的表示更新下一步输入
        # 为简化解答，假设 batch_size=1
        decoder_input = encoder_outputs[0, idx].unsqueeze(1)
        
    return torch.tensor(pointers)
