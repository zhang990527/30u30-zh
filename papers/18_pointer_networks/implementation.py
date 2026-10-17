"""
指针网络（Ptr-Net）实现
按 30u30 第 18 天的格式组织

用 PyTorch 编写的完整教学实现。
用于输出索引对应输入位置的任务，
如排序、凸包与旅行商问题。

依据：Vinyals 等（2015）《指针网络》（Pointer Networks）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerAttention(nn.Module):
    """
    指针注意力机制（Vinyals 等，2015）。
    
    此模块不把输入混合成上下文向量，
    而是将注意力分数归一化后直接作为选择概率。
    
    公式（公式 3）：
    u_j^i = v^T * tanh(W_1 * e_j + W_2 * d_i)
    p(C_i | C_1, ..., C_{i-1}, P) = softmax(u^i)
    
    参数：
        hidden_size（int）：编码器与解码器隐藏状态的维度。
    """
    def __init__(self, hidden_size):
        super().__init__()
        # W1 变换编码器状态（键）
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        # W2 变换解码器状态（查询）
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        # v 将组合后的能量投影为标量分数
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        计算指针选择的对数概率。
        
        参数：
            decoder_hidden（Tensor）：当前解码器状态 (batch, hidden_size)
            encoder_outputs（Tensor）：全部编码器状态 (batch, seq_len, hidden_size)
            mask（Tensor，可选）：无效位置的布尔掩码 (batch, seq_len)
            
        返回：
            log_probs（Tensor）：选择的对数概率 (batch, seq_len)
        """
        # === 第 1 步：状态投影 ===
        # (batch, 1, hidden_size)
        decoder_proj = self.W2(decoder_hidden).unsqueeze(1)
        # (batch, seq_len, hidden_size)
        encoder_proj = self.W1(encoder_outputs)
        
        # === 第 2 步：加性注意力（公式 3）===
        # (batch, seq_len, hidden_size)
        combined = torch.tanh(encoder_proj + decoder_proj)
        
        # === 第 3 步：提取分数 ===
        # (batch, seq_len)
        scores = self.v(combined).squeeze(-1)
        
        # === 第 4 步：应用可选掩码 ===
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
            
        return F.log_softmax(scores, dim=-1)

class PointerNetwork(nn.Module):
    """
    用于序列到序列指向任务的指针网络。
    
    架构：
    - 输入投影：从原始特征映射到隐藏空间
    - 编码器（LSTM）：处理输入序列
    - 解码器（LSTM）：自回归生成输出指针
    - 指针注意力：选择头
    
    参数：
        input_size（int）：每个输入项的特征数，例如排序为 1、坐标为 2。
        hidden_size（int）：嵌入和 LSTM 隐藏状态的维度。
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 输入坐标／特征的基础投影
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.pointer = PointerAttention(hidden_size)
        
    def forward(self, inputs, trg_len=None):
        """
        用贪心选择的指针执行前向传播。
        
        参数：
            inputs（Tensor）：输入序列特征 (batch, seq_len, input_size)
            trg_len（int，可选）：输出步数，默认为输入 seq_len。
            
        返回：
            log_pointers（Tensor）：选择对数概率序列 (batch, trg_len, seq_len)；原注释误称普通概率
        """
        batch_size, seq_len, _ = inputs.size()
        if trg_len is None:
            trg_len = seq_len
            
        # === 第 1 步：投影与编码 ===
        x = self.input_proj(inputs)
        encoder_outputs, (h, c) = self.encoder(x)
        
        # === 第 2 步：初始化解码器 ===
        # 初始解码器输入为零，隐藏状态来自编码器
        decoder_input = torch.zeros(batch_size, 1, self.hidden_size, device=inputs.device)
        decoder_hidden = (h, c)
        
        all_log_pointers = []
        # 可在 TSP 等任务中用掩码防止重复选择；需检查下方是否实际启用
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=inputs.device)
        
        # === 第 3 步：自回归解码 ===
        for _ in range(trg_len):
            # 3a. 执行解码器单步更新
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # 3b. 计算指针概率
            log_probs = self.pointer(decoder_output.squeeze(1), encoder_outputs, mask)
            all_log_pointers.append(log_probs)
            
            # 3c. 选择下一个输入（演示采用贪心选择）
            top_idx = log_probs.argmax(dim=-1)
            
            # 3d. 用选中项的表示更新解码器输入
            decoder_input = torch.stack([encoder_outputs[i, idx] for i, idx in enumerate(top_idx)]).unsqueeze(1)
            
        return torch.stack(all_log_pointers, dim=1)

if __name__ == "__main__":
    print('指针网络实现')
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointerNetwork(input_size=1, hidden_size=64).to(device)
    
    # 用 2 个长度为 5 的随机序列验证
    dummy_input = torch.rand(2, 5, 1).to(device)
    output = model(dummy_input)
    
    print(f"设备：{device}")
    print(f"输入形状：{dummy_input.shape}（批次，序列长度，特征）")
    print(f"输出形状：{output.shape}（批次，步数，指针）")
    print('\n[OK] 实现验证完成。')
