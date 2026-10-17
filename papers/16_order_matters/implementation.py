"""
指针网络与集合处理实现
===================================================

本模块实现用神经网络处理集合的核心组件：
1. 指针网络——从输入中选择元素
2. 集合编码器——原仓库称为顺序不变编码，逐元素输出实际为置换等变，见 README
3. 读取—处理—写出（Read-Process-Write）框架

论文：《顺序很重要：面向集合的序列到序列学习》（Order Matters: Sequence to Sequence for Sets）
       Vinyals et al. (2015)
       https://arxiv.org/abs/1511.06391
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PointerNetwork(nn.Module):
    """
    从集合中选择元素的指针网络。
    
    该网络不从固定词表生成词元，而是“指向”
    输入序列中的元素。
    
    关键性质：输出空间由输入集合的索引构成。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 编码器：处理输入集合
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 解码器：生成指针
        self.decoder_cell = nn.LSTMCell(input_dim, hidden_dim)
        
        # 用于指针选择的注意力机制
        # 公式：e_i = v^T tanh(W1*encoder_i + W2*decoder)
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        max_steps: int,
        teacher_forcing: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        指针网络的前向传播。

        参数：
            inputs: [batch, seq_len, input_dim]——输入集合（可能已填充）
            input_lengths: [batch]——填充前的实际长度
            max_steps: 要指向的元素数量
            teacher_forcing: [batch, max_steps]——真实目标指针
            temperature: softmax 温度
            
        返回：
            pointers: [batch, max_steps]——选中的索引
            log_probs: [batch, max_steps]——对数概率
            attention_weights: 用于可视化的 [batch, seq_len] 张量列表
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        device = inputs.device
        
        # 阶段 1：编码
        encoder_outputs, (hidden, cell) = self.encoder(inputs)
        
        # 初始化解码器状态
        decoder_hidden = hidden[-1]
        decoder_cell = cell[-1]
        
        # 从零向量开始（尚无上一次选择）
        decoder_input = torch.zeros(batch_size, self.input_dim, device=device)
        
        # 填充掩码：1 = 有效，0 = 填充
        mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = (mask < input_lengths.unsqueeze(1)).float()
        
        # 阶段 2：解码
        pointers = []
        log_probs_list = []
        attention_weights_list = []
        
        # 跟踪已选元素，以实现无放回采样
        selection_mask = mask.clone()
        
        for step in range(max_steps):
            decoder_hidden, decoder_cell = self.decoder_cell(
                decoder_input, (decoder_hidden, decoder_cell)
            )
            decoder_hidden = self.dropout(decoder_hidden)
            
            # 注意力机制（原仓库标注为论文公式 1）
            encoder_proj = self.W1(encoder_outputs)
            decoder_proj = self.W2(decoder_hidden).unsqueeze(1)
            combined = torch.tanh(encoder_proj + decoder_proj)
            scores = self.v(combined).squeeze(-1) / temperature
            
            # 屏蔽填充位置与已选元素
            scores = scores.masked_fill(selection_mask == 0, float('-inf'))
            
            probs = F.softmax(scores, dim=-1)
            log_probs = F.log_softmax(scores, dim=-1)
            
            if teacher_forcing is not None and step < teacher_forcing.size(1):
                pointer = teacher_forcing[:, step]
            else:
                pointer = torch.argmax(probs, dim=-1)
            
            pointers.append(pointer)
            log_probs_list.append(log_probs.gather(1, pointer.unsqueeze(1)))
            attention_weights_list.append(probs)
            
            selection_mask = selection_mask.scatter(1, pointer.unsqueeze(1), 0)
            decoder_input = inputs[torch.arange(batch_size, device=device), pointer]
        
        pointers = torch.stack(pointers, dim=1)
        log_probs = torch.cat(log_probs_list, dim=1)
        
        return pointers, log_probs, attention_weights_list


class SetEncoder(nn.Module):
    """
    集合编码器（原仓库称顺序不变；逐元素输出实际为置换等变，见 README）。
    原仓库声称保证 encoder({a, b}) = encoder({b, a})；该等式需区分逐元素输出与聚合表示。
    使用不带位置编码的自注意力。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 嵌入输入
        x = self.embedding(x)
        
        # 不带位置编码的自注意力产生置换等变的逐元素输出（原仓库此处称顺序不变）
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        
        return x


class ReadProcessWrite(nn.Module):
    """
    读取—处理—写出框架的实现。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_set_encoder: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.use_set_encoder = use_set_encoder
        
        if use_set_encoder:
            # 阶段 1：读取
            self.encoder = SetEncoder(input_dim, hidden_dim, **kwargs)
            # 阶段 3：写出
            self.pointer_net = PointerNetwork(hidden_dim, hidden_dim, **kwargs)
        else:
            self.pointer_net = PointerNetwork(input_dim, hidden_dim, **kwargs)
    
    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        max_steps: int,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        if self.use_set_encoder:
            encoded = self.encoder(inputs)
            pointers, log_probs, attentions = self.pointer_net(
                encoded, input_lengths, max_steps, **kwargs
            )
        else:
            pointers, log_probs, attentions = self.pointer_net(
                inputs, input_lengths, max_steps, **kwargs
            )
        
        return pointers, log_probs, attentions


def compute_loss(log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_log_probs = log_probs.gather(1, targets)
    return -target_log_probs.mean()


def accuracy(pointers: torch.Tensor, targets: torch.Tensor) -> float:
    correct = (pointers == targets).all(dim=1).float()
    return correct.mean().item()


if __name__ == "__main__":
    print('指针网络实现已加载。')
