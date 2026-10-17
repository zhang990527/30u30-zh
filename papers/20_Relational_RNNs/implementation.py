"""
第 20 天：关系循环神经网络（RMC）实现
=====================================================================

关系记忆核心（RMC）的 PyTorch 教学实现，包含：
- 对记忆槽位进行多头点积注意力（MHDPA）
- 使用 LSTM 式门控更新记忆
- 比较 RMC 与标准 LSTM 基线

论文：《关系循环神经网络》（Relational recurrent neural networks）
        作者：Adam Santoro 等（2018）

可理解为这样的 RNN：隐藏状态是一组向量，
在更新前会相互交流。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelationalMemory(nn.Module):
    """
    Santoro 等人（2018）提出的关系记忆核心（RMC）。
    
    循环模块，在每个时间步使用自注意力，
    使记忆槽位交互。
    """
    def __init__(self, mem_slots, mem_size, input_size, num_heads=4, num_blocks=1, dropout=0.1):
        super().__init__()
        self.mem_slots = mem_slots
        self.mem_size = mem_size
        self.input_size = input_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        # ---------------------------------------------------------------------
        # 1. 输入投影
        # ---------------------------------------------------------------------
        # 实现说明：
        # 原仓库将“把投影输入与记忆矩阵每行拼接”归于 Santoro 等（2018）第 2 节；
        # 译注：官方 Sonnet 源码将输入作为额外槽位，不能将此处拼接方式当作已核实的论文原话。
        # 本章约定：M_in = [M_{t-1}; x_projected_tiled]
        #
        # 将输入投影为与记忆槽位相同大小，
        # 以支持相加或拼接交互。
        self.input_projector = nn.Linear(input_size, mem_size)
        
        # ---------------------------------------------------------------------
        # 2. 注意力机制：对记忆槽位应用自注意力
        # ---------------------------------------------------------------------
        # 注意力输入为拼接后的 [Memory; Input_Tiled]。
        # 特征维度 = mem_size（记忆）+ mem_size（输入）
        feature_dim_in = 2 * mem_size 
        
        self.attention_key_size = mem_size // num_heads 
        self.attention_value_size = mem_size // num_heads

        # Q、K、V 投影
        self.q_proj = nn.Linear(feature_dim_in, mem_size)
        self.k_proj = nn.Linear(feature_dim_in, mem_size)
        self.v_proj = nn.Linear(feature_dim_in, mem_size)
        
        self.output_proj = nn.Linear(mem_size, mem_size) 
        
        self.layer_norm_mem = nn.LayerNorm(mem_size)
        self.layer_norm_output = nn.LayerNorm(mem_size)
        
        # ---------------------------------------------------------------------
        # 3. 门控机制：LSTM 式输入门／遗忘门
        # ---------------------------------------------------------------------
        # 实现说明：
        # 标准 RMC 实现（例如 DeepMind Sonnet）用门控
        # 将注意力处理后的候选记忆与旧记忆结合。
        #
        # 更新规则：
        # m_t = f_t * m_{prev} + i_t * tanh(m_attended)
        
        self.gate_forget = nn.Linear(feature_dim_in, mem_size)
        self.gate_input = nn.Linear(feature_dim_in, mem_size)
        
        # 将遗忘门偏置初始化为 1.0，以鼓励保留长期依赖
        nn.init.constant_(self.gate_forget.bias, 1.0)
        # 基本 RMC 无输出门，直接产生下一记忆状态。
        
        # 状态初始化
        # 原仓库说明：“将初始记忆状态视为可学习参数。”这是本实现选择。
        self.initial_memory = nn.Parameter(torch.randn(1, mem_slots, mem_size) * 0.01)

    def forward(self, input_step, memory_state=None):
        """
        RMC 的单步更新。
        
        input_step: (batch, input_size)
        memory_state: (batch, mem_slots, mem_size)
        """
        batch_size = input_step.size(0)
        
        if memory_state is None:
            memory_state = self.initial_memory.expand(batch_size, -1, -1)
            
        # ---------------------------------------------------------------------
        # 1. 投影输入并复制
        # ---------------------------------------------------------------------
        # input_proj: (batch, mem_size)
        input_proj = self.input_projector(input_step)
        
        # 复制为：(batch, mem_slots, mem_size)
        input_tiled = input_proj.unsqueeze(1).expand(-1, self.mem_slots, -1)
        
        # ---------------------------------------------------------------------
        # 2. 扩充记忆（记忆 + 输入）
        # ---------------------------------------------------------------------
        # memory_aug: (batch, mem_slots, 2*mem_size)
        memory_aug = torch.cat([memory_state, input_tiled], dim=-1)
        
        # ---------------------------------------------------------------------
        # 3. 多头自注意力
        # ---------------------------------------------------------------------
        # 从 [M_old; x] 创建 Q、K、V
        q = self.q_proj(memory_aug)
        k = self.k_proj(memory_aug)
        v = self.v_proj(memory_aug)
        
        # 重塑为多头形式：(batch, heads, slots, head_dim)
        head_dim = self.mem_size // self.num_heads
        
        q = q.view(batch_size, self.mem_slots, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, self.mem_slots, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, self.mem_slots, self.num_heads, head_dim).transpose(1, 2)
        
        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        probs = F.softmax(scores, dim=-1)
        
        # 加权求和并重新组合
        output = torch.matmul(probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, self.mem_slots, self.mem_size)
        
        # 线性变换 + 残差连接 + 层归一化
        output = self.output_proj(output)
        m_attended = output + memory_state
        m_attended = self.layer_norm_mem(m_attended)
        
        # ---------------------------------------------------------------------
        # 4. 门控更新
        # ---------------------------------------------------------------------
        # 用原始扩充状态 [M_old; x] 计算门
        f_gate = torch.sigmoid(self.gate_forget(memory_aug))
        i_gate = torch.sigmoid(self.gate_input(memory_aug))
        
        # 候选记忆状态（tanh 非线性）
        # 用注意力处理后的 output 作为新内容的基础
        candidate = torch.tanh(output) 
        
        # 最终 LSTM 式更新：
        # m_new = f * m_old + i * candidate
        next_memory = f_gate * memory_state + i_gate * candidate
        
        # 最终 LSTM 式更新：
        # m_new = f * m_old + i * candidate
        next_memory = f_gate * memory_state + i_gate * candidate
        
        return next_memory, probs

class RelationalRNN(nn.Module):
    """
    用 RMC 处理序列的封装。
    """
    def __init__(self, input_size, mem_slots, mem_size, output_size, num_heads=4):
        super().__init__()
        self.rmc = RelationalMemory(mem_slots, mem_size, input_size, num_heads)
        self.out = nn.Linear(mem_slots * mem_size, output_size)
    
    def forward(self, x, return_attention=False):
        # x: (batch, seq, input)
        batch_size = x.size(0)
        memory = None
        attention_history = []
        
        for t in range(x.size(1)):
            input_step = x[:, t, :]
            memory, probs = self.rmc(input_step, memory)
            if return_attention:
                attention_history.append(probs)
        
        # 展平记忆以供输出
        # (batch, slots*size)
        flat_memory = memory.view(batch_size, -1)
        output = self.out(flat_memory)
        
        if return_attention:
            # 堆叠历史：(batch, seq, heads, slots, slots)
            return output, torch.stack(attention_history, dim=1)
            
        return output

class StandardLSTM(nn.Module):
    """
    用于比较的 LSTM 基线模型。
    使用与 RMC 接近的参数量，以尽量公平比较；公平性仍取决于具体配置。
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (batch, seq, input)
        # LSTM 返回 (output, (h_n, c_n))
        # output: (batch, seq, hidden_size)
        output, (h_n, c_n) = self.lstm(x)
        
        # 本章距离任务只使用最终状态
        # h_n: (num_layers, batch, hidden_size)
        final_state = h_n[-1]
        
        return self.out(final_state)
