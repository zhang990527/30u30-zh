import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 可以导入之前的练习；为简化，此处保持自包含
class RelationalMemoryStep(nn.Module):
    def __init__(self, mem_slots, mem_size, input_size, num_heads=4):
        super().__init__()
        self.mem_slots = mem_slots
        self.mem_size = mem_size
        self.num_heads = num_heads
        
        # 1. 输入投影：输入 -> 记忆大小
        self.input_proj = nn.Linear(input_size, mem_size)
        
        # 2. 注意力投影
        # 注意力输入为 [Memory; Input_Tiled]，维度为 2 * mem_size
        feature_dim = 2 * mem_size
        
        self.q_proj = nn.Linear(feature_dim, mem_size)
        self.k_proj = nn.Linear(feature_dim, mem_size)
        self.v_proj = nn.Linear(feature_dim, mem_size)
        
        self.output_proj = nn.Linear(mem_size, mem_size)
        self.layer_norm = nn.LayerNorm(mem_size)
        
        # 3. 门控（输入、遗忘）
        # 使用简单的 LSTM 式更新
        self.gate_f = nn.Linear(feature_dim, mem_size)
        self.gate_i = nn.Linear(feature_dim, mem_size)
        
    def forward(self, input_step, memory_state):
        batch_size = input_step.size(0)
        
        # TODO：投影输入并复制，以匹配记忆槽位
        # input_proj: (batch, mem_size)
        # input_tiled: (batch, slots, mem_size)
        input_proj = # 在此填写代码
        input_tiled = input_proj.unsqueeze(1).expand(-1, self.mem_slots, -1)
        
        # TODO：拼接记忆与复制后的输入
        # memory_aug: (batch, slots, 2*mem_size)
        memory_aug = # 在此填写代码
        
        # TODO：对 memory_aug 执行多头注意力
        # 本练习可使用 PyTorch 的 F.multi_head_attention_forward，
        # 或自行实现“投影 -> 分数 -> softmax -> 加权值”的逻辑。
        # 为巩固练习 1，此处手工完成投影和矩阵乘法。
        
        q = self.q_proj(memory_aug)
        k = self.k_proj(memory_aug)
        v = self.v_proj(memory_aug)
        
        # 重塑为多头形式 (batch, heads, slots, dim/heads)
        head_dim = self.mem_size // self.num_heads
        q = q.view(batch_size, self.mem_slots, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, self.mem_slots, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, self.mem_slots, self.num_heads, head_dim).transpose(1, 2)
        
        # 注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, v)
        
        # 恢复形状
        attended = attended.transpose(1, 2).contiguous().view(batch_size, self.mem_slots, self.mem_size)
        
        # 输出投影 + 记忆残差连接 + 层归一化
        output = self.output_proj(attended)
        m_attended = self.layer_norm(output + memory_state)
        
        # TODO：门控更新
        # 使用 memory_aug 计算 f_gate 和 i_gate
        # 用 tanh(output) 或 tanh(m_attended) 计算候选状态
        # 更新记忆
        
        f_gate = # 在此填写代码
        i_gate = # 在此填写代码
        
        next_memory = # 在此填写代码
        
        return next_memory

def test_rmc_step():
    print('正在测试 RMC 单步更新……')
    batch, slots, mem_dim, input_dim = 2, 4, 32, 10
    rmc = RelationalMemoryStep(slots, mem_dim, input_dim)
    
    x = torch.randn(batch, input_dim)
    m = torch.randn(batch, slots, mem_dim)
    
    m_new = rmc(x, m)
    
    assert m_new.shape == (batch, slots, mem_dim)
    print('测试通过！')

if __name__ == "__main__":
    test_rmc_step()
