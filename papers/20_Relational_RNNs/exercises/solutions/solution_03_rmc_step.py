import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelationalMemoryStep(nn.Module):
    def __init__(self, mem_slots, mem_size, input_size, num_heads=4):
        super().__init__()
        self.mem_slots = mem_slots
        self.mem_size = mem_size
        self.num_heads = num_heads
        
        # 1. 输入投影：输入 -> 记忆大小
        self.input_proj = nn.Linear(input_size, mem_size)
        
        # 2. 注意力投影
        feature_dim = 2 * mem_size
        self.q_proj = nn.Linear(feature_dim, mem_size)
        self.k_proj = nn.Linear(feature_dim, mem_size)
        self.v_proj = nn.Linear(feature_dim, mem_size)
        
        self.output_proj = nn.Linear(mem_size, mem_size)
        self.layer_norm = nn.LayerNorm(mem_size)
        
        # 3. 门控（LSTM 风格）
        self.gate_f = nn.Linear(feature_dim, mem_size)
        self.gate_i = nn.Linear(feature_dim, mem_size)
        
    def forward(self, input_step, memory_state):
        batch_size = input_step.size(0)
        
        # 投影并复制输入
        input_proj = self.input_projector(input_step) if hasattr(self, 'input_projector') else self.input_proj(input_step)
        input_tiled = input_proj.unsqueeze(1).expand(-1, self.mem_slots, -1)
        
        # 拼接记忆与复制后的输入
        memory_aug = torch.cat([memory_state, input_tiled], dim=-1)
        
        # 执行多头注意力
        q = self.q_proj(memory_aug)
        k = self.k_proj(memory_aug)
        v = self.v_proj(memory_aug)
        
        head_dim = self.mem_size // self.num_heads
        q = q.view(batch_size, self.mem_slots, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, self.mem_slots, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, self.mem_slots, self.num_heads, head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, v)
        
        attended = attended.transpose(1, 2).contiguous().view(batch_size, self.mem_slots, self.mem_size)
        
        output = self.output_proj(attended)
        m_attended = self.layer_norm(output + memory_state)
        
        # 门控更新
        f_gate = torch.sigmoid(self.gate_f(memory_aug))
        i_gate = torch.sigmoid(self.gate_i(memory_aug))
        
        candidate = torch.tanh(output) # 新信息
        
        next_memory = f_gate * memory_state + i_gate * candidate
        
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
