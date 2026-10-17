import torch
import torch.nn as nn

class LSTMUpdate(nn.Module):
    def __init__(self, input_size, state_size):
        super().__init__()
        # 假设门的输入是 [input, state] 的拼接
        feature_dim = input_size + state_size
        
        # 定义遗忘门、输入门与候选状态的线性层
        self.gate_f = nn.Linear(feature_dim, state_size)
        self.gate_i = nn.Linear(feature_dim, state_size)
        self.gate_c = nn.Linear(feature_dim, state_size)
        
    def forward(self, input_vec, state):
        """
        参数：
            input_vec：(batch, slots, input_size)，表示注意力处理后的新信息
            state：(batch, slots, state_size)，表示上一时刻的旧记忆
        """
        # 沿最后一维拼接
        combined = torch.cat([input_vec, state], dim=-1)
        
        # 计算门的激活值
        f = torch.sigmoid(self.gate_f(combined))
        i = torch.sigmoid(self.gate_i(combined))
        c = torch.tanh(self.gate_c(combined))
        
        # 计算下一状态
        next_state = f * state + i * c
        
        return next_state

def test_gating():
    print('正在测试 LSTM 式门控……')
    batch, slots, dim = 2, 4, 16
    updater = LSTMUpdate(dim, dim) # 为简化，输入大小等于状态大小
    
    x = torch.randn(batch, slots, dim)
    m = torch.randn(batch, slots, dim)
    
    m_new = updater(x, m)
    
    assert m_new.shape == (batch, slots, dim)
    print('测试通过！')

if __name__ == "__main__":
    test_gating()
