import torch
import torch.nn as nn

class LSTMUpdate(nn.Module):
    def __init__(self, input_size, state_size):
        super().__init__()
        # 假设门的输入是 [input, state] 的拼接
        feature_dim = input_size + state_size
        
        # TODO：定义遗忘门、输入门和候选状态的线性层
        # 每层应输出大小为 state_size 的张量
        self.gate_f = # 在此填写代码
        self.gate_i = # 在此填写代码
        self.gate_c = # 在此填写代码
        
    def forward(self, input_vec, state):
        """
        参数：
            input_vec：(batch, slots, input_size)，表示注意力处理后的新信息
            state：(batch, slots, state_size)，表示上一时刻的旧记忆
        """
        # 沿最后一维拼接
        combined = torch.cat([input_vec, state], dim=-1)
        
        # TODO：计算门的激活值
        # 注意：f、i 使用 sigmoid，候选状态使用 tanh
        f = # 在此填写代码
        i = # 在此填写代码
        c = # 在此填写代码
        
        # TODO：计算下一状态
        # next_state = f * state + i * candidate
        next_state = # 在此填写代码
        
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
