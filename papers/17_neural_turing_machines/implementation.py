"""
神经图灵机（NTM）实现
==========================================

本文件实现 Graves 等人（2014）描述的
神经图灵机架构，用于本章教学。

关键原则：
1. 可微性：读取、写入、寻址都可微，
   因而可通过反向传播端到端训练。
2. 解耦存储：不同于标准 RNN，存储容量（N x M）
   可独立于控制器参数数量调整。
3. 寻址：同时使用基于内容（相似度）与基于位置（移位）
   的寻址机制。

对应位置：
- 第 3 节：存储器与读写头
- 第 3.3.1 节：寻址流程
- 第 4 节：实验结果与训练稳定性

基于原论文《神经图灵机》（Neural Turing Machines，arxiv.org/abs/1410.5401）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NTMMemory(nn.Module):
    """
    外部存储矩阵（第 3 节）。
    大小为 N x M（存储位置数 x 向量维度）。
    """
    def __init__(self, N, M):
        super().__init__()
        self.N = N
        self.M = M
        
        # 第 3.2 节：存储初始化。
        # 初始存储使用缓冲区；原文误称其“可学习”，实际 register_buffer 不参与参数优化。
        # Graves 提到初始存储可以学习；此处实现未采用可学习参数。
        self.register_buffer('init_memory', torch.full((N, M), 1e-6))

    def reset(self, batch_size):
        """为新序列初始化存储。"""
        return self.init_memory.clone().repeat(batch_size, 1, 1)


class NTMHead(nn.Module):
    """
    读写头接口（第 3.1、3.2 节）。
    实现四步寻址流程（第 3.3.1 节），生成 w_t。
    """
    def __init__(self, memory, controller_size, is_read=True):
        super().__init__()
        self.N = memory.N
        self.M = memory.M
        self.is_read = is_read
        
        # 寻址参数（第 3.3.1 节）：
        # k（M）：键向量（公式 5）
        # beta（1）：键强度（公式 5）
        # g（1）：插值门（公式 7）
        # s（3）：循环卷积的移位权重（公式 8）
        # gamma（1）：锐化因子（公式 9）
        self.addr_size = self.M + 1 + 1 + 3 + 1
        
        # 写头还需要额外向量（第 3.3 节）：
        # e（M）：擦除向量（公式 3）
        # a（M）：添加向量（公式 4）
        if not is_read:
            self.addr_size += 2 * self.M
            
        self.fc = nn.Linear(controller_size, self.addr_size)
        
        # 初始权重（第 2 节）；原文称“可学习”，此处实际使用固定缓冲区
        # 通常初始关注第一个存储位置。
        self.register_buffer('init_w', torch.zeros(self.N))
        self.init_w[0] = 1.0 

    def reset(self, batch_size):
        return self.init_w.clone().repeat(batch_size, 1)

    def _addressing(self, controller_out, prev_w, memory):
        """
        计算寻址权重 w_t。
        对应论文公式 5 到 9。
        """
        # 0. 从控制器隐藏状态生成全部寻址参数
        out = self.fc(controller_out)
        
        ptr = 0
        k = out[:, ptr:ptr+self.M]; ptr += self.M
        # beta/gamma 应满足正值约束（第 3.3.1 节）；Softplus 保证 beta>0，1+Softplus 保证 gamma>=1。
        beta = F.softplus(out[:, ptr:ptr+1]); ptr += 1
        g = torch.sigmoid(out[:, ptr:ptr+1]); ptr += 1
        # 移位 s 是 [-1, 0, 1] 三个取值上的分布（公式 8）
        s = F.softmax(out[:, ptr:ptr+3], dim=1); ptr += 3
        gamma = 1 + F.softplus(out[:, ptr:ptr+1]); ptr += 1
        
        erase = None
        add = None
        if not self.is_read:
            erase = torch.sigmoid(out[:, ptr:ptr+self.M]); ptr += self.M
            add = out[:, ptr:ptr+self.M]

        # 1. 内容寻址（公式 5、6）
        # 根据内容本身的相似性确定关注位置（联想检索）。
        norm_k = k.norm(p=2, dim=1, keepdim=True) + 1e-8
        norm_mem = memory.norm(p=2, dim=2) + 1e-8
        sim = torch.matmul(memory, k.unsqueeze(2)).squeeze(2) / (norm_k * norm_mem)
        w_c = F.softmax(beta * sim, dim=1)

        # 2. 插值（公式 7）
        # 将新检索分布 w_c 与上次关注分布 prev_w 混合。
        # 使 NTM 可以“停留”或移动到新位置。
        w_g = g * w_c + (1 - g) * prev_w

        # 3. 卷积移位（公式 8）
        # 执行循环卷积，实现相对移动。
        # 用于遍历序列，例如复制任务。
        w_s = torch.zeros_like(w_g)
        for i in range(self.N):
            # 循环移位：s[-1]*w[i+1] + s[0]*w[i] + s[1]*w[i-1]；译注：这是原注释的符号示意，需核对下方索引约定。
            w_s[:, i] = (
                s[:, 0] * torch.roll(w_g, shifts=1, dims=1)[:, i] +
                s[:, 1] * w_g[:, i] +
                s[:, 2] * torch.roll(w_g, shifts=-1, dims=1)[:, i]
            )

        # 4. 锐化（公式 9）
        # 没有锐化时，每次移位都会使关注分布更加模糊。
        # 锐化使分布的峰值更突出，以维持集中关注。
        w_pow = w_s ** gamma
        w = w_pow / (w_pow.sum(dim=1, keepdim=True) + 1e-8)

        return w, erase, add

    def forward(self, controller_out, prev_w, memory_state):
        w, erase, add = self._addressing(controller_out, prev_w, memory_state)
        
        if self.is_read:
            # 读取操作（公式 2）：对存储行加权求和
            read_vec = torch.matmul(w.unsqueeze(1), memory_state).squeeze(1)
            return read_vec, w
        else:
            return w, erase, add


class NTM(nn.Module):
    """
    神经图灵机封装（第 2 节）。
    通过读写头将控制器（LSTM）与存储器连接。
    """
    def __init__(self, input_size, output_size, controller_size, N, M, num_reads=1, num_writes=1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.controller_size = controller_size
        self.N = N
        self.M = M
        
        self.memory = NTMMemory(N, M)
        
        # 第 2 节：控制器同时接收外部输入与上一时刻的读取向量。
        # 这构成包含存储状态的“思考”循环。
        self.controller = nn.LSTMCell(input_size + num_reads * M, controller_size)
        
        self.read_heads = nn.ModuleList([NTMHead(self.memory, controller_size, is_read=True) for _ in range(num_reads)])
        self.write_heads = nn.ModuleList([NTMHead(self.memory, controller_size, is_read=False) for _ in range(num_writes)])
        
        # 最终输出层投影
        self.fc_out = nn.Linear(controller_size + num_reads * M, output_size)
        
        # 控制器的初始隐藏状态；原注释称可学习，请以实际注册方式为准
        self.register_buffer('h_bias', torch.zeros(controller_size))
        self.register_buffer('c_bias', torch.zeros(controller_size))

    def reset(self, batch_size):
        """为新序列重置 NTM（第 2 节）。"""
        memory_state = self.memory.reset(batch_size)
        h = self.h_bias.clone().repeat(batch_size, 1)
        c = self.c_bias.clone().repeat(batch_size, 1)
        
        # 将读写头重置到初始权重分布（通常为位置 0）
        read_ws = [head.reset(batch_size) for head in self.read_heads]
        write_ws = [head.reset(batch_size) for head in self.write_heads]
        
        # 初始读取向量通常置零
        prev_reads = [torch.zeros(batch_size, self.M, device=memory_state.device) for _ in range(len(self.read_heads))]
        
        return memory_state, (h, c), read_ws, write_ws, prev_reads

    def forward(self, x, state):
        memory_state, (h, c), read_ws, write_ws, prev_reads = state
        
        # 1. 控制器前向计算（第 2 节）
        # 将当前输入与上一步读取的内容拼接。
        inp = torch.cat([x] + prev_reads, dim=1)
        h, c = self.controller(inp, (h, c))
        
        # 2. 读取步骤（公式 2）
        new_read_ws = []
        new_reads = []
        for i, head in enumerate(self.read_heads):
            r, w = head(h, read_ws[i], memory_state)
            new_reads.append(r)
            new_read_ws.append(w)
            
        # 3. 写入步骤（公式 3、4）
        new_write_ws = []
        for i, head in enumerate(self.write_heads):
            w, erase, add = head(h, write_ws[i], memory_state)
            new_write_ws.append(w)
            
            # 擦除步骤（公式 3）：移除旧信息
            erase_term = torch.matmul(w.unsqueeze(2), erase.unsqueeze(1))
            memory_state = memory_state * (1 - erase_term)
            
            # 添加步骤（公式 4）：加入新信息
            add_term = torch.matmul(w.unsqueeze(2), add.unsqueeze(1))
            memory_state = memory_state + add_term
            
        # 4. 产生输出（第 2 节）
        # 输出基于控制器状态和当前读到的信息。
        out_inp = torch.cat([h] + new_reads, dim=1)
        out = self.fc_out(out_inp)
        
        new_state = (memory_state, (h, c), new_read_ws, new_write_ws, new_reads)
        return out, new_state

if __name__ == "__main__":
    # 冒烟检查：初始化小型 NTM 并执行一步
    ntm = NTM(input_size=8, output_size=8, controller_size=100, N=128, M=20)
    print(f"NTM 初始化完成，参数数量为 {sum(p.numel() for p in ntm.parameters())}。")
    
    batch_size = 4
    state = ntm.reset(batch_size)
    x = torch.randn(batch_size, 8)
    
    output, _ = ntm(x, state)
    print(f"前向传播完成，输出形状：{output.shape}")
