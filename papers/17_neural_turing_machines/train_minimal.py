"""
NTM 训练脚本：复制任务
==================================

本脚本实现 Graves 等人（2014）第 4.1 节定义的“复制任务”，
用作主要基准，检验 NTM
是否学会了简单的算法过程。

任务：
1. 输入随机 8 比特向量序列。
2. 输入一个独特的分隔符。
3. 观察 NTM 是否按原顺序输出完全相同的序列。

标准 LSTM 尝试把全部历史压缩到固定大小向量，
因而在长序列上常难以完成此任务。NTM 则将存储器
当作纸带使用，以解决这一问题（第 4.1 节）。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
from implementation import NTM

def generate_copy_instance(batch_size, seq_len, width):
    """
    为单次训练步骤生成数据。
    
    格式（第 4.1 节）：
    - 输入阶段：[数据比特] 后跟 [第 Width+1 个分量为 1，Python 索引为 width]
    - 输出阶段：输入为零，模型必须输出 [数据比特]。
    """
    # 创建随机比特向量 {0, 1}^width
    data = torch.randn(batch_size, seq_len, width).sign().add(1).div(2)
    
    # 输入序列：[data] + [delimiter] + [输出阶段的零输入]
    # 总长度：L（输入）+ 1（分隔符）+ L（输出）
    full_len = 2 * seq_len + 1
    inputs = torch.zeros(batch_size, full_len, width + 1)
    targets = torch.zeros(batch_size, full_len, width)
    
    inputs[:, :seq_len, :width] = data
    inputs[:, seq_len, width] = 1.0 # 分隔比特
    
    # 目标是后续 L 个时间步中的数据
    targets[:, seq_len+1:, :] = data
    
    return inputs.transpose(0, 1), targets.transpose(0, 1)

def train():
    # 实验超参数（第 4 节）
    batch_size = 32
    max_seq_len = 20 # 可增大以测试泛化（第 4.1 节）
    width = 8
    controller_size = 100
    N, M = 128, 20
    
    model = NTM(width + 1, width, controller_size, N, M)
    
    # 第 4 节：Graves 建议使用 RMSProp 提高 NTM 稳定性。
    # 学习率 1e-4 是常见的稳定起点。
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4, momentum=0.9, alpha=0.95)
    
    # 二元交叉熵适合比特复制任务。
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"开始复制任务训练（N={N}, M={M}, 控制器={controller_size})...")
    
    for i in range(10001):
        optimizer.zero_grad()
        
        # 改变序列长度，使模型学习复制“规则”，
        # 而非固定长度的模式（第 4.1 节）。
        cur_len = random.randint(1, max_seq_len)
        inputs, targets = generate_copy_instance(batch_size, cur_len, width)
        
        state = model.reset(batch_size)
        out_list = []
        
        # 顺序处理完整任务序列
        for t in range(inputs.size(0)):
            out, state = model(inputs[t], state)
            out_list.append(out)
            
        out_seq = torch.stack(out_list)
        
        # 仅在回忆阶段计算损失
        loss = criterion(out_seq[cur_len+1:], targets[cur_len+1:])
        loss.backward()
        
        # 稳定性的关键（第 4 节）：
        # 梯度裁剪有助于避免循环移位中的梯度爆炸。
        # 原仓库提示：否则模型可能在 100 次迭代内产生 NaN；并非确定性保证。
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        
        optimizer.step()
        
        if i % 100 == 0:
            with torch.no_grad():
                # 比特错误率（BER）：错误比特所占百分比
                preds = (out_seq[cur_len+1:] > 0).float()
                error = torch.abs(preds - targets[cur_len+1:]).mean().item()
            print(f"迭代 {i:5d} | 长度 {cur_len:2d} | 损失 {loss.item():.4f} | BER {error*100:.2f}%")

        if i % 1000 == 0 and i > 0:
            # 收敛检查：通常以 BER < 1% 作为“解决”的标准。
            pass

    # 保存权重以供可视化
    torch.save(model.state_dict(), "papers/17_neural_turing_machines/ntm_copy_final.pth")
    print('训练完成。权重已保存至 papers/17_neural_turing_machines/ntm_copy_final.pth')

if __name__ == "__main__":
    train()
