import torch
import torch.nn as nn
import torch.optim as optim
from implementation import RelationalRNN
from exercise_04_nth_farthest import generate_nth_farthest_batch

def train():
    # 超参数
    seq_len = 10
    dim = 16
    slots = 4
    mem_size = 32
    heads = 4
    epochs = 500
    batch_size = 32
    
    # TODO：初始化 RelationalRNN 模型
    # 使用上面的超参数；输出大小 == dim（回归）
    model = # 在此填写代码
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print('开始训练……')
    
    for step in range(epochs):
        # TODO：生成一批数据
        inputs, targets = # 在此填写代码
        
        optimizer.zero_grad()
        
        # TODO：前向传播
        output = # 在此填写代码
        
        # TODO：计算损失并反向传播
        loss = # 在此填写代码
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"步数 {step}：损失 = {loss.item():.6f}")

    print('\n训练完成！')
    
    # 验证
    model.eval()
    with torch.no_grad():
        inputs, targets = generate_nth_farthest_batch(1, seq_len, dim)
        output = model(inputs)
        print(f"最终损失：{criterion(output, targets).item():.6f}")

if __name__ == "__main__":
    train()
