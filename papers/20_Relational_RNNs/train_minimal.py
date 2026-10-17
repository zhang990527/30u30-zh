"""
第 20 天：关系 RNN 最小训练脚本
=============================================================================

关系记忆核心（RMC）的完整可运行演示。
运行以比较关系任务中 RMC 与 LSTM 的表现；不保证 RMC 始终更优。

用法：
    python train_minimal.py                             # 同时训练 RMC 和 LSTM
    python train_minimal.py --model rmc --visualize     # 训练 RMC 并显示注意力
    python train_minimal.py --seq_len 50                # 尝试更难的长序列

本脚本的功能：
1. 生成本章“第 N 远”数据（几何推理教学任务）
2. 分别训练 RMC 与 LSTM 基线以作比较
3. 可视化 RMC 注意力权重；并非对“思考过程”的完整解释
4. 保存对比图

核心思路：RMC 显式使用注意力关联过去信息，
而 LSTM 通过门控循环状态隐式处理；原文称其梯度会消失，这并非所有情况下必然发生。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
from implementation import RelationalRNN, StandardLSTM
from visualization import plot_attention_heatmap, plot_comparison

def generate_nth_farthest_batch(batch_size, seq_len, vector_dim, n_index=0):
    """
    为本章距离任务生成一个批次。
    """
    inputs = torch.rand(batch_size, seq_len, vector_dim) * 2 - 1
    targets = []
    
    for i in range(batch_size):
        seq = inputs[i]
        ref_vec = seq[n_index]
        dists = torch.norm(seq - ref_vec, dim=1)
        farthest_idx = torch.argmax(dists)
        target_vec = seq[farthest_idx]
        targets.append(target_vec)
        
    targets = torch.stack(targets)
    return inputs, targets

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model_name, model, args, inputs_targets_gen):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    losses = []
    
    print(f"\n正在训练 {model_name}...")
    print(f"参数数量：{count_parameters(model):,}")
    
    model.train()
    for step in range(args.epochs):
        inputs, targets = inputs_targets_gen(args.batch_size, args.seq_len, args.dim)
        
        optimizer.zero_grad()
        if model_name == "RMC":
            output = model(inputs) # RMC 的默认返回值
        else:
            output = model(inputs) # LSTM 的默认返回值
            
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 100 == 0:
            print(f"[{model_name}] 步数 {step}：损失 = {loss.item():.6f}")
            
    return losses

def main():
    parser = argparse.ArgumentParser(description='第 20 天：RMC 与 LSTM 比较')
    parser.add_argument('--seq_len', type=int, default=15, help='序列长度')
    parser.add_argument('--dim', type=int, default=16, help='向量维度')
    parser.add_argument('--slots', type=int, default=4, help='RMC 记忆槽位数')
    parser.add_argument('--mem_size', type=int, default=32, help='RMC 每个记忆槽位的大小')
    parser.add_argument('--heads', type=int, default=4, help='RMC 注意力头数')
    parser.add_argument('--lstm_hidden', type=int, default=64, help='LSTM 隐藏状态大小')
    parser.add_argument('--epochs', type=int, default=2000, help='训练步数')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model', type=str, choices=['rmc', 'lstm', 'both'], default='both')
    parser.add_argument('--visualize', action='store_true', help='可视化注意力（仅 RMC）')
    parser.add_argument('--save_plot', action='store_true', help='保存损失比较图')
    parser.add_argument('--output_dir', type=str, default='.')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 任务生成器
    gen_data = generate_nth_farthest_batch
    
    results = {}
    
    # 1. 训练 RMC
    if args.model in ['rmc', 'both']:
        rmc = RelationalRNN(args.dim, args.slots, args.mem_size, args.dim, args.heads)
        rmc_losses = train_model("RMC", rmc, args, gen_data)
        results['rmc'] = (rmc, rmc_losses)
        
        # 验证
        rmc.eval()
        inputs, targets = gen_data(100, args.seq_len, args.dim)
        with torch.no_grad():
            out = rmc(inputs)
            val_loss = nn.MSELoss()(out, targets).item()
        print(f"RMC 最终验证损失：{val_loss:.6f}")

    # 2. 训练 LSTM
    if args.model in ['lstm', 'both']:
        lstm = StandardLSTM(args.dim, args.lstm_hidden, args.dim)
        lstm_losses = train_model("LSTM", lstm, args, gen_data)
        results['lstm'] = (lstm, lstm_losses)
        
        # 验证
        lstm.eval()
        inputs, targets = gen_data(100, args.seq_len, args.dim)
        with torch.no_grad():
            out = lstm(inputs)
            val_loss = nn.MSELoss()(out, targets).item()
        print(f"LSTM 最终验证损失：{val_loss:.6f}")

    # 3. 比较与绘图
    if args.model == 'both' and args.save_plot:
        plot_path = os.path.join(args.output_dir, "loss_comparison.png")
        plot_comparison(results['rmc'][1], results['lstm'][1], plot_path)
        
    # 4. 可视化注意力（仅 RMC）
    if args.visualize and 'rmc' in results:
        print('\n正在可视化注意力……')
        rmc = results['rmc'][0]
        rmc.eval()
        inputs, _ = gen_data(1, args.seq_len, args.dim)
        
        # 前向传播并返回注意力
        _, attn_weights = rmc(inputs, return_attention=True)
        # attn_weights: (batch, seq, heads, slots, slots)
        
        # 取批次中第一项：(seq, heads, slots, slots)
        attn_sample = attn_weights[0]
        
        plot_path = os.path.join(args.output_dir, "attention_heatmap.png")
        plot_attention_heatmap(attn_sample, plot_path)

if __name__ == "__main__":
    main()
