"""
最小 LSTM 训练脚本
============================

在任意文本文件上训练字符级 LSTM。

用法：
    # 训练模型
    python train_minimal.py --data data/input.txt --epochs 200
    
    # 使用训练好的模型生成文本
    python train_minimal.py --generate --checkpoint lstm_model.pkl --length 500
    
    # 自定义超参数
    python train_minimal.py --data data/input.txt --hidden-size 128 --seq-length 50 --lr 0.001
"""

import numpy as np
import argparse
from implementation import LSTM


def load_data(filepath):
    """
    加载文本数据并创建字符映射。
    
    参数：
        filepath：文本文件路径
        
    返回：
        data：原始文本字符串
        chars：排序后的不重复字符列表
        char_to_idx：字符到索引的映射字典
        idx_to_char：索引到字符的映射字典
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        print(f"错误：文件 '{filepath}' 不存在！")
        print('请通过 --data 提供有效的文本文件')
        exit(1)
    
    # 获取不重复字符
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    
    print(f"数据加载成功！")
    print(f"  - 字符总数：{len(data):,}")
    print(f"  - 不重复字符数：{vocab_size}")
    print(f"  - 字符集合：{''.join(chars[:50])}" + ("..." if len(chars) > 50 else ""))
    
    # 创建映射
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    return data, chars, char_to_idx, idx_to_char


def train(args):
    """
    训练 LSTM 模型。
    
    参数：
        args：命令行参数
    """
    # 加载数据
    data, chars, char_to_idx, idx_to_char = load_data(args.data)
    vocab_size = len(chars)
    
    # 将文本转换为索引
    data_indices = [char_to_idx[ch] for ch in data]
    data_size = len(data_indices)
    
    # 初始化 LSTM
    print(f"\n正在初始化 LSTM……")
    print(f"  - 隐藏状态维度：{args.hidden_size}")
    print(f"  - 序列长度：{args.seq_length}")
    print(f"  - 学习率：{args.lr}")
    print(f"  - 参数量：约 {4 * args.hidden_size * (args.hidden_size + vocab_size) + vocab_size * args.hidden_size:,}")
    
    lstm = LSTM(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        seq_length=args.seq_length,
        learning_rate=args.lr
    )
    
    # 训练循环
    print(f"\n{'='*60}")
    print('开始训练……')
    print(f"{'='*60}\n")
    
    # 初始化状态
    h_prev = np.zeros((args.hidden_size, 1))
    C_prev = np.zeros((args.hidden_size, 1))
    
    # 用于记录损失
    smooth_loss = -np.log(1.0/vocab_size) * args.seq_length  # 初始损失估计
    losses = []
    
    # 数据指针
    p = 0
    
    # 计算总迭代次数
    sequences_per_epoch = data_size // args.seq_length
    total_iterations = args.epochs * sequences_per_epoch
    
    iteration = 0
    epoch = 0
    
    while epoch < args.epochs:
        # 在数据末尾或训练开始时重置
        if p + args.seq_length + 1 >= data_size or iteration == 0:
            h_prev = np.zeros((args.hidden_size, 1))
            C_prev = np.zeros((args.hidden_size, 1))
            p = 0
            if iteration > 0:
                epoch += 1
                print(f"\n--- 轮次 {epoch}/{args.epochs} 完成 ---")
        
        # 获取一批数据
        inputs = data_indices[p:p+args.seq_length]
        targets = data_indices[p+1:p+args.seq_length+1]
        
        # 数据不足时跳过
        if len(inputs) < args.seq_length:
            p = 0
            continue
        
        # 前向传播
        loss, h_prev, C_prev, cache = lstm.forward(inputs, targets, h_prev, C_prev)
        
        # 更新平滑损失（指数移动平均）
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        losses.append(smooth_loss)
        
        # 反向传播
        grads = lstm.backward(inputs, targets, cache)
        
        # 更新权重
        lstm.update_weights(grads)
        
        # 移动数据指针
        p += args.seq_length
        iteration += 1
        
        # 打印进度
        if iteration % 100 == 0:
            print(f"迭代 {iteration}/{total_iterations} | 轮次 {epoch+1}/{args.epochs} | 损失：{smooth_loss:.4f}")
            
            # 生成样本文本
            if iteration % 500 == 0:
                print("\n" + "-" * 60)
                print('生成样本：')
                sample_h = np.copy(h_prev)
                sample_C = np.copy(C_prev)
                sample_indices = lstm.sample(sample_h, sample_C, inputs[0], 200, temperature=0.8)
                sample_text = ''.join([idx_to_char[i] for i in sample_indices])
                print(sample_text)
                print("-" * 60 + "\n")
    
    # 训练完成
    print(f"\n{'='*60}")
    print('训练完成！')
    print(f"{'='*60}")
    print(f"最终损失：{smooth_loss:.4f}")
    print(f"迭代总数：{iteration}")
    
    # 保存模型
    lstm.save(args.checkpoint)
    print(f"\n模型已保存至：{args.checkpoint}")
    
    # 最终生成
    print(f"\n{'='*60}")
    print('最终生成样本（temperature=0.5）：')
    print(f"{'='*60}\n")
    
    h = np.zeros((args.hidden_size, 1))
    C = np.zeros((args.hidden_size, 1))
    sample_indices = lstm.sample(h, C, char_to_idx[chars[0]], 500, temperature=0.5)
    sample_text = ''.join([idx_to_char[i] for i in sample_indices])
    print(sample_text)
    
    print(f"\n{'='*60}")
    print('可以尝试不同温度：')
    print("  python train_minimal.py --generate --checkpoint lstm_model.pkl --temperature 0.3")
    print("  python train_minimal.py --generate --checkpoint lstm_model.pkl --temperature 1.0")
    print("  python train_minimal.py --generate --checkpoint lstm_model.pkl --temperature 1.5")


def generate(args):
    """
    使用训练好的模型生成文本。
    
    参数：
        args：命令行参数
    """
    # 加载模型
    print(f"正在从以下路径加载模型：{args.checkpoint}...")
    lstm = LSTM.load(args.checkpoint)
    
    # 需要字符映射；它们应随模型一起保存
    # 目前从数据文件重新生成映射
    if args.data is None:
        print('错误：生成时必须提供 --data，以获得字符映射')
        exit(1)
    
    _, chars, char_to_idx, idx_to_char = load_data(args.data)
    
    print(f"\n正在生成 {args.length} 个字符……")
    print(f"温度：{args.temperature}")
    print(f"起始字符：'{args.seed}'\n")
    print("=" * 60)
    
    # 初始化状态
    h = np.zeros((lstm.hidden_size, 1))
    C = np.zeros((lstm.hidden_size, 1))
    
    # 获取起始字符索引
    if args.seed in char_to_idx:
        seed_idx = char_to_idx[args.seed]
    else:
        print(f"提示：起始字符 '{args.seed}' 不在词表中，将使用词表首字符。")
        seed_idx = 0
    
    # 生成
    sample_indices = lstm.sample(h, C, seed_idx, args.length, temperature=args.temperature)
    sample_text = ''.join([idx_to_char[i] for i in sample_indices])
    
    print(sample_text)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='训练字符级 LSTM，或使用已有模型生成文本')
    
    # 模式
    parser.add_argument('--generate', action='store_true', 
                       help='生成模式（默认：训练）')
    
    # 数据
    parser.add_argument('--data', type=str, default='data/input.txt',
                       help='训练数据路径（默认：data/input.txt）')
    
    # 训练超参数
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='隐藏状态维度（默认：128）')
    parser.add_argument('--seq-length', type=int, default=50,
                       help='BPTT 序列长度（默认：50）')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率（默认：0.001）')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数（默认：50）')
    
    # 生成参数
    parser.add_argument('--checkpoint', type=str, default='lstm_model.pkl',
                       help='模型保存或加载路径（默认：lstm_model.pkl）')
    parser.add_argument('--length', type=int, default=500,
                       help='要生成的字符数（默认：500）')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='采样温度（默认：0.8）')
    parser.add_argument('--seed', type=str, default='T',
                       help='生成时的起始字符（默认：T）')
    
    args = parser.parse_args()
    
    # 打印标题
    print("\n" + "=" * 60)
    print('LSTM 字符级语言模型')
    print("=" * 60 + "\n")
    
    if args.generate:
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
