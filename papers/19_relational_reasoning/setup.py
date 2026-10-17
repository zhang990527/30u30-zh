"""
setup.py：第 19 天诊断与验证程序
按 30u30 第 19 天的格式组织

本脚本通过程序检查关系网络的
架构性质。

检查项目：
1. 排列不变性（标准检查）。
2. 基数敏感性：验证 sum 能区分集合大小变化，
   而 mean 会抹去此例中的大小信息。
3. 坐标保留：验证注入 (x, y)
   不会破坏预期张量形状。
"""

import sys
import torch
from implementation import RelationNetwork, add_coordinates

def verify_permutation_invariance() -> bool:
    print('正在验证关系归纳偏置（排列不变性）……')
    model = RelationNetwork(object_dim=8, output_dim=4)
    model.eval()
    
    objects = torch.randn(1, 10, 8)
    with torch.no_grad():
        out_orig = model(objects)
        
    indices = torch.randperm(10)
    shuffled_objects = objects[:, indices, :]
    with torch.no_grad():
        out_shuffled = model(shuffled_objects)
        
    diff = torch.abs(out_orig - out_shuffled).max().item()
    if diff < 1e-5:
        print(f"  [PASS] 打乱顺序后输出不变（差值：{diff:.2e})")
        return True
    return False

def verify_cardinality_bias() -> bool:
    """
    验证 sum 对对象数敏感，这对计数任务很重要，
    如第 2.1 节所述。
    """
    print('\n正在验证基数偏置（求和与均值）……')
    
    # 所有对象相同（全 1）
    objs_5 = torch.ones(1, 5, 8)
    objs_10 = torch.ones(1, 10, 8)
    
    # 1. 测试均值聚合器（此例中不随集合大小变化）
    model_mean = RelationNetwork(object_dim=8, aggregator='mean')
    model_mean.eval()
    with torch.no_grad():
        out_mean_5 = model_mean(objs_5)
        out_mean_10 = model_mean(objs_10)
    
    # 相同对象的均值不应随数量变化而明显改变
    mean_diff = torch.abs(out_mean_5 - out_mean_10).max().item()
    
    # 2. 测试求和聚合器（对集合大小敏感）
    model_sum = RelationNetwork(object_dim=8, aggregator='sum')
    model_sum.eval()
    with torch.no_grad():
        out_sum_5 = model_sum(objs_5)
        out_sum_10 = model_sum(objs_10)
        
    sum_diff = torch.abs(out_sum_5 - out_sum_10).max().item()
    
    print(f"  [INFO] 均值聚合器差值（N=5 与 N=10）：{mean_diff:.2e}")
    print(f"  [INFO] 求和聚合器差值（N=5 与 N=10）：{sum_diff:.2e}")
    
    if sum_diff > mean_diff * 10:
        print('  [PASS] 求和聚合器成功保留了集合大小信息。')
        return True
    return False

def verify_coordinates() -> bool:
    print('\n正在验证坐标注入……')
    objs = torch.randn(2, 5, 8)
    objs_coords = add_coordinates(objs)
    
    if objs_coords.shape == (2, 5, 10):
        print('  [PASS] 已成功拼接 (x, y) 坐标。')
        return True
    return False

if __name__ == "__main__":
    print("=" * 50)
    print('第 19 天：关系推理')
    print('诊断与验证程序')
    print("=" * 50)
    print()
    
    success = True
    success &= verify_permutation_invariance()
    success &= verify_cardinality_bias()
    success &= verify_coordinates()
    
    print("\n" + "-" * 50)
    if success:
        print('  [LOGICAL SMOKE TEST] 通过')
        print('  此次样例检查支持以下架构性质：')
        print('  1. 排列不变性（不依赖顺序）')
        print('  2. 基数感知（求和与均值的偏置影响计数）')
        print('  3. 空间信息就绪（坐标保留检查通过）')
        print("-" * 50)
    else:
        print('  [LOGICAL SMOKE TEST] 失败')
        print("-" * 50)
        sys.exit(1)
