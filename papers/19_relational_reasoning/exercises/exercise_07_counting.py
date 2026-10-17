import torch
from implementation import RelationNetwork

def compare_aggregators():
    """
    TODO：通过样例检查 sum 比 mean 更适合保留计数信息。
    第 2.1 节原仓库引文：“选择求和 [...] 使模型可以计数。”
    
    1. 创建包含 2 个全 1 向量的基准集合。
    2. 创建包含 8 个全 1 向量的较大集合。
    3. 比较 aggregator='mean' 的 RelationNetwork 输出幅值。
    4. 比较 aggregator='sum' 的 RelationNetwork 输出幅值。
    5. 展示此例中 mean 不随集合大小变化，而 sum 保留计数信息。
    """
    # 在此填写代码
    pass

if __name__ == "__main__":
    print("-" * 50)
    print('练习 7：基数偏置（求和与均值）')
    print("-" * 50)
    compare_aggregators()
    print('练习 7：[COMPLETE]')
