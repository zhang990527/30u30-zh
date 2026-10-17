import torch
from implementation import add_coordinates


def test_spatial_logic():
    """
    解答 5：验证练习 test_spatial_logic()
    所要求的坐标注入工具函数。

    检查项：
    - 返回形状为 (B, N, D+2)
    - 第一个对象的 x 坐标为 -1.0（归一化网格）
    """
    batch, N, D = 1, 4, 8
    objs = torch.randn(batch, N, D)

    objs_with_coords = add_coordinates(objs)

    # 维度检查
    assert objs_with_coords.shape == (batch, N, D + 2), (
        f"预期形状 {(batch, N, D+2)}，实际得到 {objs_with_coords.shape}"
    )

    # x 坐标由 linspace(-1, 1, N) 创建，第一个应为 -1.0
    first_x = objs_with_coords[0, 0, -2].item()
    assert abs(first_x + 1.0) < 1e-6, f"预期首个 x 坐标约为 -1.0，实际得到 {first_x}"

    print('解答 5：[PASS]')
