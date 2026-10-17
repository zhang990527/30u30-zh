"""复杂动力学：咖啡混合模拟与复杂性测量

实现 Aaronson 2011 年博文《复杂动力学第一定律》（The First Law of Complexodynamics）中的核心思路：
1. 咖啡—牛奶混合模拟（随机邻居交换的二维像素网格）
2. 用 gzip 近似柯尔莫哥洛夫复杂度（KC）
3. 粗粒化 KC（Sean Carroll 在博文评论中的建议）
4. 两段式编码的精致复杂度（sophistication）代理量
5. 多尺度复杂性分析（Luca Trevisan 的建议）
6. 跟踪熵以作比较

原仓库希望展示：熵单调增加，而复杂性（gzip 大小、粗粒化 KC 或精致复杂度代理量）在中间时刻达到峰值后下降，即 Aaronson 的“第一定律”猜想所描述的“隆起”。
核验说明：gzip 原始大小未必出现隆起；全局像素熵在本模拟中守恒；平均局部熵也不保证每一步单调。代码是教学代理量，不是复杂熵的精确计算或猜想证明。

参考：https://scottaaronson.blog/?p=762
作者：30u30 项目"""

import numpy as np
import gzip
from typing import Tuple, Dict, List, Optional
import warnings


# ============================================================================
# 第 1 节：咖啡—牛奶网格模拟
# ============================================================================
# 本章将 Aaronson 的模型描述为黑色（咖啡）和白色（牛奶）像素的二维数组。
# 初始时两者分离（上半黑、下半白）；每一步
# 随机选择一对相邻咖啡—牛奶像素并交换。这是离散
# 随机扩散过程。
# ============================================================================

def create_initial_grid(size: int = 64) -> np.ndarray:
    """创建初始咖啡—牛奶网格：上半部分 = 1（咖啡/黑），下半部分 = 0（牛奶/白）。

参数：
    size：方形网格的边长。
返回：
    形状为 (size, size)、取值为 0 或 1 的二维 NumPy 数组。"""
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[:size // 2, :] = 1  # 上半部分为咖啡
    return grid


def find_adjacent_pairs(grid: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """找出所有一格为咖啡（1）、另一格为牛奶（0）的相邻像素对。
仅考虑水平与垂直邻居（四连通）。
这是本章所述模型中“可交换”像素对的集合。
参数：
    grid：二维二值数组。
返回：
    ((r1, c1), (r2, c2)) 像素对列表。"""
    rows, cols = grid.shape
    pairs = []

    # 水平邻居
    for r in range(rows):
        for c in range(cols - 1):
            if grid[r, c] != grid[r, c + 1]:
                pairs.append(((r, c), (r, c + 1)))

    # 垂直邻居
    for r in range(rows - 1):
        for c in range(cols):
            if grid[r, c] != grid[r + 1, c]:
                pairs.append(((r, c), (r + 1, c)))

    return pairs


def swap_step(grid: np.ndarray) -> np.ndarray:
    """执行一次交换：随机选取一对相邻咖啡—牛奶像素并交换。

这是本章咖啡模型的基本操作。每个时间步从所有异色相邻像素对中均匀选择一对交换。
参数：
    grid：二维二值数组（原地修改）。
返回：
    网格（同一对象，已原地修改）。"""
    pairs = find_adjacent_pairs(grid)
    if len(pairs) == 0:
        return grid  # 无异色相邻对（原注释称完全混合或分离；实际均匀单色网格才会如此）

    idx = np.random.randint(len(pairs))
    (r1, c1), (r2, c2) = pairs[idx]
    grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]
    return grid


def batch_swap(grid: np.ndarray, n_swaps: int = 100) -> np.ndarray:
    """为提高效率，批量尝试多次交换。

大网格每“步”只交换一次很慢。这里批量尝试 n_swaps 次随机交换，使用快速随机方法选相邻像素对，不在每次枚举全部像素对。

原说明称这是精确模型的近似，批内像素对可能重叠，大网格影响可忽略。
核验说明：本实现随机选择像素与方向并可能跳过，而非每次均匀选择异色边；时间参数化和边界行为均有差异，“影响可忽略”未作验证。
参数：
    grid：二维二值数组（原地修改）。
    n_swaps：每批尝试交换的次数。
返回：
    网格（同一对象，已原地修改）。"""
    rows, cols = grid.shape

    for _ in range(n_swaps):
        # 随机选择一个像素
        r = np.random.randint(rows)
        c = np.random.randint(cols)

        # 随机选择一个邻居方向（上、下、左、右）
        direction = np.random.randint(4)
        if direction == 0 and r > 0:
            nr, nc = r - 1, c
        elif direction == 1 and r < rows - 1:
            nr, nc = r + 1, c
        elif direction == 2 and c > 0:
            nr, nc = r, c - 1
        elif direction == 3 and c < cols - 1:
            nr, nc = r, c + 1
        else:
            continue  # 越过边界，跳过

        # 仅当两格不同（一格咖啡、一格牛奶）时交换
        if grid[r, c] != grid[nr, nc]:
            grid[r, c], grid[nr, nc] = grid[nr, nc], grid[r, c]

    return grid


def run_simulation(
    grid_size: int = 64,
    n_steps: int = 50000,
    swaps_per_step: int = 50,
    measure_interval: int = 500,
    seed: Optional[int] = 42
) -> Dict:
    """运行完整咖啡混合模拟，按间隔测量复杂性。

主实验创建一半咖啡、一半牛奶的网格，运行随机交换，定期测量：
- 香农熵（原说明预期单调增加；这里的全局像素熵实际守恒）
- gzip 复杂度（原说明预期“隆起”；不保证出现）
- 各尺度的粗粒化 KC
- 两段式编码的精致复杂度代理量
参数：
    grid_size：网格边长。
    n_steps：批量交换总步数。
    swaps_per_step：每步尝试的单次交换数。
    measure_interval：每隔多少步测量一次复杂性。
    seed：用于复现的随机种子。
返回：
    字典，包含：
        'times'：测量时刻列表
        'grids'：网格快照列表
        'entropy'：熵值列表
        'gzip_complexity'：gzip 压缩大小列表
        'coarse_kc'：{block_size: 粗粒化 KC 值列表} 字典
        'two_part'：(part1, part2) 元组列表
        'fraction_mixed'：相邻异色像素对比例列表"""
    if seed is not None:
        np.random.seed(seed)

    grid = create_initial_grid(grid_size)

    # 存储数据
    times = []
    grids = []
    entropy_vals = []
    gzip_vals = []
    coarse_kc_vals = {bs: [] for bs in [2, 4, 8]}
    two_part_vals = []
    fraction_mixed_vals = []

    # 测量函数
    def measure(step):
        times.append(step * swaps_per_step)
        grids.append(grid.copy())
        entropy_vals.append(grid_entropy(grid))
        gzip_vals.append(gzip_complexity(grid))
        for bs in coarse_kc_vals:
            if grid_size // bs >= 4:  # 至少需要 4x4 的粗网格
                coarse_kc_vals[bs].append(coarse_grained_kc(grid, bs))
            else:
                coarse_kc_vals[bs].append(0)
        two_part_vals.append(two_part_code(grid, block_size=4))
        fraction_mixed_vals.append(boundary_fraction(grid))

    # 初始测量
    measure(0)

    # 运行模拟
    for step in range(1, n_steps + 1):
        batch_swap(grid, n_swaps=swaps_per_step)
        if step % measure_interval == 0:
            measure(step)

    return {
        'times': times,
        'grids': grids,
        'entropy': entropy_vals,
        'gzip_complexity': gzip_vals,
        'coarse_kc': coarse_kc_vals,
        'two_part': two_part_vals,
        'fraction_mixed': fraction_mixed_vals,
        'grid_size': grid_size,
        'n_steps': n_steps,
        'swaps_per_step': swaps_per_step,
    }


# ============================================================================
# 第 2 节：用 gzip 近似柯尔莫哥洛夫复杂度
# ============================================================================
# KC 不可计算，但 gzip 加固定解压开销可给出描述长度上界。Aaronson
# 讨论过这一思路，Lauren Ouellette 曾在实验中使用
# （后来发展为第 7 天的论文）。
#
# 对二维网格，将像素序列化为字节再压缩。压缩后的
# 大小反映网格中可被压缩器识别的结构：
# - 单一颜色：高度可压缩（KC 低）
# - 随机噪声：难以压缩（KC 高）
# - 有结构的卷须：居中（中等 KC，但预期精致复杂度高）
# ============================================================================

def gzip_complexity(grid: np.ndarray) -> int:
    """使用 gzip 压缩大小近似网格的柯尔莫哥洛夫复杂度。
参数：
    grid：二维二值 NumPy 数组。
返回：
    压缩后的字节数。"""
    data = grid.astype(np.uint8).tobytes()
    return len(gzip.compress(data, compresslevel=9))


def gzip_complexity_bytes(data: bytes) -> int:
    """原始字节的 gzip 复杂度。
参数：
    data：原始字节串。
返回：
    压缩后的字节数。"""
    return len(gzip.compress(data, compresslevel=9))


def robust_gzip_complexity(grid: np.ndarray) -> float:
    """较稳健的 KC 估计：对行优先和列优先序列化的 gzip 大小求平均。

gzip 使用 LZ77（滑动窗口），对序列化顺序敏感。对不同顺序求平均有助于减轻这一伪影。
参数：
    grid：二维二值 NumPy 数组。
返回：
    平均压缩大小，单位为字节。"""
    # 行优先（默认）
    data_row = grid.astype(np.uint8).tobytes()
    size_row = len(gzip.compress(data_row, compresslevel=9))

    # 列优先
    data_col = grid.T.astype(np.uint8).tobytes()
    size_col = len(gzip.compress(data_col, compresslevel=9))

    return (size_row + size_col) / 2.0


# ============================================================================
# 第 3 节：粗粒化 KC（Sean Carroll 的建议）
# ============================================================================
# Carroll 在博文评论（#6-7）建议：在某个尺度模糊或粗粒化位图，
# 再测量粗粒化版本的 KC。这将
# 捕捉宏观结构，而非像素级噪声。
#
# 思路：在合适粗粒化尺度，卷须边界
# 形成难以压缩的结构。尺度过细时，噪声主导；
# 尺度过粗时，所有结构都会被抹去。
# ============================================================================

def coarse_grain(grid: np.ndarray, block_size: int) -> np.ndarray:
    """对 block_size x block_size 块求平均，以粗粒化网格。
参数：
    grid：二维数组。
    block_size：求平均的块大小。
返回：
    粗粒化二维数组（小于输入）。"""
    h, w = grid.shape
    h_new = h // block_size
    w_new = w // block_size

    result = np.zeros((h_new, w_new), dtype=np.float64)
    for i in range(h_new):
        for j in range(w_new):
            block = grid[i * block_size:(i + 1) * block_size,
                        j * block_size:(j + 1) * block_size]
            result[i, j] = block.mean()

    return result


def coarse_grained_kc(grid: np.ndarray, block_size: int) -> int:
    """网格粗粒化版本的 KC 近似。
步骤：
1. 对 block_size x block_size 块求平均
2. 量化为 uint8（0-255）
3. 用 gzip 压缩
参数：
    grid：二维二值数组。
    block_size：粗粒化尺度。
返回：
    粗粒化表示的压缩大小。"""
    coarse = coarse_grain(grid, block_size)
    # 量化为 256 个等级
    quantized = (coarse * 255).astype(np.uint8)
    return gzip_complexity_bytes(quantized.tobytes())


def multiscale_kc(grid: np.ndarray, scales: Optional[List[int]] = None) -> Dict[int, int]:
    """计算多个粗粒化尺度的 KC（Trevisan 的想法）。

Trevisan（博文评论 #17）建议随时间跟踪各尺度 KC，得到复杂性所在位置的二维图景（尺度 x 时间）。
参数：
    grid：二维二值数组。
    scales：块大小列表，默认采用 2 的幂。
返回：
    尺度到压缩大小的映射字典。"""
    grid_size = min(grid.shape)
    if scales is None:
        scales = [2**k for k in range(1, int(np.log2(grid_size)))]

    result = {}
    for s in scales:
        if grid_size // s >= 2:  # 至少需要 2x2 的粗网格
            result[s] = coarse_grained_kc(grid, s)
    return result


# ============================================================================
# 第 4 节：两段式编码（精致复杂度代理量）
# ============================================================================
# 精致复杂度（Koppel，1988；Gács–Tromp–Vitányi）衡量
# 两段式编码中“模型”部分的复杂度：
#
#   第一部分：包含 x 的集合 S 的描述（“模型”）
#   第二部分：x 在 S 中的索引（“给定模型后的数据”）
#
# 精致复杂度 = K(S)，即模型的复杂度（非正式概括）。
#
# 使用如下方法近似：
#   第一部分 = 网格的粗粒化版本（结构模型）
#   第二部分 = 残差（精确网格减去粗粒度重建）
#
# 第一部分大小被预期呈现特征性隆起：t=0 时低（模型
# 简单），中间时刻高（卷须边界复杂），
# 平衡时低（简单均匀模型）；具体代理量需实测。
# ============================================================================

def two_part_code(
    grid: np.ndarray,
    block_size: int = 4
) -> Tuple[int, int]:
    """计算网格的两段式编码大小。

第一部分（模型/精致复杂度）：粗粒度描述。
第二部分（给定模型后的数据）：重建精确网格的残差。
核验说明：残差被量化，因此本实现并未提供经过验证的无损编码和解码器。
参数：
    grid：二维二值数组。
    block_size：模型的粗粒化尺度。
返回：
    (part1_size, part2_size)，单位为字节。"""
    h, w = grid.shape
    grid_float = grid.astype(np.float64)

    # 第一部分：粗粒度模型
    coarse = coarse_grain(grid_float, block_size)
    model_quantized = (coarse * 255).astype(np.uint8)
    part1 = gzip_complexity_bytes(model_quantized.tobytes())

    # 从粗粒度模型重建
    reconstructed = np.repeat(
        np.repeat(coarse, block_size, axis=0),
        block_size, axis=1
    )[:h, :w]

    # 第二部分：残差
    # 将残差缩放到 [0, 255] 范围后压缩
    residual = grid_float - reconstructed
    # 残差位于 [-1, 1]，映射到 [0, 255]
    residual_uint8 = ((residual + 1.0) * 127.5).astype(np.uint8)
    part2 = gzip_complexity_bytes(residual_uint8.tobytes())

    return part1, part2


def sophistication_proxy(grid: np.ndarray, block_size: int = 4) -> int:
    """精致复杂度代理量 = 两段式编码的第一部分。
参数：
    grid：二维二值数组。
    block_size：粗粒化尺度。
返回：
    第一部分大小，单位为字节（“模型”复杂度）。"""
    part1, _ = two_part_code(grid, block_size)
    return part1


# ============================================================================
# 第 5 节：熵与混合度量
# ============================================================================
# 像素取值分布的香农熵。对二值网格，其表达式为
# H(p) = -p*log(p) - (1-p)*log(1-p)，p 是 1 的比例。
#
# 在咖啡模拟中，全局熵不能跟踪混合程度。它初始为
# 1.0（一半黑、一半白，二值分布的最大熵），并始终
# 接近 1.0（交换守恒 1 的比例）。
#
# 更有用的是局部熵：在滑动窗口中计算熵，
# 跟踪各邻域混合的均匀程度。
# ============================================================================

def grid_entropy(grid: np.ndarray) -> float:
    """整体像素取值分布的香农熵。

对二值网格，H = -p*log2(p) - (1-p)*log2(1-p)。
注意：咖啡模拟中它始终接近 1.0，因为交换守恒咖啡/牛奶像素总数；偶数边长的初始半半网格中它恒为 1.0。
参数：
    grid：二维二值数组。
返回：
    香农熵，单位为比特。"""
    p = grid.mean()
    if p == 0.0 or p == 1.0:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def local_entropy(grid: np.ndarray, window_size: int = 8) -> np.ndarray:
    """用滑动窗口计算网格的局部熵。

用于跟踪混合时，比全局熵提供更多信息。
初期：部分窗口全 0 或全 1（各自熵为 0），跨边界窗口除外。
平衡时：窗口内约为 50/50（熵接近 1.0）。
原说明称平均局部熵随混合单调增加；核验说明：它可以波动，不保证每步单调。
参数：
    grid：二维二值数组。
    window_size：滑动窗口边长。
返回：
    局部熵值的二维数组。"""
    h, w = grid.shape
    h_out = h - window_size + 1
    w_out = w - window_size + 1
    result = np.zeros((h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            window = grid[i:i + window_size, j:j + window_size]
            p = window.mean()
            if p == 0.0 or p == 1.0:
                result[i, j] = 0.0
            else:
                result[i, j] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    return result


def mean_local_entropy(grid: np.ndarray, window_size: int = 8) -> float:
    """全网格的平均局部熵。用于跟踪混合时通常比全局熵更有用，但不保证逐步单调。
参数：
    grid：二维二值数组。
    window_size：滑动窗口边长。
返回：
    平均局部熵。"""
    le = local_entropy(grid, window_size)
    return le.mean()


def boundary_fraction(grid: np.ndarray) -> float:
    """异色相邻像素对（一格咖啡、一格牛奶）所占比例。

它跟踪咖啡区和牛奶区的“界面长度”。原说明称初期低（单一水平边界），混合时达到峰值（类分形卷须），均匀时降低。
核验说明：微观二值随机网格仍有大量异色边，边界比例未必在后期下降。

注意：它与复杂性有关，但不同；棋盘格边界比例最大，描述却简单。
参数：
    grid：二维二值数组。
返回：
    边界像素对比例（0 到 1）。"""
    h, w = grid.shape
    total_pairs = 0
    diff_pairs = 0

    # 水平像素对
    h_diff = np.sum(grid[:, :-1] != grid[:, 1:])
    h_total = h * (w - 1)

    # 垂直像素对
    v_diff = np.sum(grid[:-1, :] != grid[1:, :])
    v_total = (h - 1) * w

    return (h_diff + v_diff) / (h_total + v_total)


# ============================================================================
# 第 6 节：归一化复杂性度量
# ============================================================================
# 为便于比较，将复杂性度量归一化到 [0, 1]。
# 定义：
#   normalized_kc = (gzip_size - min_gzip_size) / (max_gzip_size - min_gzip_size)
# 其中 min 为全零网格的 gzip 大小，max 为随机网格的参考大小。
# ============================================================================

def compute_normalization_bounds(grid_size: int, n_samples: int = 5) -> Dict:
    """计算用于归一化的 gzip 大小参考界。
参数：
    grid_size：网格边长。
    n_samples：用于平均的随机网格数。
返回：
    包含 'min_gzip'（全零）和 'max_gzip'（随机平均值）的字典。"""
    # 参考最小值：全零网格
    zeros = np.zeros((grid_size, grid_size), dtype=np.uint8)
    min_gzip = gzip_complexity(zeros)

    # 参考最大值：对随机网格求平均
    max_gzips = []
    for _ in range(n_samples):
        rand_grid = np.random.randint(0, 2, size=(grid_size, grid_size), dtype=np.uint8)
        max_gzips.append(gzip_complexity(rand_grid))
    max_gzip = np.mean(max_gzips)

    return {'min_gzip': min_gzip, 'max_gzip': max_gzip}


def normalize_complexity(
    values: List[float],
    bounds: Dict
) -> List[float]:
    """用预计算的参考界将复杂性值归一化到 [0, 1]。
参数：
    values：原始复杂性值。
    bounds：包含 'min_gzip' 和 'max_gzip' 的字典。
返回：
    归一化后的值。"""
    lo = bounds['min_gzip']
    hi = bounds['max_gzip']
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


# ============================================================================
# 第 7 节：实用函数
# ============================================================================

def grid_to_image(grid: np.ndarray) -> np.ndarray:
    """将二值网格转换为灰度图像（语义约定：0 = 白、1 = 黑）。
参数：
    grid：二维二值数组。
返回：
    适合 matplotlib imshow 的二维数组（0.0=白、1.0=黑是本函数的语义说明；实际颜色还取决于色图）。"""
    return grid.astype(np.float64)


def save_grid(grid: np.ndarray, filepath: str) -> None:
    """将网格保存为 NumPy 二进制文件。
参数：
    grid：二维二值数组。
    filepath：输出路径。"""
    np.save(filepath, grid)


def load_grid(filepath: str) -> np.ndarray:
    """从 NumPy 二进制文件加载网格。
参数：
    filepath：输入路径。
返回：
    二维二值数组。"""
    return np.load(filepath)


def summarize_results(results: Dict) -> str:
    """输出模拟结果摘要。
参数：
    results：run_simulation() 返回的字典。
返回：
    摘要字符串。"""
    times = results['times']
    gzip_vals = results['gzip_complexity']
    entropy_vals = results['entropy']

    # 寻找复杂性峰值
    peak_idx = np.argmax(gzip_vals)
    peak_time = times[peak_idx]
    peak_val = gzip_vals[peak_idx]

    lines = [
        "=" * 60,
        '咖啡混合模拟结果',
        "=" * 60,
        f"网格大小：{results['grid_size']}x{results['grid_size']}",
        f"交换尝试总次数：{results['n_steps'] * results['swaps_per_step']:,}",
        f"测量次数：{len(times)}",
        "",
        '复杂性（gzip 压缩大小）：',
        f"  初值：       {gzip_vals[0]} 字节",
        f"  峰值：       {peak_val} 字节（交换尝试次数为 {peak_time:,})",
        f"  终值：       {gzip_vals[-1]} 字节",
        f"  峰值/初值：{peak_val / max(gzip_vals[0], 1):.2f}x",
        "",
        '熵（全局香农熵）：',
        f"  初值：{entropy_vals[0]:.4f} 比特",
        f"  终值：{entropy_vals[-1]:.4f} 比特",
        "",
        '隆起概况（是否成立须看下方判定）：',
        f"  复杂性从 {gzip_vals[0]} 上升至 {peak_val} 字节",
        f"  随后变为 {gzip_vals[-1]} 字节。",
    ]

    if peak_val > gzip_vals[0] and peak_val > gzip_vals[-1]:
        lines.append('  [OK] 观察到特征性复杂性隆起。')
    else:
        lines.append('  [NOTE] 隆起不明显。可尝试更多步或更大网格；代理量本身不保证出现峰值。')

    lines.append("=" * 60)
    return "\n".join(lines)


# ============================================================================
# 主程序：直接执行时运行小型演示
# ============================================================================

if __name__ == "__main__":
    print('咖啡混合模拟')
    print('根据 Aaronson《复杂动力学第一定律》（2011）编写的教学模拟')
    print()

    # 小型演示
    results = run_simulation(
        grid_size=32,
        n_steps=5000,
        swaps_per_step=20,
        measure_interval=100,
        seed=42
    )

    print(summarize_results(results))

    # 展示几个网格快照
    print('\n网格快照已保存至 data/ 目录')
    print('运行 train_minimal.py 可执行完整模拟并绘图。')
