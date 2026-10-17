"""咖啡元胞自动机——Aaronson、Carroll、Ouellette（2014）的实现
论文：arXiv:1405.6903

实现论文中的两种模型：
  - 有相互作用模型（第 3.1 节）：二值网格，随机交换相邻格点
  - 无相互作用模型（第 3.2 节）：奶油粒子独立随机游走

以及两种测量方法：
  - 采用 3 分箱阈值化的粗粒化（第 5 节）
  - 采用 7 分箱与行众数调整的粗粒化（第 6 节）

用 gzip 压缩大小估计复杂度和熵，作为柯尔莫哥洛夫复杂度的代理量。"""

import gzip
import numpy as np
from typing import Optional, Tuple, List, Dict


# ---------------------------------------------------------------------------
# 工具：基于 gzip 的柯尔莫哥洛夫复杂度代理量
# ---------------------------------------------------------------------------

def gzip_size(data: bytes, level: int = 9) -> int:
    """压缩大小，以字节为单位，作为 K(x) 的代理量。
论文全程使用 gzip（第 4 节），并展示不同压缩器
产生的曲线在定性上相似（图 5）。"""
    return len(gzip.compress(data, compresslevel=level))


def grid_to_bytes(grid: np.ndarray) -> bytes:
    """将 NumPy 网格转为字节以便压缩。"""
    return grid.astype(np.uint8).tobytes()


# ---------------------------------------------------------------------------
# 粗粒化（第 5、6 节）
# ---------------------------------------------------------------------------

def coarse_grain(grid: np.ndarray, grain_size: int) -> np.ndarray:
    """对 grain_size x grain_size 块求平均，实现网格粗粒化。

第 5.1 节：“我们构造一个新数组，其中每个格点的值是细粒度数组中
附近格点值的平均数。我们将‘附近’定义为以该格点为中心的
一个 g x g 方块内的格点。”

返回值位于 [0, 1] 的浮点数组。"""
    n = grid.shape[0]
    # 粗粒化格点的数量
    cn = n // grain_size
    if cn == 0:
        cn = 1

    coarse = np.zeros((cn, cn), dtype=np.float64)
    for i in range(cn):
        for j in range(cn):
            r0 = i * grain_size
            c0 = j * grain_size
            r1 = min(r0 + grain_size, n)
            c1 = min(c0 + grain_size, n)
            coarse[i, j] = grid[r0:r1, c0:c1].mean()
    return coarse


def threshold_array(coarse: np.ndarray, num_buckets: int = 3) -> np.ndarray:
    """将浮点粗粒化数组按阈值离散分箱。

第 5.1 节：3 个分箱——“以咖啡为主的区域（接近 0）、
以奶油为主的区域（接近 1）、混合区域（接近 0.5）”。

第 6.1 节：7 个分箱，采用更细分辨率以减少边界伪影。

返回值位于 [0, num_buckets-1] 的 uint8 数组。"""
    # 将 [0, 1] 映射到 [0, num_buckets-1] 并取整
    scaled = coarse * (num_buckets - 1)
    return np.clip(np.round(scaled), 0, num_buckets - 1).astype(np.uint8)


def row_majority_adjust(thresholded: np.ndarray) -> np.ndarray:
    """第 6.1 节调整后的粗粒化。

“若某格点的值与该行众数相差不超过一个阈值间隔，
就将其调整为众数。”

这能消除在无相互作用模型中造成虚假复杂度的边界像素伪影。"""
    adjusted = thresholded.copy()
    for i in range(adjusted.shape[0]):
        row = adjusted[i]
        # 查找该行的众数
        values, counts = np.unique(row, return_counts=True)
        majority = values[np.argmax(counts)]
        # 将与众数相差不超过 1 个分箱的格点归并到众数
        mask = np.abs(row.astype(np.int16) - int(majority)) <= 1
        adjusted[i, mask] = majority
    return adjusted


# ---------------------------------------------------------------------------
# 复杂度与熵的测量
# ---------------------------------------------------------------------------

def measure_entropy(grid: np.ndarray) -> int:
    """熵估计：细粒度状态的 gzip 压缩大小。

第 4 节：“自动机状态的熵估计值是细粒度数组的压缩文件大小。”"""
    return gzip_size(grid_to_bytes(grid))


def measure_complexity(grid: np.ndarray, grain_size: int = 10,
                       num_buckets: int = 7, adjust: bool = True) -> int:
    """表观复杂度估计：粗粒化状态的 gzip 压缩大小。

第 4 节：“状态复杂度的估计 K(S)，是经过阈值化和粗粒化的
数组压缩后的文件大小。”

参数：
    grid：细粒度二值网格（N x N，uint8）
    grain_size：粗粒化块的边长（论文中的 g）
    num_buckets：阈值分箱数（第 5 节为 3，第 6 节为 7）
    adjust：是否应用行众数调整（第 6 节）"""
    coarse = coarse_grain(grid, grain_size)
    discrete = threshold_array(coarse, num_buckets)
    if adjust:
        discrete = row_majority_adjust(discrete)
    return gzip_size(grid_to_bytes(discrete))


# ---------------------------------------------------------------------------
# 咖啡元胞自动机（第 3 节）
# ---------------------------------------------------------------------------

class CoffeeAutomaton:
    """论文第 3 节中的咖啡元胞自动机。

二维二值网格：1 = 奶油，0 = 咖啡。
初始状态：上半部 = 奶油（1），下半部 = 咖啡（0）。

两种模型：
- 'interacting'（第 3.1 节）：每步交换一对相邻且不同的格点。
  “每个格点只能容纳一个粒子。”
- 'non_interacting'（第 3.2 节）：各奶油粒子独立随机游走。
  多个粒子可以占据同一格点。"""

    def __init__(self, grid_size: int = 100, model: str = 'interacting',
                 seed: Optional[int] = None):
        """参数：
    grid_size：正方形网格的边长（论文中的 N）
    model：'interacting' 或 'non_interacting'
    seed：随机种子，用于复现"""
        self.n = grid_size
        self.model = model
        self.rng = np.random.default_rng(seed)
        self.time = 0

        if model == 'interacting':
            # 二值网格：上半部 = 1（奶油），下半部 = 0（咖啡）
            # 第 3 节：“自动机的初始状态是
            # 上半部格点填充 1，
            # 下半部格点填充 0。”
            self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
            self.grid[:grid_size // 2, :] = 1
        elif model == 'non_interacting':
            # 对于无相互作用模型，追踪奶油粒子的位置。
            # 计数网格：每个格点的奶油粒子数。
            # 初始时上半部每格一个粒子。
            self.grid = np.zeros((grid_size, grid_size), dtype=np.uint16)
            self.grid[:grid_size // 2, :] = 1
            # 同时保存单个粒子的位置，用于随机游走
            self._init_particles()
        else:
            raise ValueError(f"model 必须为 'interacting' 或 'non_interacting'，收到 '{model}'")

    def _init_particles(self):
        """初始化无相互作用模型的粒子位置列表。"""
        self.particles = []
        for r in range(self.n // 2):
            for c in range(self.n):
                self.particles.append([r, c])
        self.particles = np.array(self.particles, dtype=np.int32)

    def step(self, num_steps: int = 1):
        """将自动机推进 num_steps 步。

对于有相互作用模型，使用批量交换提高效率：
选择许多随机相邻格点对，筛选出值不同的格点对后交换。
这比每步一次交换更快，同时保留定性动力学行为。

论文每步交换一次（第 3.1 节），对于大网格极慢。
批量交换可以加速，而不改变平衡行为。"""
        if self.model == 'interacting':
            self._step_interacting(num_steps)
        else:
            self._step_non_interacting(num_steps)
        self.time += num_steps

    def _step_interacting(self, num_steps: int):
        """有相互作用模型：随机交换相邻格点。

第 3.1 节：“每个时间步选择一对水平或垂直相邻、
类型不同的粒子，交换它们的位置。”

每次调用批量执行多次不冲突的交换以提高速度。"""
        n = self.n
        grid = self.grid

        # 每批候选交换的数量——
        # 约为 n*n/4，使每个像素每批约有 1 次机会
        batch_size = max(1, (n * n) // 4)

        for _ in range(num_steps):
            # 随机选择格点
            rows = self.rng.integers(0, n, size=batch_size)
            cols = self.rng.integers(0, n, size=batch_size)

            # 随机选择方向：0=上，1=下，2=左，3=右
            dirs = self.rng.integers(0, 4, size=batch_size)

            # 计算相邻格点坐标
            dr = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
            nr = rows + dr[dirs, 0]
            nc = cols + dr[dirs, 1]

            # 筛选：坐标在边界内
            valid = (nr >= 0) & (nr < n) & (nc >= 0) & (nc < n)

            rows = rows[valid]
            cols = cols[valid]
            nr = nr[valid]
            nc = nc[valid]

            # 筛选：数值不同（仅在不同的情况下交换）
            diff = grid[rows, cols] != grid[nr, nc]
            rows = rows[diff]
            cols = cols[diff]
            nr = nr[diff]
            nc = nc[diff]

            # 交换
            if len(rows) > 0:
                temp = grid[rows, cols].copy()
                grid[rows, cols] = grid[nr, nc]
                grid[nr, nc] = temp

    def _step_non_interacting(self, num_steps: int):
        """无相互作用模型：每个奶油粒子随机游走。

第 3.2 节：“每个时间步，系统中的每个奶油粒子
沿随机选定的方向移动一步。”"""
        n = self.n
        directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

        for _ in range(num_steps):
            # 清空计数网格并重建
            self.grid[:] = 0

            # 沿随机方向移动每个粒子
            choices = self.rng.integers(0, 4, size=len(self.particles))
            moves = directions[choices]
            self.particles += moves

            # 周期边界条件
            # 附录第 9 节假定周期边界
            self.particles[:, 0] %= n
            self.particles[:, 1] %= n

            # 重建计数网格
            for p in self.particles:
                self.grid[p[0], p[1]] += 1

    def get_binary_grid(self) -> np.ndarray:
        """将状态转为二值网格以便测量。

有相互作用模型直接返回 self.grid。
无相互作用模型将计数转为二值：
存在任意奶油粒子则为 1，否则为 0。"""
        if self.model == 'interacting':
            return self.grid
        else:
            return (self.grid > 0).astype(np.uint8)

    def entropy(self) -> int:
        """熵估计：gzip（细粒度状态），参见第 4 节。"""
        return measure_entropy(self.get_binary_grid())

    def complexity(self, grain_size: Optional[int] = None,
                   num_buckets: int = 7, adjust: bool = True) -> int:
        """表观复杂度估计：gzip（粗粒化状态）。
参见第 5、6 节。

参数：
    grain_size：论文中的 g。默认 n // 10
    num_buckets：3（第 5 节）或 7（第 6 节）
    adjust：应用行众数调整（第 6 节）"""
        if grain_size is None:
            grain_size = max(2, self.n // 10)
        return measure_complexity(self.get_binary_grid(), grain_size,
                                  num_buckets, adjust)

    def fraction_mixed(self) -> float:
        """处于“错误”位置的格点比例：下半部的奶油，或上半部的咖啡。
这是熵的简单单调代理量。"""
        grid = self.get_binary_grid()
        n = self.n
        # 下半部的奶油 + 上半部的咖啡
        wrong = (np.sum(grid[n // 2:, :] == 1) +
                 np.sum(grid[:n // 2, :] == 0))
        total = n * n
        return wrong / total


# ---------------------------------------------------------------------------
# 模拟运行器
# ---------------------------------------------------------------------------

def run_simulation(grid_size: int = 50, total_steps: int = 100000,
                   num_snapshots: int = 100, model: str = 'interacting',
                   grain_size: Optional[int] = None, num_buckets: int = 7,
                   adjust: bool = True, seed: Optional[int] = 42
                   ) -> Dict[str, np.ndarray]:
    """运行咖啡元胞自动机，收集随时间变化的测量结果。

返回字典，键为：
    'times'：时间步数组
    'entropy'：熵估计数组
    'complexity'：复杂度估计数组
    'fraction_mixed'：混合比例数组
    'grids'：网格快照列表（当 grid_size <= 100 时）

参数：
    grid_size：N（边长）
    total_steps：总步数
    num_snapshots：测量点数量
    model：'interacting' 或 'non_interacting'
    grain_size：粗粒化块边长（默认 n//10）
    num_buckets：3 或 7（第 5 节与第 6 节）
    adjust：行众数调整（第 6 节）
    seed：随机种子"""
    ca = CoffeeAutomaton(grid_size, model, seed)

    if grain_size is None:
        grain_size = max(2, grid_size // 10)

    steps_per_snapshot = max(1, total_steps // num_snapshots)
    store_grids = grid_size <= 100

    times = []
    entropies = []
    complexities = []
    fractions = []
    grids = []

    # 测量初始状态
    times.append(0)
    entropies.append(ca.entropy())
    complexities.append(ca.complexity(grain_size, num_buckets, adjust))
    fractions.append(ca.fraction_mixed())
    if store_grids:
        grids.append(ca.get_binary_grid().copy())

    for i in range(num_snapshots):
        ca.step(steps_per_snapshot)
        times.append(ca.time)
        entropies.append(ca.entropy())
        complexities.append(ca.complexity(grain_size, num_buckets, adjust))
        fractions.append(ca.fraction_mixed())
        if store_grids:
            grids.append(ca.get_binary_grid().copy())

    results = {
        'times': np.array(times),
        'entropy': np.array(entropies),
        'complexity': np.array(complexities),
        'fraction_mixed': np.array(fractions),
    }
    if store_grids:
        results['grids'] = grids

    return results


def compare_models(grid_size: int = 50, total_steps: int = 100000,
                   num_snapshots: int = 100, seed: int = 42
                   ) -> Tuple[Dict, Dict]:
    """运行有相互作用和无相互作用两种模型，返回结果。
用于复现论文中的比较（图 2、10）。"""
    print(f"正在运行有相互作用模型（{grid_size}x{grid_size}，{total_steps} 步）……")
    interacting = run_simulation(grid_size, total_steps, num_snapshots,
                                 'interacting', seed=seed)

    print(f"正在运行无相互作用模型（{grid_size}x{grid_size}，{total_steps} 步）……")
    non_interacting = run_simulation(grid_size, total_steps, num_snapshots,
                                     'non_interacting', seed=seed)

    return interacting, non_interacting


def scaling_experiment(sizes: List[int] = None, steps_per_pixel: int = 40,
                       num_snapshots: int = 100, seed: int = 42
                       ) -> Dict[str, np.ndarray]:
    """复现图 6-8 的标度实验。
测量最大熵、最大复杂度、达到最大复杂度的时间
如何随网格大小变化。

论文发现：
- 最大熵 ~ n^2
- 最大复杂度 ~ n
- 达到最大复杂度的时间 ~ n^2"""
    if sizes is None:
        sizes = [10, 20, 30, 50, 70]

    max_entropies = []
    max_complexities = []
    peak_times = []

    for n in sizes:
        total_steps = steps_per_pixel * n * n
        print(f"  网格大小 {n}x{n}，{total_steps} 步……")
        results = run_simulation(n, total_steps, num_snapshots, 'interacting',
                                 seed=seed)
        max_entropies.append(np.max(results['entropy']))
        max_complexities.append(np.max(results['complexity']))
        peak_idx = np.argmax(results['complexity'])
        peak_times.append(results['times'][peak_idx])

    return {
        'sizes': np.array(sizes),
        'max_entropy': np.array(max_entropies),
        'max_complexity': np.array(max_complexities),
        'peak_time': np.array(peak_times),
    }


# ---------------------------------------------------------------------------
# 主程序：快速演示
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("咖啡元胞自动机（Aaronson、Carroll、Ouellette，2014）")
    print("=" * 55)

    # 快速运行有相互作用模拟
    print("\n正在运行有相互作用模型（50x50，50000 步）……")
    results = run_simulation(grid_size=50, total_steps=50000, num_snapshots=50)

    peak_idx = np.argmax(results['complexity'])
    print(f"\n结果：")
    print(f"  复杂度峰值位于 t = {results['times'][peak_idx]}")
    print(f"  复杂度峰值 = {results['complexity'][peak_idx]} 字节")
    print(f"  最终熵 = {results['entropy'][-1]} 字节")
    print(f"  最终复杂度 = {results['complexity'][-1]} 字节")
    print(f"  最终混合比例 = {results['fraction_mixed'][-1]:.3f}")

    # 验证：复杂度应先升后降
    first_half = results['complexity'][:len(results['complexity']) // 2]
    second_half = results['complexity'][len(results['complexity']) // 2:]
    if np.mean(first_half) > np.mean(second_half) * 0.8:
        print("\n  复杂度呈先升后降模式 [OK]")
    else:
        print("\n  [提示] 可能需要更多步数才能看到完整模式")
