import os
import torch

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import warnings
import random
import math
import pickle
from torch.quasirandom import SobolEngine

warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")


def set_seed(seed_value):
    """
    设置所有必要的种子以保证可重复性

    参数:
        seed_value: 种子值
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


set_seed(666)


class SB:
    def __init__(
        self,
        A,
        h=None,
        K=1,
        delta=1,
        dt=1.0,
        sigma=1.0,
        M=2,
        n_iter=1000,
        xi=None,
        sk=False,
        batch_size=1,
        a=0,
        b=0.2,
        gamma=10,
        init_method=None,
        method="ISASB",
        device="cpu",
    ):
        self.device = device
        self.N = A.shape[0]
        self.A = A.to(self.device)
        self.h = h
        self.batch_size = batch_size
        self.K = K
        self.delta = delta
        self.dt = dt
        self.M = M
        self.n_iter = n_iter
        self.sigma = sigma
        self.method = method
        self.gamma = gamma
        if self.method == "ISASB":
            self.p = 1 - torch.exp(-self.gamma * torch.linspace(0, 1, self.n_iter))
        else:
            self.p = torch.linspace(0, 1, self.n_iter)
        self.dm = self.dt / self.M
        self.sk = sk
        self.xi = xi
        self.a = a
        self.b = b
        self.init_method = init_method
        if xi is not None:
            self.xi = xi
        else:
            if isinstance(self.Q, torch.Tensor):
                self.xi = 1 / torch.abs(self.Q.sum(dim=1)).max().item()
            else:
                self.xi = 1 / np.abs(self.Q.sum(axis=1)).max()
        self.initialize()

    def initialize(self):
        """
        初始化粒子位置和速度
        """
        if self.init_method == "sobol":
            sobol_engine = SobolEngine(dimension=self.N, scramble=True)
            sobol_samples = sobol_engine.draw(self.batch_size).to(self.device)
            self.x = (sobol_samples - 0.5) * 0.1
            self.x = self.x.T
        elif self.init_method == "prop":
            proportions = [0.2, 0.3, 0.5]
            self.x = torch.zeros((self.N, self.batch_size), device=self.device)
            for b in range(self.batch_size):
                prob = random.choice(proportions)
                binary_samples = torch.bernoulli(
                    torch.full((self.N,), prob, device=self.device)
                )
                negative_values = -0.05 * torch.rand_like(
                    binary_samples, device=self.device
                )
                positive_values = 0.05 * torch.rand_like(
                    binary_samples, device=self.device
                )
                self.x[:, b] = torch.where(
                    binary_samples == 0, negative_values, positive_values
                )
        else:
            self.x = 0.01 * (
                torch.rand(self.N, self.batch_size, device=self.device) - 0.5
            )
        self.y = 0.01 * (torch.rand(self.N, self.batch_size, device=self.device) - 0.5)
        #---- 改动：计算gbest ----
        initial_energies = self.calc_energy(self.x)
        best_index = initial_energies.argmin()
        self.gbest = torch.clone(self.x[:, best_index]).unsqueeze(1)
        self.gbest_energy = initial_energies[best_index].item()

    def calc_energy(self, x):
        """
        计算能量

        参数:
            x: 粒子位置

        返回:
            energy: 能量
        """
        # 确保x是二值的
        binary_x = torch.sign(x).to(self.device)
        binary_x[binary_x == 0] = 1.0
        self.A = self.A.float()
        # 计算QUBO能量: x^T Q x
        if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
            energy = -0.5 * torch.sum(
                torch.sparse.mm(self.A, binary_x) * binary_x, dim=0
            )
        else:
            energy = -0.5 * torch.sum(self.A @ binary_x * binary_x, dim=0)
            
        # 如果有线性项，加上线性项的贡献
        if self.h is not None:
            energy = energy - torch.sum(self.h * binary_x, dim=0)
            
        return energy           

    def calc_cut(self, x):
        """
        计算割值

        参数:
            x: 粒子位置

        返回:
            cut: 割值
        """
        sign = torch.sign(x)
        sign[sign == 0] = 1.0
        energy = self.calc_energy(x)
        cut = -0.5 * energy - 0.25 * self.A.sum()
        return cut

    # GbSB
    def update_b_comprehensive_learning(self):
        """
        更新粒子位置和速度，使用综合学习策略

        返回:
            cut_values: 割值列表
        """

        self.velocity_memory = torch.zeros_like(self.y)  # 初始化动量项
        i_tensor = torch.arange(self.n_iter, device=self.device, dtype=torch.float32)
        
        # 预计算概率参数
        probability = 0.1 + 0.45 * (1 + torch.cos(math.pi * i_tensor / self.n_iter))
        for i in range(self.n_iter):
            # 目标函数
            current_energy = self.calc_energy(self.x)
            min_energy = current_energy.min().item()
                
            # 更新 gbest
            if min_energy < self.gbest_energy:
                min_idx = torch.argmin(current_energy)
                self.best_energy = min_energy
                self.gbest = self.x[:, min_idx].clone().unsqueeze(1)
                
            step = i / self.n_iter
            
            # 榜样学习
            gbest_diff = self.gbest - self.x
            gbest_norm = torch.norm(gbest_diff, dim=0, keepdim=True)
            
            # 调整衰减因子
            decay_factor = 1 / (1 + gbest_norm) * (1 - step)
            info_share_term = decay_factor * gbest_diff
            
            # 增强方向协同性
            cos_sim = torch.cosine_similarity(info_share_term, self.y, dim=0)
            info_share_term = torch.where(
                cos_sim < -self.b, -info_share_term, info_share_term
            )
            
            # 历史加权平滑
            alpha = 0.9 - 0.4 * step
            self.velocity_memory = (
                alpha * self.velocity_memory + (1 - alpha) * info_share_term
            )
            info_share_term = self.velocity_memory

            if self.h is None:
                self.y = (
                    self.y
                    + (
                        -(self.delta - self.p[i]) * self.x
                        + self.xi * (torch.sparse.mm(self.A, self.x))
                        + info_share_term
                    )
                    * self.dt
                )
            else:
                self.y += (
                    -(self.delta - self.p[i]) * self.x
                    + self.xi * (torch.sparse.mm(self.A, self.x) + self.h)
                    + info_share_term
                ) * self.dt

            self.x += self.dt * self.y * self.delta
            
            # 动态扰动增强多样性
            rand_mask = torch.rand_like(self.x) < probability[i]
            self.x = torch.where(rand_mask, self.x - self.a, self.x)
            
            cond = torch.abs(self.x) > 1
            self.x = torch.where(cond, torch.sign(self.x), self.x)
            self.y = torch.where(cond, torch.zeros_like(self.y), self.y)


    # GdSB
    def update_d_comprehensive_learning(self):
        """
        更新粒子位置和速度，使用综合学习策略

        返回:
            cut_values: 割值列表
        """
        self.velocity_memory = torch.zeros_like(self.y)  # 初始化动量项
        i_tensor = torch.arange(self.n_iter, device=self.device, dtype=torch.float32)
        
        # 预计算概率参数
        probability = 0.1 + 0.45 * (1 + torch.cos(math.pi * i_tensor / self.n_iter))
        for i in range(self.n_iter):
            # 目标函数
            current_energy = self.calc_energy(self.x)
            min_energy = current_energy.min().item()
                
            # 更新 gbest
            if min_energy < self.gbest_energy:
                min_idx = torch.argmin(current_energy)
                self.best_energy = min_energy
                self.gbest = self.x[:, min_idx].clone().unsqueeze(1)
                
            step = i / self.n_iter
            
            # 榜样学习
            gbest_diff = self.gbest - self.x
            gbest_norm = torch.norm(gbest_diff, dim=0, keepdim=True)
            
            # 调整衰减因子
            decay_factor = 1 / (1 + gbest_norm) * (1 - step)
            info_share_term = decay_factor * gbest_diff
            
            # 增强方向协同性
            cos_sim = torch.cosine_similarity(info_share_term, self.y, dim=0)
            info_share_term = torch.where(
                cos_sim < -self.b, -info_share_term, info_share_term
            )
            
            # 历史加权平滑
            alpha = 0.9 - 0.4 * step
            self.velocity_memory = (
                alpha * self.velocity_memory + (1 - alpha) * info_share_term
            )
            info_share_term = self.velocity_memory

            if self.h is None:
                self.y = (
                    self.y
                    + (
                        -(self.delta - self.p[i]) * self.x
                        + self.xi * (torch.sparse.mm(self.A, torch.sign(self.x)))
                        + info_share_term
                    )
                    * self.dt
                )
            else:
                self.y += (
                    -(self.delta - self.p[i]) * self.x
                    + self.xi * (torch.sparse.mm(self.A, torch.sign(self.x)) + self.h)
                    + info_share_term
                ) * self.dt
                
            self.x += self.dt * self.y * self.delta
            
            # 动态扰动增强多样性
            rand_mask = torch.rand_like(self.x) < probability[i]
            self.x = torch.where(rand_mask, self.x - self.a, self.x)
            
            cond = torch.abs(self.x) > 1
            self.x = torch.where(cond, torch.sign(self.x), self.x)
            self.y = torch.where(cond, torch.zeros_like(self.y), self.y)


def read_gset(filename, negate=True):
    """
    读取图数据

    参数:
        filename: 文件名
        negate: 是否取反

    返回:
        G: 邻接矩阵
    """
    graph = pd.read_csv(filename, sep=" ")
    n_v = int(graph.columns[0])
    n_e = int(graph.columns[1])
    assert n_e == graph.shape[0], "The number of edges is not matched"
    G = csr_matrix(
        (graph.iloc[:, -1], (graph.iloc[:, 0] - 1, graph.iloc[:, 1] - 1)),
        shape=(n_v, n_v),
    )
    G = G + G.T
    if negate:
        return -G
    else:
        return G


if __name__ == "__main__":
    graphs = ["G1"]
    known_maxs = [
        11624
    ]
    
    # 选择GbSB还是GdSB
    algorithm = 'b'
    
    if algorithm == 'd':
        # a值
        a_values = {
            "G1": 0.01,
        }

        # 余弦翻转阈值
        b_values = {
            "G1": 0.05,
        }
        
        xis = {
            "G1": 0.02,
        }
        
        dts = {
            "G1": 1,
        }
        
        niters = {
            "G1": 10000,
        }
        
        gammas = {
            "G1": 5.0,
        }
    elif algorithm == 'b':
        # a值
        a_values = {
            "G1": 0.01,
        }

        # 余弦翻转阈值
        b_values = {
            "G1": 0.33,
        }

        xis = {
            "G1": 0.05,
        }

        dts = {
            "G1": 1,
        }
        
        niters = {
            "G1": 10000,
        }
        
        gammas = {
            "G1": 5.0,
        }
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
        
    device = "cuda"
    # n_iter = 10000
    num_tests = 1
    

    # 遍历图和已知最优解
    for graph, known_max in zip(graphs, known_maxs):
        if graph == "K2000":
            J = read_gset(f"SB/graphs/WK2000_1.rud", negate=True)
        else:
            J = read_gset(f"SB/graphs/{graph}.txt", negate=True)

        J = torch.from_numpy(J.todense())
        N = J.shape[0]
        
        if graph in xis:
            xi = xis[graph]
        else:
            xi = 1 / torch.abs(J.sum(axis=1)).max()
            
        
        J = J.cuda().float()
        J = J.to_sparse_csr()
        J = J.to(device)
        results = []
        times = []
        a = a_values[graph]
        b = b_values[graph]
        xi = xis[graph]
        dt = dts[graph]
        n_iter = niters[graph]
        gamma = gammas[graph]
        
        print("Graph:", graph)
        print(f"Algorithm: G{algorithm}SB")
        for _ in range(num_tests):
            s = SB(
                J,
                n_iter=n_iter,
                xi=xi,
                dt=dt,
                K=1,
                sk=False,
                batch_size=1000,
                a=a,
                b=b,
                gamma=gamma,
                init_method='prop',
                method="ISASB",
                device=device,
            )
            # 选择算法
            if algorithm == 'd':
                s.update_d_comprehensive_learning()
            elif algorithm == 'b':
                s.update_b_comprehensive_learning()
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            # 解
            best_sample = torch.sign(s.x).clone()

            energy = -0.5 * torch.sum(J @ best_sample * best_sample, dim=0)
            cut = -0.5 * energy - 0.25 * J.sum()
            max_cut = torch.max(cut)
            
            # 找到最优值概率
            proportion_max_cut = torch.mean((cut == known_max).float())
            results.append(
                {
                    "max_cut": max_cut.item(),
                    "proportion_max_cut": proportion_max_cut.item(),
                }
            )

        # num_test次找到最优解的平均概率
        proportions_max_cut = [result["proportion_max_cut"] for result in results]
        mean_proportion_max_cut = np.mean(proportions_max_cut)

        print(f"Average proportion of maximum cuts: {mean_proportion_max_cut}")

