import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# # 定义一个钩子函数，专门检查流经这里的梯度
# def check_grad_hook(grad):
#     if torch.isnan(grad).any() or torch.isinf(grad).any():
#         print(f"!!! [Backward Hook 警告] 梯度流经 decay_s4 时发现 NaN/Inf !!!")
#         print(f"    梯度统计: Min={grad.min()}, Max={grad.max()}")
        
#         # 【救命操作】: 如果发现梯度坏了，强制把它变成 0
#         # 这样可以防止这个 NaN 继续向后传播导致 SigmoidBackward 报错崩溃
#         return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    
#     # 如果梯度正常，原样返回
#     return grad
import torch
import torch.nn as nn
import torch.nn.functional as F

class DendriteNode_Basal(nn.Module):
    def __init__(self, n_neuron=64, n_leaf=3, leaf_dim=4):
        super().__init__()
        assert n_leaf == 3
        assert leaf_dim == 4
        self.n_neuron = n_neuron
        self.n_leaf = n_leaf
        self.leaf_dim = leaf_dim
        # 叶子节点s1,s2,s3
        # 叶子二次项: [1, 64, 3, 6], pair 顺序固定: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        self.leaf_quad = nn.Parameter(torch.zeros(1, n_neuron, n_leaf, 6))
        # 叶子 tau (log): [1, 64, 3, 1]
        self.decay_leaf = nn.Parameter(torch.rand(1, n_neuron, n_leaf, 1)*4)
        # 枝干节点s4 (整合 s1,s2)
        self.w41 = nn.Parameter(torch.ones(1, n_neuron, 1, 1))
        self.w42 = nn.Parameter(torch.ones(1, n_neuron, 1, 1))
        # init_tau_s4 = torch.full((1, n_neuron, 1, 1), 8.0)
        self.decay_s4 = nn.Parameter(torch.rand(1, n_neuron, 1, 1)*4)
        # bAP 衰减: 每个神经元、每个节段一份 [1, 64, 3, 1]
        self.bAP_decay_leaf = nn.Parameter(torch.full((1, n_neuron, n_leaf, 1), 0.1))
        self.bAP_decay_s4 = nn.Parameter(torch.full((1, n_neuron, 1, 1), 0.1))
        # dendrite整合输出 (整合 s3, s4)
        self.ws4 = nn.Parameter(torch.ones(1, n_neuron, 1, 1))
        self.ws3 = nn.Parameter(torch.ones(1, n_neuron, 1, 1))

    def _leaf_bilinear(self, h_leaf):
        """
        h_leaf: [B,64,3,4]
        return: [B,64,3,1]
        """
        # 线性部分
        lin = (h_leaf).sum(dim=-1, keepdim=True)  # [B,64,3,1]
        # 4分量
        x0, x1, x2, x3 = [h_leaf[..., i] for i in range(4)]
        # 6 个 pair
        pairs = torch.stack([x0*x1, x0*x2, x0*x3, x1*x2, x1*x3, x2*x3], dim=-1)
        quad = (pairs * self.leaf_quad).sum(dim=-1, keepdim=True)  # [B,64,3,1]
        return lin + quad

    def forward(self, leaf_in, bAP, prev_h_leaf, prev_h4):
        """
        leaf_in: [B,64,3,4]
        bAP:     [B,64]
        return:  [B,64]
        """
        B = leaf_in.size(0)
        # 0) 状态初始化
        if prev_h_leaf is None:
            prev_h_leaf = leaf_in.new_zeros(B, self.n_neuron, self.n_leaf, self.leaf_dim)
            prev_h4 = leaf_in.new_zeros(B, self.n_neuron, 1, 1)
        decay_leaf = torch.sigmoid(self.decay_leaf)
        decay_s4 = torch.sigmoid(self.decay_s4)
        bAP_decay_leaf = torch.tanh(self.bAP_decay_leaf)
        bAP_decay_s4 = torch.tanh(self.bAP_decay_s4)
        # 1) bAP
        bAP_4d = bAP.unsqueeze(2).unsqueeze(3) # [B,N,1,1]
        bAP_s4 = bAP_4d * bAP_decay_s4  # [B,64,1,1]
        bAP_s1 = bAP_s4 * bAP_decay_leaf[:, :, 0:1, :]
        bAP_s2 = bAP_s4 * bAP_decay_leaf[:, :, 1:2, :]
        bAP_s3 = bAP_4d * bAP_decay_leaf[:, :, 2:3, :]
        bAP_leaf = torch.cat([bAP_s1, bAP_s2, bAP_s3], dim=2)
        h_leaf_prev = prev_h_leaf
        h_leaf_bAP = h_leaf_prev + bAP_leaf  # [B,64,3,4]
        h4_prev = prev_h4
        h4_bAP = h4_prev + bAP_s4
        # 2) 叶子输入
        h_leaf_new = decay_leaf * h_leaf_bAP + (1 - decay_leaf) * leaf_in.unsqueeze(-1).unsqueeze(-1) #[B,64,3,4]
        # 3) 叶子非线性
        leaf_sc = self._leaf_bilinear(h_leaf_new)  # [B,64,3,1]
        s1 = leaf_sc[:, :, 0:1, :]  # [B,64,1,1]
        s2 = leaf_sc[:, :, 1:2, :]
        s3 = leaf_sc[:, :, 2:3, :]
        # 4) s4 更新
        u4 = self.w41 * s1 + self.w42 * s2  # [B,64,1,1]
        h4_new = decay_s4 * h4_bAP + (1 - decay_s4) * u4
        # 5) dendrite输出
        u_dendrite = self.ws4 * h4_new + self.ws3 * s3  # [B,64,1,1]
        out = u_dendrite.squeeze(-1).squeeze(-1) 
        return h_leaf_new, h4_new, out # [B,64]
    
class DendriteNode_Apical(nn.Module):
    def __init__(self, n_neuron=64, n_leaf=3, leaf_dim=4):
        super().__init__()
        assert n_leaf == 3
        assert leaf_dim == 4
        self.n_neuron = n_neuron
        self.n_leaf = n_leaf
        self.leaf_dim = leaf_dim
        # 叶子节点s1,s2,s3
        # 叶子二次项: [1, 64, 3, 6], pair 顺序固定: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        self.leaf_quad = nn.Parameter(torch.zeros(1, n_neuron, n_leaf, 6))
        self.decay_leaf = nn.Parameter(torch.rand(1, n_neuron, n_leaf, 1)*4)
        # 枝干节点s4 (整合 s1,s2)
        self.w41 = nn.Parameter(torch.ones(1, n_neuron, 1, 1))
        self.w42 = nn.Parameter(torch.ones(1, n_neuron, 1, 1))
        self.decay_s4 = nn.Parameter(torch.rand(1, n_neuron, 1, 1)*4)
        # dendrite整合输出 (整合 s3, s4)
        self.ws4 = nn.Parameter(torch.ones(1, n_neuron, 1, 1))
        self.ws3 = nn.Parameter(torch.ones(1, n_neuron, 1, 1))

    def _leaf_bilinear(self, h_leaf):
        """
        h_leaf: [B,64,3,4]
        return: [B,64,3,1]
        """
        # 线性部分
        lin = (h_leaf).sum(dim=-1, keepdim=True)  # [B,64,3,1]
        # 4分量
        x0, x1, x2, x3 = [h_leaf[..., i] for i in range(4)]
        # 6 个 pair
        pairs = torch.stack([x0*x1, x0*x2, x0*x3, x1*x2, x1*x3, x2*x3], dim=-1)
        quad = (pairs * self.leaf_quad).sum(dim=-1, keepdim=True)  # [B,64,3,1]
        return lin + quad

    def forward(self, leaf_in, prev_h_leaf, prev_h4):
        """
        leaf_in: [B,64,3,4]
        bAP:     [B,64]
        return:  [B,64]
        """
        B = leaf_in.size(0)
        # 0) 状态初始化
        if prev_h_leaf is None:
            prev_h_leaf = leaf_in.new_zeros(B, self.n_neuron, self.n_leaf, self.leaf_dim)
            prev_h4 = leaf_in.new_zeros(B, self.n_neuron, 1, 1)
        decay_leaf = torch.sigmoid(self.decay_leaf)
        decay_s4 = torch.sigmoid(self.decay_s4)
        # 2) 叶子输入
        h_leaf_new = decay_leaf * prev_h_leaf + (1 - decay_leaf) * leaf_in.unsqueeze(-1).unsqueeze(-1) # [B,64,3,4]
        # 3) 叶子非线性
        leaf_sc = self._leaf_bilinear(h_leaf_new)  # [B,64,3,1]
        s1 = leaf_sc[:, :, 0:1, :]  # [B,64,1,1]
        s2 = leaf_sc[:, :, 1:2, :]
        s3 = leaf_sc[:, :, 2:3, :]
        # 4) s4 更新
        u4 = self.w41 * s1 + self.w42 * s2  # [B,64,1,1]
        h4_new = decay_s4 * prev_h4 + (1 - decay_s4) * u4
        # 5) dendrite输出
        u_dendrite = self.ws4 * h4_new + self.ws3 * s3  # [B,64,1,1]
        out = u_dendrite.squeeze(-1).squeeze(-1) 
        return h_leaf_new, h4_new, out #[B,64]

# ====== 脉冲 soma：LIF (compile friendly)======
class _STEHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        sig = torch.sigmoid(ctx.alpha * x)
        return grad_output * (ctx.alpha * sig * (1. - sig)), None

def ste_heaviside(x, alpha: float = 4.0):
    return _STEHeaviside.apply(x, alpha)

class SomaNode(nn.Module):
    def __init__(
            self,
            neuron_num: int,
            tau: float = 2.0,
            v_threshold: float = 1.0,
            v_reset: float = 0.0,
            v_rest: float = 0.0,
            alpha: float = 4.0,
    ):
        super().__init__()
        self.neuron_num = neuron_num
        self.tau = tau
        self.v_th = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.alpha = alpha

        # 预建常量
        self.register_buffer('v_th_tensor', torch.tensor(self.v_th))
        self.register_buffer('v_reset_tensor', torch.tensor(self.v_reset))
        self.register_buffer('v_rest_tensor', torch.tensor(self.v_rest))
        self.register_buffer('one_over_tau', torch.tensor(1.0 / self.tau))

        # 状态缓存 (lazy init)
        self.v = None
        self.s = None

    @torch.no_grad()
    def reset_state(self, batch_size=None):
        """重置膜电位与 spike 状态"""
        if batch_size is None and self.v is not None:
            batch_size = self.v.size(0)
        elif batch_size is None:
            batch_size = 1
        device = self.v_th_tensor.device
        self.v = torch.full((batch_size, self.neuron_num), self.v_rest, device=device)
        self.s = torch.zeros((batch_size, self.neuron_num), device=device)

    def membrane_update_one_step(self, x: torch.Tensor,) -> torch.Tensor:
        # 1. 状态检查与初始化
        if self.v is None or self.v.size(0) != x.size(0):
            self.reset_state(batch_size=x.size(0))
        # 2. 膜电位更新公式
        v_next = self.v + (-(self.v - self.v_rest_tensor) + x) * self.one_over_tau
        # 3. 脉冲生成判断
        u = v_next - self.v_th_tensor
        s_now = ste_heaviside(u, self.alpha)

        # 5. 状态更新（无梯度）
        with torch.no_grad():
            v_new = torch.where(s_now.bool(), self.v_reset_tensor, v_next)
            self.v.copy_(v_new)
            self.s.copy_(s_now)
        return s_now

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.membrane_update_one_step(x)


# ====== burst编码：fire由Is决定，burst由Ia决定 ======
class Burst(nn.Module):
    def __init__(self,ratio=0.2):
        super().__init__()
        # self.burst_ratio = nn.Parameter(torch.tensor(0.2))
        self.burst_ratio = torch.tensor(ratio) 

    def forward(self, Ia: torch.Tensor) -> torch.Tensor:
        burst_a = F.relu(Ia) * self.burst_ratio
        burst_b = torch.floor(burst_a)
        burst_num_raw = burst_b + (burst_a - burst_a.detach()) + 1
        x = burst_num_raw
        x_clamped = x.clamp(1.0, 4.0)
        burst_num = x + (x_clamped - x).detach()
        return burst_num


# ====== branch输入稀疏分配 ======
def make_idx_balanced(in_dim: int = 256, out_dim: int = 768) -> torch.Tensor:
    assert out_dim >= in_dim, f"out_dim {out_dim} 必须 >= in_dim {in_dim}"
    base = torch.arange(in_dim)
    rest = torch.randint(0, in_dim, (out_dim - in_dim,))  #","表示元组
    idx = torch.cat([base, rest], dim=0)
    idx = idx[torch.randperm(out_dim)]
    return idx

class SparseDistributorToLeaf(nn.Module):
    def __init__(self, in_dim=256, n_neuron=64, n_leaf=3, leaf_dim=4, learn_weight=True):
        super().__init__()
        out_dim = n_neuron * n_leaf * leaf_dim  # 64*3*4=768
        idx = make_idx_balanced(in_dim, out_dim)
        self.register_buffer("idx", idx)
        if learn_weight:
            self.w = nn.Parameter(torch.ones(out_dim))
        else:
            self.register_buffer("w", torch.ones(out_dim))
        self.n_neuron = n_neuron
        self.n_leaf = n_leaf
        self.leaf_dim = leaf_dim

    def forward(self, x):  # x: [B,256]
        B = x.size(0)
        sel = x[:, self.idx]                # [B, 768]
        y = sel * self.w                    # [B, 768]
        y = y.view(B, self.n_neuron, self.n_leaf, self.leaf_dim)  # [B,64,3,4]
        return y


# ====== dendrite branch ======
class DendriteNode(nn.Module):
    def __init__(self, n_neuron=64, n_leaf=3, leaf_dim=4):
        super().__init__()
        assert n_leaf == 3
        assert leaf_dim == 4
        self.n_neuron = n_neuron
        self.n_leaf = n_leaf
        self.leaf_dim = leaf_dim
        # 叶子节点s1,s2,s3
        # 叶子二次项: [1, 64, 3, 6], pair 顺序固定: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        self.leaf_quad = nn.Parameter(torch.zeros(1, n_neuron, n_leaf, 6))
        # 叶子 tau (log): [1, 64, 3, 1]
        self.decay_leaf = nn.Parameter(torch.rand(1, n_neuron, n_leaf, 1)*4)
        # 枝干节点s4 (整合 s1,s2)
        self.w41 = nn.Parameter(torch.ones(1, n_neuron, 1, 1))
        self.w42 = nn.Parameter(torch.ones(1, n_neuron, 1, 1))
        # init_tau_s4 = torch.full((1, n_neuron, 1, 1), 8.0)
        self.decay_s4 = nn.Parameter(torch.rand(1, n_neuron, 1, 1)*4)
        # bAP 衰减: 每个神经元、每个节段一份 [1, 64, 3, 1]
        self.bAP_decay_leaf = nn.Parameter(torch.full((1, n_neuron, n_leaf, 1), 0.1))
        self.bAP_decay_s4 = nn.Parameter(torch.full((1, n_neuron, 1, 1), 0.1))
        # dendrite整合输出 (整合 s3, s4)
        self.ws4 = nn.Parameter(torch.ones(1, n_neuron, 1, 1))
        self.ws3 = nn.Parameter(torch.ones(1, n_neuron, 1, 1))

    def _leaf_bilinear(self, h_leaf):
        """
        h_leaf: [B,64,3,4]
        return: [B,64,3,1]
        """
        # 线性部分
        lin = (h_leaf).sum(dim=-1, keepdim=True)  # [B,64,3,1]
        # 4分量
        x0, x1, x2, x3 = [h_leaf[..., i] for i in range(4)]
        # 6 个 pair
        pairs = torch.stack([x0*x1, x0*x2, x0*x3, x1*x2, x1*x3, x2*x3], dim=-1)
        quad = (pairs * self.leaf_quad).sum(dim=-1, keepdim=True)  # [B,64,3,1]
        return lin + quad

    def forward(self, leaf_in, bAP, prev_h_leaf, prev_h4):
        """
        leaf_in: [B,64,3,4]
        bAP:     [B,64]
        return:  [B,64]
        """
        B = leaf_in.size(0)
        # 0) 状态初始化
        if prev_h_leaf is None:
            prev_h_leaf = leaf_in.new_zeros(B, self.n_neuron, self.n_leaf, self.leaf_dim)
            prev_h4 = leaf_in.new_zeros(B, self.n_neuron, 1, 1)
        decay_leaf = torch.sigmoid(self.decay_leaf)
        decay_s4 = torch.sigmoid(self.decay_s4)
        # decay_s4 = torch.sigmoid(torch.clamp(self.decay_s4, -10, 10))
        bAP_decay_leaf = torch.tanh(self.bAP_decay_leaf)
        bAP_decay_s4 = torch.tanh(self.bAP_decay_s4)
        # 1) bAP
        bAP_4d = bAP.unsqueeze(2).unsqueeze(3) # [B,N,1,1]
        bAP_s4 = bAP_4d * bAP_decay_s4  # [B,64,1,1]
        bAP_s1 = bAP_s4 * bAP_decay_leaf[:, :, 0:1, :]
        bAP_s2 = bAP_s4 * bAP_decay_leaf[:, :, 1:2, :]
        bAP_s3 = bAP_4d * bAP_decay_leaf[:, :, 2:3, :]
        bAP_leaf = torch.cat([bAP_s1, bAP_s2, bAP_s3], dim=2)
        h_leaf_prev = prev_h_leaf
        h_leaf_bAP = h_leaf_prev + bAP_leaf  # [B,64,3,4]
        h4_prev = prev_h4
        h4_bAP = h4_prev + bAP_s4
        # 2) 叶子输入
        h_leaf_new = decay_leaf * h_leaf_bAP + (1 - decay_leaf) * leaf_in.unsqueeze(-1).unsqueeze(-1) # [B,64,3,4]
        # 3) 叶子非线性
        leaf_sc = self._leaf_bilinear(h_leaf_new)  # [B,64,3,1]
        s1 = leaf_sc[:, :, 0:1, :]  # [B,64,1,1]
        s2 = leaf_sc[:, :, 1:2, :]
        s3 = leaf_sc[:, :, 2:3, :]
        # 4) s4 更新
        u4 = self.w41 * s1 + self.w42 * s2  # [B,64,1,1]
        h4_new = decay_s4 * h4_bAP + (1 - decay_s4) * u4
        # 5) dendrite输出
        u_dendrite = self.ws4 * h4_new + self.ws3 * s3  # [B,64,1,1]
        out = u_dendrite.squeeze(-1).squeeze(-1) 
        return h_leaf_new, h4_new, out #[B,64]

# ====== 锥体神经元主模块：apical/oblique/basal -> soma ======
class BDLIF_nosparse(nn.Module):
    def __init__(self, neuron_num: int, project_dim: int, feedback_dim: int, ratio):
        super().__init__()
        # self.dist_apical = SparseDistributorToLeaf(in_dim=feedback_dim, n_neuron=neuron_num)
        # self.dist_basal = SparseDistributorToLeaf(in_dim=project_dim, n_neuron=neuron_num)
        self.dist_apical = nn.Linear(feedback_dim, neuron_num)
        self.dist_basal = nn.Linear(project_dim, neuron_num)
        self.apical = DendriteNode_Apical(n_neuron=neuron_num)
        self.basal = DendriteNode_Basal(n_neuron=neuron_num)
        # self.apical = DendriteNode(n_neuron=neuron_num)
        # self.basal = DendriteNode(n_neuron=neuron_num)
        self.soma = SomaNode(neuron_num)
        self.burst = Burst(ratio)
        self.wa = nn.Parameter(torch.ones(1, neuron_num))
        self.wb = nn.Parameter(torch.ones(1, neuron_num))
        self.leak = 0.6

    def forward(self, apical_input, basal_input, AP, h_leaf_apical, h_s4_apical,h_leaf_basal, h_s4_basal):
        apical_input_dist = self.dist_apical(apical_input)
        basal_input_dist = self.dist_basal(basal_input)
        h_leaf_apical, h_s4_apical, Ia = self.apical(apical_input_dist, h_leaf_apical, h_s4_apical)
        h_leaf_basal, h_s4_basal, Ib = self.basal(basal_input_dist, AP, h_leaf_basal, h_s4_basal)
        Is = self.wb*Ib + self.wa*Ia
        AP_raw = self.soma(Is)
        burst_fac = self.burst(Ia)
        burst = burst_fac.detach() + self.leak * (burst_fac - burst_fac.detach())
        AP_burst = AP_raw * burst
        return AP_burst, AP_raw, h_leaf_apical, h_s4_apical,h_leaf_basal, h_s4_basal

    @torch.no_grad()
    def reset_neuron(self, batch_size=None):
        self.soma.reset_state(batch_size=batch_size)
        
class BDLIF(nn.Module):
    def __init__(self, neuron_num: int, project_dim: int, feedback_dim: int, ratio):
        super().__init__()
        self.dist_apical = SparseDistributorToLeaf(in_dim=feedback_dim, n_neuron=neuron_num)
        self.dist_basal = SparseDistributorToLeaf(in_dim=project_dim, n_neuron=neuron_num)
        self.apical = DendriteNode(n_neuron=neuron_num)
        self.basal = DendriteNode(n_neuron=neuron_num)
        self.soma = SomaNode(neuron_num)
        self.burst = Burst(ratio)
        self.wa = nn.Parameter(torch.ones(1, neuron_num))
        self.wb = nn.Parameter(torch.ones(1, neuron_num))
        self.leak = 0.6

    def forward(self, apical_input, basal_input, AP, h_leaf_apical, h_s4_apical,h_leaf_basal, h_s4_basal):
        apical_input_dist = self.dist_apical(apical_input)
        basal_input_dist = self.dist_basal(basal_input)
        h_leaf_apical, h_s4_apical, Ia = self.apical(apical_input_dist, AP, h_leaf_apical, h_s4_apical)
        h_leaf_basal, h_s4_basal, Ib = self.basal(basal_input_dist, AP, h_leaf_basal, h_s4_basal)
        Is = self.wb*Ib + self.wa*Ia
        AP_raw = self.soma(Is)
        burst_fac = self.burst(Ia)
        burst = burst_fac.detach() + self.leak * (burst_fac - burst_fac.detach())
        AP_burst = AP_raw * burst
        return AP_burst, AP_raw, h_leaf_apical, h_s4_apical,h_leaf_basal, h_s4_basal

    @torch.no_grad()
    def reset_neuron(self, batch_size=None):
        self.soma.reset_state(batch_size=batch_size)
        

# ---------------- BDN：两层RSNN（down + top） ----------------
class BDN(nn.Module):
    def __init__(self, hidden_size=[64,32], in_dim=256, project_dim=256, out_dim=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        
        # [修改点] 定义 Input 降维旁路 (Input -> Apical Skip Connection)
        # 底层旁路：in_dim -> hidden_size[1] (Feedback 维度)
        self.input_skip_0 = nn.Linear(in_dim, hidden_size[1])
        # 顶层旁路：hidden_size[0] -> out_dim (Feedback 维度)
        self.input_skip_1 = nn.Linear(hidden_size[0], out_dim)

        self.project_0 = nn.Linear(in_dim, project_dim)
        self.hidden_0 = BDLIF(hidden_size[0], project_dim=project_dim, feedback_dim=hidden_size[1], ratio=0.1)
        self.feedback_0 = nn.Linear(hidden_size[1],hidden_size[1])

        self.project_1 = nn.Linear(hidden_size[0], hidden_size[0])
        self.hidden_1 = BDLIF(hidden_size[1], project_dim=hidden_size[0], feedback_dim=out_dim, ratio=1)
        self.project_2 = nn.Linear(hidden_size[1], out_dim) 
        self.output = SomaNode(out_dim)
        self.feedback_1 = nn.Linear(out_dim, out_dim)
        

    @torch.no_grad()
    def reset_net(self, batch_size):
        self.hidden_0.reset_neuron(batch_size)
        self.hidden_1.reset_neuron(batch_size)
        self.output.reset_state(batch_size)

    def forward(self, x: torch.Tensor):  # [B, T, input_dim]
        B, T, _ = x.shape
        self.reset_net(B)
        y_out_list = []
        prev_y_0 = x.new_zeros(B, self.hidden_size[0])
        prev_y_1 = x.new_zeros(B, self.hidden_size[1])
        prev_y_out = x.new_zeros(B, self.out_dim)
        h0_leaf_apical, h0_s4_apical,h0_leaf_basal, h0_s4_basal = None, None, None, None
        h1_leaf_apical, h1_s4_apical,h1_leaf_basal, h1_s4_basal = None, None, None, None
        
        for t in range(1, T + 1):
            # ================= Layer 0 =================
            x_t = x[:, t-1, :]
            basal_0_t = self.project_0(x_t)
            
            # [修改点] Apical Input = Feedback + 降维后的 Input Skip
            feedback_signal_0 = self.feedback_0(prev_y_1)
            skip_input_0 = self.input_skip_0(x_t)
            apical_0_t = feedback_signal_0 + skip_input_0 
            
            y_0_t, bAP_0_t, h0_leaf_apical, h0_s4_apical, h0_leaf_basal, h0_s4_basal = self.hidden_0(
                apical_0_t, basal_0_t, prev_y_0, 
                h0_leaf_apical, h0_s4_apical, h0_leaf_basal, h0_s4_basal
            )
            prev_y_0 = bAP_0_t
            
            # ================= Layer 1 =================
            basal_1_t = self.project_1(y_0_t)
            
            # [修改点] Apical Input = Feedback + 降维后的 Input Skip (Layer 1 的输入是 y_0_t)
            feedback_signal_1 = self.feedback_1(prev_y_out)
            skip_input_1 = self.input_skip_1(y_0_t)
            apical_1_t = feedback_signal_1 + skip_input_1
            
            y_1_t, bAP_1_t, h1_leaf_apical, h1_s4_apical, h1_leaf_basal, h1_s4_basal = self.hidden_1(
                apical_1_t, basal_1_t, prev_y_1, 
                h1_leaf_apical, h1_s4_apical, h1_leaf_basal, h1_s4_basal
            )
            prev_y_1, prev_y_out = bAP_1_t, bAP_1_t
            
            y_out_list.append(y_1_t)

        y_out_all = torch.stack(y_out_list, dim=1)
        # y_out = y_out_all.mean(dim=1) # 原代码注释掉的
        return y_out_all # [B,out_dim]


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------- 结构测试 ----------------
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device =', device)
    torch.manual_seed(0)
    # torch.backends.cudnn.benchmark = True  # 固定卷积要开
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    # torch.set_float32_matmul_precision('high')  # highest：纯FP32；high：FP32+TF32；medium：允许更低精度混合计

    B, T, input_dim = 4, 16, 256
    x = torch.randn(B, T, input_dim).to(device)
    net = BDN(hidden_size=[64,32], in_dim=input_dim, project_dim=256, out_dim=32).to(device)
    param = count_trainable_params(net)
    print("params: ", param)
    # net = torch.compile(net,mode="max-autotune")
    y = net(x)
    print(y.shape)
    # 再换一个 batch_size
    x2 = torch.randn(10, T, input_dim).cuda()
    y2 = net(x2)
    print( y2.shape)