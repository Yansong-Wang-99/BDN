import sys

import numpy as np
import torch
import torch.nn as nn
from numpy import prod
from torch.distributions.beta import Beta
import torch.nn.functional as F
import utils

from pathlib import Path
# 获取当前文件所在的目录
current_file_path = Path(__file__).resolve()
sys.path.append(str(current_file_path.parents[4]))
import BDNb

spike_args = {}
spike_args['thresh'] = utils.args.thresh
spike_args['lens'] = utils.args.lens
spike_args['decay'] = utils.args.decay


class Net(torch.nn.Module):
    def __init__(self, args, inputsize, taskcla, nlab, nhid=40, nlayers=3):
        super(Net, self).__init__()

        self.spike_window = args.spike_windows
        self.args = args
        ncha, size, size2 = inputsize
        self.taskcla = taskcla
        self.labsize = nlab
        self.layers = nn.ModuleList()
        self.nlayers = nlayers

        # fcs hidden layer
        self.fcs = SpikeBDN(args=self.args, in_dim=size * size2 * ncha, hidden_size=[nhid, nlab], out_dim=nlab, project_dim=size * size2 * ncha)

        return

    def forward(self, t, x, laby, e=-1):
        input_ = x.reshape(x.size(0), -1).clone()
        y = self.fcs(input_, laby)

        return y#, hidden_out

# ---------------- BDN：两层RSNN（down + top） ----------------
class SpikeBDN(nn.Module):
    def __init__(self, args=None, hidden_size=[64,32], in_dim=256, project_dim=256, out_dim=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.spike_window = args.spike_windows
        self.NI = torch.empty(out_dim, hidden_size[0]).cuda()
        self.NI = reset_weights_NI(self.NI)
        self.NI.requires_grad = False
        
        # [修改点] 定义 Input 降维旁路 (Input -> Apical Skip Connection)
        # 底层旁路：in_dim -> hidden_size[1] (Feedback 维度)
        self.input_skip_0 = nn.Linear(in_dim, hidden_size[1])
        # 顶层旁路：hidden_size[0] -> out_dim (Feedback 维度)
        self.input_skip_1 = nn.Linear(hidden_size[0], out_dim)

        self.project_0 = nn.Linear(in_dim, project_dim)
        self.hidden_0 = BDNb.BDLIF_nosparse(hidden_size[0], project_dim=project_dim, feedback_dim=hidden_size[1], ratio=0.1)
        self.feedback_0 = nn.Linear(hidden_size[1],hidden_size[1])

        self.project_1 = nn.Linear(hidden_size[0], hidden_size[0])
        self.hidden_1 = BDNb.BDLIF_nosparse(hidden_size[1], project_dim=hidden_size[0], feedback_dim=out_dim, ratio=0.05)
        self.project_2 = nn.Linear(hidden_size[1], out_dim) 
        self.output = BDNb.SomaNode(out_dim)
        self.feedback_1 = nn.Linear(out_dim, out_dim)
        

    @torch.no_grad()
    def reset_net(self, batch_size):
        self.hidden_0.reset_neuron(batch_size)
        self.hidden_1.reset_neuron(batch_size)
        self.output.reset_state(batch_size)

    def forward(self, x: torch.Tensor, y):  # [B, T, input_dim]
        B, _ = x.shape
        T = self.spike_window
        self.reset_net(B)
        y_out_list = []
        prev_y_0 = x.new_zeros(B, self.hidden_size[0])
        prev_y_1 = x.new_zeros(B, self.hidden_size[1])
        h0_leaf_apical, h0_s4_apical,h0_leaf_basal, h0_s4_basal = None, None, None, None

        for t in range(1, T + 1):
            # ================= Layer 0 =================
            x_t = x # 每个时刻都是一样的图像
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
        
            # mask jsc
            u_mask = torch.mean(x_t, 0, False)
            u_mask = F.interpolate(u_mask.unsqueeze(0).unsqueeze(0), size=[self.hidden_size[0]])
            u_mask = u_mask.squeeze(0)
            if utils.bias[utils.args.experiment] is not None:
                bias = utils.bias[utils.args.experiment]  # bias is the threshold, u_mask may be zeros
            else:
                bias = u_mask.max() - utils.delta_bias[utils.args.experiment]  # ensure the u_mask is not zeros
                # bias = 0

            u_mask = (u_mask > bias).float()
            y_0_t = y_0_t * u_mask.expand_as(y_0_t)
            
            # ================= Layer 1 =================

            skip_input_1 = self.input_skip_1(y_0_t)
            apical_1_t = skip_input_1 #+ feedback_signal_1

            y_1_t = apical_1_t

            y_out_list.append(y_1_t)

        y_out_all = torch.stack(y_out_list, dim=1).mean(1)

        if y is not None:

            mod_gradient = y.mm(self.NI.view(-1, prod(self.NI.shape[1:]))).view(y_0_t.shape)

            if torch.isnan(mod_gradient).any() or torch.isinf(mod_gradient).any():
                mod_gradient = torch.nan_to_num(mod_gradient, nan=0.0, posinf=1.0, neginf=-1.0)

            y_0_t.backward(gradient=mod_gradient, retain_graph=True)

            err = (F.softmax(y_out_all, dim=1) - y) / T
            y_1_t.backward(gradient=err, retain_graph=True)

        return y_out_all # [B,out_dim]
    
class SpikeLinear(torch.nn.Module):
    def __init__(self, args, in_features, out_features, nlab, layer=None):
        super(SpikeLinear, self).__init__()
        self.args = args
        if layer != -1:
            self.fc = torch.nn.Linear(in_features, out_features, bias=False)
            self.NI = torch.empty(2 * nlab, out_features).cuda()
            self.NI = reset_weights_NI(self.NI)
            self.NI.requires_grad = False
        else:
            self.fc = torch.nn.Linear(in_features, out_features, bias=True)
        self.in_features = in_features
        self.out_features = out_features
        self.nlab = nlab
        self.mem = None
        self.spike = None
        self.sumspike = None
        self.time_counter = 0
        self.layer = layer
        self.spike_window = args.spike_windows
        self.old_spike = None

    # t: current task, x: input, y: output
    def forward(self, x, y, input_):
        if self.time_counter == 0:
            batchsize = x.shape[0]
            self.mem = torch.zeros((batchsize, self.out_features)).cuda()
            self.spike = torch.zeros((batchsize, self.out_features)).cuda()
            self.sumspike = torch.zeros((batchsize, self.out_features)).cuda()
            self.old_spike = torch.zeros((batchsize, self.out_features)).cuda()
        self.time_counter += 1
        self.mem, self.spike = mem_update(self.fc, x, self.mem, self.spike, self.old_spike)
        if self.layer != -1:
            u_mask = torch.mean(input_, 0, False)
            u_mask = F.interpolate(u_mask.unsqueeze(0).unsqueeze(0), size=[self.out_features])
            u_mask = u_mask.squeeze(0)
            if utils.bias[utils.args.experiment] is not None:
                bias = utils.bias[utils.args.experiment]  # bias is the threshold, u_mask may be zeros
            else:
                bias = u_mask.max() - utils.delta_bias[utils.args.experiment]  # ensure the u_mask is not zeros
            u_mask = torch.sigmoid(1000 * (u_mask - bias))
            self.spike = self.spike * u_mask.expand_as(self.spike)
        self.sumspike += self.spike

        self.old_spike = self.spike.clone().detach()

        # y=None for inference
        if self.time_counter == self.spike_window:
            self.time_counter = 0
            if y is not None:
                # Hidden layers
                if self.layer != -1:
                    neuromodulator_level = expectation(y).mm(self.NI.view(-1, prod(self.NI.shape[1:]))).view(self.spike.shape)
                    self.spike.backward(gradient=local_modulation(neuromodulator_level), retain_graph=True)

                # Output layers
                else:
                    # MSE
                    err = (self.sumspike / self.spike_window) - y
                    err = torch.matmul(err, torch.eye(err.shape[1]).to(err.device))
                    self.spike.backward(gradient=err, retain_graph=True)

        return self.spike


def expectation(labels):
    sigma = 1
    delta_mu = 1
    max_len = labels.shape[1]
    a = np.array([np.sqrt(2) * np.sqrt(np.log(max_len / i)) * sigma for i in range(1, max_len + 1)])
    a = a / a.max() * (2 * (max_len - delta_mu))
    b = delta_mu + a
    a = torch.tensor(a.astype('int')).to(labels.device)
    assert len(set(a.cpu().numpy().tolist())) == len(a.cpu().numpy().tolist()), 'error in expectation'
    b = torch.tensor(b.astype('int')).to(labels.device)
    Ea = a[torch.max(labels, 1)[1].cpu()]
    Eb = b[torch.max(labels, 1)[1].cpu()]
    Ea = torch.zeros(labels.shape[0], 2 * labels.shape[1], device=labels.device).scatter_(1, Ea.unsqueeze(1).long(), 1.0)
    Eb = torch.zeros(labels.shape[0], 2 * labels.shape[1], device=labels.device).scatter_(1, Eb.unsqueeze(1).long(), 1.0)
    return (Ea + Eb) / 2


def reset_weights_NI(NI):
    if utils.args.distribution == 'uniform':
        torch.nn.init.uniform_(NI, a=0.0, b=0.2)
    elif utils.args.distribution == 'normal':
        torch.nn.init.normal_(NI, mean=0.5, std=1)
        NI.clamp_(0, 1)
    elif utils.args.distribution == 'beta':
        dist = Beta(torch.ones_like(NI) * 0.5, torch.ones_like(NI) * 0.5)
        NI.data = dist.sample()
    elif utils.args.distribution == 'binary_tree':
        binary_tree_init_long_root(NI)
    return NI * 0.1

def binary_tree_init(tensor, mean=0.0, spread=1.0):
    """
    稀疏二叉树结构初始化 (Masked Random):
    - 结构：短边索引 i 对应二叉树节点，控制长边的一段区间（Mask）。
      i=0 (Root) -> [0, L]
      i=1 (Left) -> [0, L/2]
      i=2 (Right)-> [L/2, L]
      ...
    - 数值：在 Mask 为 1 的位置，保留随机初始化的值（-1到1）。
      0 表示无连接。
    """
    with torch.no_grad():
        # 1. 初始化为随机噪声 (满足均值和范围要求)
        # 假设 spread 是范围边界，即 [-spread + mean, spread + mean]
        lower = mean - spread
        upper = mean + spread
        tensor.uniform_(lower, upper)
        
        # 2. 构建二叉树 Mask
        mask = torch.zeros_like(tensor)
        rows, cols = tensor.shape
        
        if rows <= cols:
            short_dim, long_dim = rows, cols
            is_transposed = False
        else:
            short_dim, long_dim = cols, rows
            is_transposed = True
            
        # BFS 队列: (range_start, range_end)
        # 初始状态: 整个区间 [0, long_dim)
        range_queue = [(0, long_dim)]
        
        for i in range(short_dim):
            if not range_queue:
                # 如果短边比树节点多，重置队列，循环利用长边空间
                range_queue = [(0, long_dim)]
            
            r_start, r_end = range_queue.pop(0)
            
            # 设置 Mask
            # 只有在有效范围内才设置
            if r_end > r_start:
                if is_transposed:
                    # [Long, Short] -> [range, i]
                    # 注意：这里我们将整个区间 [r_start, r_end] 设为 1
                    # 这意味着第 i 个短边节点连接到长边的 [r_start, r_end] 所有节点
                    mask[r_start:r_end, i] = 1.0
                else:
                    # [Short, Long] -> [i, range]
                    mask[i, r_start:r_end] = 1.0
                
                # 加入子节点
                mid = (r_start + r_end) // 2
                # 左子区间
                range_queue.append((r_start, mid))
                # 右子区间
                range_queue.append((mid, r_end))
        
        # 3. 应用 Mask
        tensor.mul_(mask)

def binary_tree_init_long_root(tensor, mean=0.0, spread=1.0):
    """
    稀疏二叉树结构初始化 (Long Side as Root):
    - 结构：长边索引 i 对应二叉树节点，控制短边的一段区间（Mask）。
      i=0 (Root, Long Dim) -> [0, S] (Short Dim)
      i=1 (Left, Long Dim) -> [0, S/2]
      i=2 (Right, Long Dim)-> [S/2, S]
      ...
    - 数值：在 Mask 为 1 的位置，保留随机初始化的值。
      0 表示无连接。
    """
    with torch.no_grad():
        # 1. 初始化为随机噪声
        lower = mean - spread
        upper = mean + spread
        tensor.uniform_(lower, upper)
        
        # 2. 构建二叉树 Mask
        mask = torch.zeros_like(tensor)
        rows, cols = tensor.shape
        
        if rows <= cols:
            short_dim, long_dim = rows, cols
            is_transposed = False
        else:
            short_dim, long_dim = cols, rows
            is_transposed = True
            
        # BFS 队列: (range_start, range_end)
        # 初始状态: 整个短边区间 [0, short_dim)
        range_queue = [(0, short_dim)]
        
        # 遍历长边的每一个索引 i (作为树节点)
        for i in range(long_dim):
            if not range_queue:
                # 如果长边比树节点多，重置队列，循环利用短边空间
                range_queue = [(0, short_dim)]
            
            r_start, r_end = range_queue.pop(0)
            
            # 设置 Mask
            if r_end > r_start:
                if is_transposed:
                    # [Long, Short] -> [i, range]
                    # 长边索引 i 控制短边区间 [r_start, r_end]
                    mask[i, r_start:r_end] = 1.0
                else:
                    # [Short, Long] -> [range, i]
                    # 长边索引 i 控制短边区间 [r_start, r_end]
                    mask[r_start:r_end, i] = 1.0
                
                # 加入子节点
                mid = (r_start + r_end) // 2
                range_queue.append((r_start, mid))
                range_queue.append((mid, r_end))
        
        # 3. 应用 Mask
        tensor.mul_(mask)
                

def local_modulation(neuromodulator_level):
    lambda_inv = utils.args.lambda_inv
    theta_max = utils.args.theta_max
    with torch.no_grad():
        nl_ = neuromodulator_level.clone().detach()
        modulation = torch.zeros_like(neuromodulator_level).cuda()
        phase_one = theta_max - (theta_max - 1) * (4 * nl_ - lambda_inv).pow(2) / lambda_inv**2
        phase_two = 4 * (nl_ - lambda_inv).pow(2) / lambda_inv**2
        phase_three = -4 * ((2 * lambda_inv - nl_) - lambda_inv).pow(2) / lambda_inv**2
        phase_four = (theta_max - 1) * (4 * (2 * lambda_inv - nl_) - lambda_inv).pow(2) / lambda_inv**2 - theta_max

        modulation[neuromodulator_level <= 0.5 * lambda_inv] = phase_one[neuromodulator_level <= 0.5 * lambda_inv]
        modulation[(0.5 * lambda_inv < neuromodulator_level) & (neuromodulator_level <= lambda_inv)] = phase_two[(0.5 * lambda_inv < neuromodulator_level) & (neuromodulator_level <= lambda_inv)]
        modulation[(lambda_inv < neuromodulator_level) & (neuromodulator_level <= 1.5 * lambda_inv)] = phase_three[(lambda_inv < neuromodulator_level) & (neuromodulator_level <= 1.5 * lambda_inv)]
        modulation[1.5 * lambda_inv < neuromodulator_level] = phase_four[1.5 * lambda_inv < neuromodulator_level]

    return modulation


# simplified STDP
class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, old_spike):
        ctx.save_for_backward(input, old_spike)
        result = input.gt(spike_args['thresh']).float()
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, old_spike = ctx.saved_tensors
        grad_input = grad_output.clone()
        new_spike = input > spike_args['thresh'] - spike_args['lens']
        delta_spike = new_spike.float() - old_spike

        return grad_input * delta_spike, None, None

act_fun = ActFun.apply

# Membrane potential (non-linear sigmoid before the neuron for better performance)
def mem_update(ops, x, mem, spike, old_spike, drop=None, lateral=None):
    if drop is None:
        mem = mem * spike_args['decay'] * (1. - spike) + torch.sigmoid(ops(x))
    else:
        mem = mem * spike_args['decay'] * (1. - spike) + torch.sigmoid(drop(ops(x)))

    if lateral:
        mem += lateral(spike)
    spike = act_fun(mem, old_spike)
    return mem, spike
