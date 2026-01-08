import torch
import torch.nn as nn
from bdn_bilinear import BDN

import math
"""
Parameters for SNN
"""

ENCODER_REGULAR_VTH = 0.999
NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2
# NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5

#群编码有参数 所以需要写伪BP
class PseudoEncoderSpikeRegular(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Regular Spike for encoder """
    @staticmethod
    def forward(ctx, input):
        return input.gt(ENCODER_REGULAR_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class PseudoEncoderSpikeNone(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Poisson Spike for encoder """
    @staticmethod
    def forward(ctx, input):  #input (batch_size,obs_dim*pop_dim)
        return input
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class PopSpikeEncoderRegularSpike(nn.Module):
    """ Learnable Population Coding Spike Encoder with Regular Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoEncoderSpikeRegular.apply
        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        if pop_dim>1:
            delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
            for num in range(pop_dim):
                tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        elif pop_dim==1:
            tmp_mean[0, :, 0] = 0
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std
        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)  #均值和标准差是参数 为了让高斯分布输出在0-1（e^(-x),x>=0）之间，前面没有加系数
        pop_volt = torch.zeros(batch_size, self.encoder_neuron_num, device=self.device)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        # Generate Regular Spike Trains
        for step in range(self.spike_ts):
            pop_volt = pop_volt + pop_act
            pop_spikes[:, :, step] = self.pseudo_spike(pop_volt)
            pop_volt = pop_volt - pop_spikes[:, :, step] * ENCODER_REGULAR_VTH
        return pop_spikes

class PopSpikeEncoderNoneSpike(PopSpikeEncoderRegularSpike):
    """ Learnable Population Coding Spike Encoder with Poisson Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__(obs_dim, pop_dim, spike_ts, mean_range, std, device)
        self.pseudo_spike = PseudoEncoderSpikeNone.apply

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)  #(batch_size,obs_dim,1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num) #(batch_size,obs_dim*pop_dim)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device) #(batch_size,obs_dim*pop_dim,spike_ts)
        # Generate Poisson Spike Trains
        for step in range(self.spike_ts):
            pop_spikes[:, :, step] = self.pseudo_spike(pop_act)
        return pop_spikes

class PopSpikeDecoder(nn.Module):
    """ Population Coding Spike Decoder """
    def __init__(self, act_dim, pop_dim, output_activation=nn.Tanh):
        """
        :param act_dim: action dimension
        :param pop_dim:  population dimension
        :param output_activation: activation function added on output
        """
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.decoder = nn.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
        self.output_activation = output_activation()

    def forward(self, pop_act):
        """
        :param pop_act: output population activity
        :return: raw_act
        """
        pop_act = pop_act.view(-1, self.act_dim, self.pop_dim)
        raw_act = self.output_activation(self.decoder(pop_act).view(-1, self.act_dim))
        return raw_act


class PseudoSpikeRect(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Derivative of Rect Function """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()


class SpikeMLP(nn.Module):
    """ 仅支持 BDN 作为主干网络的 Spike MLP """
    def __init__(self, in_pop_dim, out_pop_dim, hidden_sizes, spike_ts, device):
        """
        :param in_pop_dim: 输入群体神经元维度
        :param out_pop_dim: 输出群体神经元维度
        :param hidden_sizes: [num_down, num_top]
        :param spike_ts: 脉冲时间步
        :param device: 设备
        """
        super().__init__()
        self.in_pop_dim = in_pop_dim
        self.out_pop_dim = out_pop_dim
        self.hidden_sizes = hidden_sizes   # [num_down, num_top]
        self.spike_ts = spike_ts
        self.device = device

        # ---- 输入线性投影 ----
        # 把输入维度映射到 BDN 的 down 层大小
        self.input_proj = nn.Linear(in_pop_dim, hidden_sizes).to(device)

        # ---- BDN 主干 ----
        self.bdn = BDN(hidden_size=[64, 32],
                             in_dim=256,
                             project_dim=256,
                             out_dim=32).to(device)


        # ---- 输出层 ----
        self.out_pop_layer = nn.Linear(32, out_pop_dim)

    def forward(self, in_pop_spikes, batch_size):
        """
        :param in_pop_spikes: 输入脉冲 [B, in_pop_dim, T]
        :param batch_size: 批大小
        :return: 输出群体活动 [B, out_pop_dim]
        """
        self.bdn.reset_net(batch_size)
        # ---- 调整维度 [B, in_pop_dim, T] -> [B, T, in_pop_dim] ----
        in_seq = in_pop_spikes.permute(0, 2, 1)

        # ---- 输入投影 [B, T, in_pop_dim] -> [B, T, num_down] ----
        x_proj = self.input_proj(in_seq)

        # ---- 输入 BDN ----
        y_top = self.bdn(x_proj)   # [B, T, num_down], [B, T, num_top]

        # ---- 输出层 ----
        out_pop_act = self.out_pop_layer(y_top)  # [B, out_pop_dim]

        return out_pop_act


class SpikeActor(nn.Module):
    """ Population Coding Spike Actor with Fix Encoder （只支持 BDN）"""
    def __init__(self, env,obs_dim, act_dim, en_pop_dim, de_pop_dim, hidden_sizes,
                 mean_range, std, spike_ts,  device):
        """
        :param obs_dim: 观测维度
        :param act_dim: 动作维度
        :param en_pop_dim: 编码器群体神经元维度
        :param de_pop_dim: 解码器群体神经元维度
        :param hidden_sizes: BDN的输入
        :param mean_range: 编码器均值范围
        :param std: 编码器高斯 std
        :param spike_ts: 脉冲时间步
        :param act_limit: 动作范围限制
        :param device: 设备

        """
        super().__init__()
        # self.act_limit = act_limit

        # ---- 编码器 ----
        self.encoder = PopSpikeEncoderNoneSpike(
            obs_dim, en_pop_dim, spike_ts, mean_range, std, device
        )

        # ---- SNN 主体 (BDN) ----
        self.snn = SpikeMLP(
            obs_dim * en_pop_dim,    # 输入维度
            act_dim * de_pop_dim,    # 输出维度
            hidden_sizes,
            spike_ts,
            device
        )

        # ---- 解码器 ----
        self.decoder = PopSpikeDecoder(act_dim, de_pop_dim)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        # 测试请将上面注释，self.register_buffer和self.register_buffer使用下面的代码
        # action_space = getattr(env, "single_action_space", env.action_space)
        # self.register_buffer(
        #     "action_scale",
        #     torch.as_tensor(
        #         (action_space.high - action_space.low) / 2.0, device=device
        #     )
        # )
        # self.register_buffer(
        #     "action_bias",
        #     torch.as_tensor(
        #         (action_space.high + action_space.low) / 2.0, device=device
        #     )
        # )
    def forward(self, obs, batch_size=None):
        """
        :param obs: 输入观测 [B, obs_dim]
        :param batch_size: 批大小
        :return: 动作 [B, act_dim]
        """

        if batch_size is None:
            batch_size = obs.shape[0]

        # 编码 -> 脉冲序列
        in_pop_spikes = self.encoder(obs, batch_size)

        # SNN 主干 -> 群体神经元活动
        out_pop_activity = self.snn(in_pop_spikes, batch_size)

        # 解码 -> 最终动作
        action = self.decoder(out_pop_activity) * self.action_scale + self.action_bias

        return action

