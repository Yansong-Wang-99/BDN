import torch
import torch.nn as nn
import math
import sys
sys.path.append('/mnt/d/BDN/code')
from BDN import BDN

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout):
        super().__init__()
        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # 前馈网络（替换为ANN结构）
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU()

        # 标准化和Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.BDN = BDN(hidden_size=[64,32], in_dim=dim_feedforward, project_dim=256, out_dim=dim_feedforward)

    def forward(self, src):
        # 自注意力部分
        src2, _ = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout(src2))

        # 前馈网络部分
        temp = self.BDN(self.linear1(src))
        # temp = self.activation(self.linear1(src))

        src2 = self.linear2(self.dropout(temp))
        src = self.norm2(src + self.dropout(src2))
        return src

class EncoderOnlyTransformer(nn.Module):
    def __init__(self, input_dim, output_size, d_model, num_heads,
                 num_encoder_layers, dim_feedforward, max_seq_length, dropout):
        super().__init__()
        self.d_model = d_model

        # 输入嵌入
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)

        # 编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(d_model, output_size)

    def forward(self, x):
        # 输入形状: (batch_size, seq_len, input_dim)

        x = self.pos_encoder(self.embedding(x))

        # 编码器处理
        for layer in self.encoder_layers:
            # print(x.shape)
            x = layer(x)
        # 输出处理（取序列最后一个时间步）
        out = self.output_layer(x)
        return out