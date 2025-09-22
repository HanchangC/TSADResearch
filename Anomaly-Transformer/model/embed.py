import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


# class TokenEmbedding(nn.Module):
#     def __init__(self, c_in, d_model):
#         super(TokenEmbedding, self).__init__()
#         padding = 1 if torch.__version__ >= '1.5.0' else 2
#         self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
#                                    kernel_size=3, padding=padding, padding_mode='circular', bias=False)
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

#     def forward(self, x):
#         x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
#         return x

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        from model.decor import DeCoR
        
        self.decor = DeCoR(
            dim=c_in,           # 输入维度
            kernel_size=3,        # 保持原有的kernel_size=3
            output_dim=d_model,    # 输出维度与输入维度相同
            groups=1,             # 不分组
            dilation=1,           # 不使用膨胀
            stride=1              # 保持原有步长
        )
        
        # # 由于DeCoR保持输入输出维度相同，需要额外的投影层来改变维度
        # self.proj = nn.Linear(in_dim, d_model, bias=False)

    def forward(self, x):
        # DeCoR期望输入shape为: [batch_size, seq_length, dim]
        # 当前输入x的shape为: [batch_size, seq_length, in_dim]
        x = self.decor(x)        # [batch_size, seq_length, d_model]
        # x = self.proj(x)         # [batch_size, seq_length, d_model]
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
