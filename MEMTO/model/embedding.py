import torch
import torch.nn as nn
import torch.nn.functional as F
from model.decor import DeCoR

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.pe = torch.zeros((max_len, d_model), dtype=torch.float)
        self.pe.requires_grad = False

        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2).float()

        self.pe[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

class TokenEmbedding(nn.Module):
    def __init__(self, in_dim, d_model):
        super(TokenEmbedding, self).__init__()
        
        
        self.decor = DeCoR(
            dim=in_dim,           # 输入维度
            kernel_size=5,        # 保持原有的kernel_size=3
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


class InputEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, device, dropout=0.0):
        super(InputEmbedding, self).__init__()
        self.device = device
        self.token_embedding = TokenEmbedding(in_dim=in_dim, d_model=d_model)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        
        try:
            x = self.token_embedding(x) + self.pos_embedding(x).cuda()
        except:
            import pdb; pdb.set_trace()
        return self.dropout(x)