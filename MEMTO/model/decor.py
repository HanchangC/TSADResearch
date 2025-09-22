import torch
from torch import nn
import torch.nn.functional as F

def efficient_linterpolate(
        x,
        offsets,
        kernel_size,
        dilation,
        stride,
        dilated_positions=None,
        device="cpu",
        _test=False,
        unconstrained=False
):
    assert x.device == offsets.device, "x and offsets must be on same device"
    kernel_rfield = dilation * (kernel_size - 1) + 1
    # Every index in x we need to consider
    if dilated_positions == None:
        dilated_positions = torch.linspace(0, kernel_rfield - 1, kernel_size, device=offsets.device,
                                           dtype=offsets.dtype)  # kernel_size

    max_t0 = (offsets.shape[-2] - 1) * stride
    t0s = torch.linspace(0, max_t0, offsets.shape[-2], device=offsets.device, dtype=offsets.dtype).unsqueeze(
        -1)  # out_length x 1
    dilated_offsets_repeated = dilated_positions + offsets

    T = t0s + dilated_offsets_repeated  # batch_size x channels x out_length x kernel_size
    if not unconstrained:
        T = torch.max(T, t0s)
        T = torch.min(T, t0s + torch.max(dilated_positions))
    else:
        T = torch.clamp(T, 0.0, float(x.shape[-1]))

    if _test:
        print("x:", x.shape)  # batch_size x in_channels x input_length
        print("offsets:", offsets.shape)  # batch_size x groups x out_length x kernel_size
        print("max_t0:", max_t0)
        print("t0s:", t0s.shape)  # out_lengths x 1
        print("dilated positions:", dilated_positions.shape)  # kernel_size
        print("dilated_offsets_repeated:", dilated_offsets_repeated.shape)
        print("T:", T.shape)  # batch_size x groups x out_length x kernel_rfield

    with torch.no_grad():
        U = torch.floor(T).to(torch.long)  # 1 x 1 x length x kernel_rfield
        U = torch.clamp(U, min=0, max=x.shape[2] - 2)

        if _test:
            print("U:", U.shape)

        U = torch.stack([U, U + 1], dim=-1)
        if U.shape[1] < x.shape[1]:
            U = U.repeat(1, x.shape[1], 1, 1, 1)
        if _test:
            print("U:", U.shape)

    x = x.unsqueeze(-1).repeat(1, 1, 1, U.shape[-1])
    x = torch.stack([x.gather(index=U[:, :, :, i, :], dim=-2) for i in range(U.shape[-2])], dim=-1)

    G = torch.max(torch.zeros(U.shape, device=device),
                  1 - torch.abs(U - T.unsqueeze(-1)))  # batch_size x groups x out_length x kernel_rfield x kernel_size

    if _test:
        print("G:", G.shape)

    mx = torch.multiply(G, x.moveaxis(-2, -1))

    return torch.sum(mx, axis=-1)  # .float()  # batch_size x channels x output_length x kernel size

class DeCoR(nn.Module):
    def __init__(self, dim, kernel_size, output_dim=None, groups=1, dilation=1, stride=1):
        super(DeCoR, self).__init__()
        self.c_in = dim
        self.kernel_size = kernel_size
        self.output_dim = dim if output_dim is None else output_dim
        self.weight = nn.Parameter(torch.randn(self.output_dim, dim//groups, kernel_size))
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=dilation*kernel_size, groups=groups, dilation=dilation, stride=stride,
                        padding='same', bias=False),
            nn.ReLU(),
            nn.Conv1d(dim, kernel_size, kernel_size*dilation, groups=groups,
                      padding='same', bias=False,stride=stride)
        )
        self.dilation = dilation
        self.stride = stride
        self.groups = groups

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')


    def forward(self, input_emb):
        # input_emb: batch_size x seq_length x dim
        hidden = torch.transpose(input_emb, 1,2)
        offset = torch.transpose(self.conv(hidden).unsqueeze(1), 2,3) # batch_size x groups x kernel_size x seq_length
        hidden = efficient_linterpolate(hidden, offset, self.kernel_size, self.dilation, self.stride, device=hidden.device)
        hidden = F.conv1d(hidden.flatten(-2, -1), self.weight, None, groups=self.groups,stride=self.kernel_size)
        return torch.transpose(hidden, 1,2) # batch_size x seq_length x dim
