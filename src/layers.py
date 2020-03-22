import torch
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self, num_channels, z_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.z_proj = nn.Linear(in_features=z_dim, out_features=num_channels * 2,)
        self.ins_norm = nn.InstanceNorm3d(num_features=num_channels, affine=False)

    def forward(self, x, z):
        scale, bias = torch.chunk(self.relu(self.z_proj(z)), chunks=2, dim=-1)
        x = self.ins_norm(x)
        x *= scale.view(*scale.size(), 1, 1, 1)
        x += bias.view(*bias.size(), 1, 1, 1)
        return x
