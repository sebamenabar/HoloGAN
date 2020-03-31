import torch
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self, num_channels, z_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.z_proj = nn.Linear(in_features=z_dim, out_features=num_channels * 2,)
        self.ins_norm = nn.InstanceNorm3d(num_features=num_channels, affine=False)
        self.init_params()

    def init_params(self):
        torch.nn.init.normal_(self.z_proj.weight, std=0.02)
        torch.nn.init.zeros_(self.z_proj.bias)

    def forward(self, x, z):
        scale, bias = torch.chunk(self.relu(self.z_proj(z)), chunks=2, dim=-1)
        x = self.ins_norm(x)
        x *= scale.view(*scale.size(), 1, 1, 1)
        x += bias.view(*bias.size(), 1, 1, 1)
        return x


class AdaIN2D(nn.Module):
    def __init__(self, num_channels, z_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.z_proj = nn.Linear(in_features=z_dim, out_features=num_channels * 2,)
        self.ins_norm = nn.InstanceNorm2d(num_features=num_channels, affine=False)
        self.init_params()

    def init_params(self):
        torch.nn.init.normal_(self.z_proj.weight, std=0.02)
        torch.nn.init.zeros_(self.z_proj.bias)

    def forward(self, x, z):
        scale, bias = torch.chunk(self.relu(self.z_proj(z)), chunks=2, dim=-1)
        x = self.ins_norm(x)
        x *= scale.view(*scale.size(), 1, 1)
        x += bias.view(*bias.size(), 1, 1)
        return x


class NeuralTensor(nn.Module):
    def __init__(self, in1_features, in2_features, out_features):
        super().__init__()
        self.bilinear = nn.Bilinear(in1_features, in2_features, out_features, bias=True)
        self.linear = nn.Linear(in1_features + in2_features, out_features, bias=False)

    def forward(self, x1, x2):
        return self.bilinear(x1, x2) + self.linear(torch.cat((x1, x2), -1))
