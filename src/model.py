import torch
import torch.nn as nn
import torch.nn.functional as F


class NTKLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_w=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features)) 
        self.scale = sigma_w / in_features ** 0.5

    def forward(self, x):
        return self.scale * F.linear(x, self.weight)


class Square(nn.Module):
    def forward(self, x):
        return torch.square(x)


class FNN(nn.Module):
    def __init__(self, p, hidden):
        super().__init__()
        self.FN = nn.Sequential(
            NTKLinear(2 * p, hidden),
            Square(),
            NTKLinear(hidden, hidden),
            Square(),
            NTKLinear(hidden, p),
        )

    def forward(self, x):
        return self.FN(x)