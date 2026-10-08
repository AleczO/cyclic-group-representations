import torch
import torch.nn as nn
import torch.nn.functional as F


class Square(nn.Module):
    def forward(self, x):
        return torch.square(x)


class FNN(nn.Module):
    def __init__(self, d, hidden=256, out_dim=None):
        super().__init__()
        out_dim = out_dim or d
        self.FN = nn.Sequential(
            nn.Linear(d, hidden),     
            Square(),
            nn.Linear(hidden, hidden),
            Square(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):              
        logits = self.FN(x)            
        return logits