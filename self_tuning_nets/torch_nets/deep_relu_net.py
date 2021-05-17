import torch
import torch.nn as nn


class DeepReluNet(torch.nn.Module):
    def __init__(self, input_dim: int, L: int, M: int):
        """L and M are theoretical parameters for Deep Relu networks
        coming from the work of Weinan et al. Available in:
        https://arxiv.org/pdf/1807.00297.pdf"""
        super(DeepReluNet, self).__init__()
        self.L = L
        depth = L + 1
        width = M + input_dim + 1

        self.input = nn.Linear(input_dim, width)
        self.hidden = nn.ModuleList()
        for _ in range(depth):
            self.hidden.append(nn.Linear(width, width))
        self.output = nn.Linear(width, 1)

    def forward(self, x):
        x = self.input(x)
        for hidden in self.hidden:
            x = hidden(x)
            x = torch.relu(x)
        return self.output(x)
