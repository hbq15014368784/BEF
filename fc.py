from __future__ import print_function
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

import torch
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self,
                 input_dim,
                 dimensions,
                 activation='relu',
                 dropout=0.):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.dimensions = dimensions
        self.activation = activation
        self.dropout = dropout
        # Modules
        self.linears = nn.ModuleList([nn.Linear(input_dim, dimensions[0])])
        for din, dout in zip(dimensions[:-1], dimensions[1:]):
            self.linears.append(nn.Linear(din, dout))

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if (i < len(self.linears) - 1):
                x = nn.functional.__dict__[self.activation](x)
                if self.dropout > 0:
                    x = nn.functional.dropout(x, self.dropout, training=self.training)
        return x

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, drop=0.0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())
        self.drop_value = drop
        self.drop = nn.Dropout(drop)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)
        
        return self.main(x)

class KANLayer(nn.Module):
    """ A basic layer for KAN 2.0 Network that uses Kolmogorov-Arnold expression """

    def __init__(self, in_dim, out_dim, degree=3):
        super(KANLayer, self).__init__()

        # KA layer could use multiple monomials for higher-dimensional non-linearities
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree

        # Define weights for monomials
        self.weights = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)
        self.bias = nn.Parameter(torch.randn(out_dim) * 0.02)

    def forward(self, x):
        # A simple polynomial form f(x) = sum( c_i * x^degree_i )
        x_poly = x
        result = torch.matmul(x_poly, self.weights) + self.bias

        return F.relu(result)


class KAN2_0(nn.Module):
    """KAN 2.0 network inspired by Kolmogorov-Arnold expression"""

    def __init__(self, dims, drop=0.0, degree=3):
        super(KAN2_0, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            layers.append(KANLayer(in_dim, out_dim, degree))
            layers.append(nn.Dropout(drop))

        layers.append(KANLayer(dims[-2], dims[-1], degree))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

