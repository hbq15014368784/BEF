import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


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

        # # Apply polynomial transformation and non-linearity
        # for _ in range(self.degree - 1):  # Apply degree-1 polynomial transform
        #     print(f"in_dim : {self.in_dim}")
        #     print(f"out_dim : {self.out_dim}")
        #     print(f"Input result shape: {result.shape}")
        #     print(f"Weight shape: {self.weights.shape}")
        #
        #     result = torch.matmul(result, self.weights) + self.bias

        return F.relu(result)


class KAN2_0Classifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout, degree=3):
        super(KAN2_0Classifier, self).__init__()
        layers = [
            KANLayer(in_dim, hid_dim, degree),  # Use KANLayer instead of nn.Linear
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            KANLayer(hid_dim, out_dim, degree)  # Use KANLayer instead of nn.Linear
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits
