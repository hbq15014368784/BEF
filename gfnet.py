import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        # x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        # weight = torch.view_as_complex(self.complex_weight)
        # x = x * weight
        # x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x_numpy = x.detach().cpu().numpy()
        x_fft_numpy = np.fft.fft2(x_numpy, axes=(1, 2))

        real_weight = self.complex_weight[..., 0]
        imag_weight = self.complex_weight[..., 1]
        weight = torch.stack((real_weight, imag_weight), dim=-1).to(x.device)

        weight = weight.permute(2, 0, 1, 3)

        weight_numpy = weight.detach().cpu().numpy()

        x_fft_weighted_numpy = x_fft_numpy * weight_numpy

        # x_ifft_numpy = np.fft.ifft2(x_fft_weighted_numpy, axes=(1, 2))
        #
        # x_ifft_torch = torch.from_numpy(x_ifft_numpy).to(x.device)
        x_ifft_numpy = np.fft.ifft2(x_fft_weighted_numpy, axes=(1, 2)).astype(np.float32)
        print(x_ifft_numpy.shape)

        # x_ifft_torch = torch.from_numpy(x_ifft_numpy).to(x.device)
        #
        # x = x_ifft_torch.reshape(B, N, C)
        x_ifft_torch = torch.from_numpy(x_ifft_numpy).to(x.device)
        x = x_ifft_torch.view(B, -1)  # 将其展平为 (B, a*b)

        return x
