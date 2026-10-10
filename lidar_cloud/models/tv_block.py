"""Temporal-vertical convolution block, with original parameter names."""
import torch
from torch import nn

class TVBlock(nn.Module):

    def __init__(self, cin, cout, kT=3, kZ=7, dilZ=1, use_bn=True, activation='gelu', dropout=0.0):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.conv_t = nn.Conv2d(cin, cout, kernel_size=(kT, 1), padding=(kT // 2, 0), bias=not use_bn)
        padZ = kZ // 2 * dilZ
        self.conv_z = nn.Conv2d(cout, cout, kernel_size=(1, kZ), padding=(0, padZ), dilation=(1, dilZ), bias=not use_bn)
        self.bn_t = nn.BatchNorm2d(cout) if use_bn else nn.Identity()
        self.bn_z = nn.BatchNorm2d(cout) if use_bn else nn.Identity()
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f'Unknown activation: {activation}')
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.use_res = cin == cout

    def forward(self, x):
        identity = x if self.use_res else None
        out = self.conv_t(x)
        out = self.bn_t(out)
        out = self.act(out)
        out = self.conv_z(out)
        out = self.bn_z(out)
        out = self.dropout(out)
        if self.use_res:
            out = out + identity
        out = self.act(out)
        return out
