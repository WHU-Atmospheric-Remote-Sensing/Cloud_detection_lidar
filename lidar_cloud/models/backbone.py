"""Frozen large temporal-vertical U-Net architecture."""
import torch
from torch import nn
from torch.nn import functional as F
from .tv_block import TVBlock

class TVBlockStack(nn.Module):

    def __init__(self, cin, cout, kT=3, kZ=5, dil1=1, dil2=1, dropout=0.0):
        super().__init__()
        self.b1 = TVBlock(cin, cout, kT=kT, kZ=kZ, dilZ=dil1, dropout=dropout)
        self.b2 = TVBlock(cout, cout, kT=kT, kZ=kZ, dilZ=dil2, dropout=dropout)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        return x

class DownZ(nn.Module):

    def __init__(self, kZ=2, mode='max'):
        super().__init__()
        if mode == 'max':
            self.pool = nn.MaxPool2d(kernel_size=(1, kZ), stride=(1, kZ))
        elif mode == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=(1, kZ), stride=(1, kZ))
        else:
            raise ValueError(f'Unknown mode: {mode}')

    def forward(self, x):
        return self.pool(x)

class UpZ(nn.Module):

    def __init__(self, c_dec, c_skip, cout, scale=2, kT=3, kZ=5, use_gating=True):
        super().__init__()
        self.scale = scale
        self.use_gating = use_gating
        self.up = nn.Upsample(scale_factor=(1, scale), mode='bilinear', align_corners=False)
        kZ_gate = 5
        padZ_gate = kZ_gate // 2
        if use_gating:
            self.gate = nn.Sequential(nn.Conv2d(c_skip, c_skip, kernel_size=(1, kZ_gate), padding=(0, padZ_gate), bias=True), nn.Sigmoid())
        else:
            self.gate = None
        self.fuse = TVBlock(c_dec + c_skip, cout, kT=kT, kZ=kZ, dilZ=1)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            if diff > 0:
                x = F.pad(x, (0, diff))
            else:
                x = x[..., :skip.shape[-1]]
        if self.use_gating and self.gate is not None:
            gate = self.gate(skip)
            skip_gated = skip * gate
        else:
            skip_gated = skip
        x = torch.cat([x, skip_gated], dim=1)
        out = self.fuse(x)
        return out

class ZBackboneV3(nn.Module):

    def __init__(self, in_ch=5, base_ch=48, kZ=5, depth=3, use_gating=True, dropout=0.0):
        super().__init__()
        self.in_ch = in_ch
        self.depth = depth
        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 8
        self.stem = TVBlockStack(in_ch, c1, kT=3, kZ=3, dil1=1, dil2=1, dropout=dropout)
        self.down1 = DownZ(kZ=2, mode='max')
        self.enc2 = TVBlockStack(c1, c2, kT=3, kZ=kZ, dil1=1, dil2=1, dropout=dropout)
        self.down2 = DownZ(kZ=2, mode='max')
        self.enc3 = TVBlockStack(c2, c3, kT=3, kZ=kZ, dil1=1, dil2=2, dropout=dropout)
        self.down3 = DownZ(kZ=2, mode='max')
        self.bott = TVBlockStack(c3, c4, kT=3, kZ=kZ, dil1=2, dil2=4, dropout=dropout)
        self.up3 = UpZ(c_dec=c4, c_skip=c3, cout=c3, scale=2, kT=3, kZ=kZ, use_gating=use_gating)
        self.dec3 = TVBlockStack(c3, c3, kT=3, kZ=kZ, dil1=1, dil2=2, dropout=dropout)
        self.up2 = UpZ(c_dec=c3, c_skip=c2, cout=c2, scale=2, kT=3, kZ=kZ, use_gating=use_gating)
        self.dec2 = TVBlockStack(c2, c2, kT=3, kZ=kZ, dil1=1, dil2=1, dropout=dropout)
        self.up1 = UpZ(c_dec=c2, c_skip=c1, cout=c1, scale=2, kT=3, kZ=3, use_gating=use_gating)
        self.dec1 = TVBlockStack(c1, c1, kT=3, kZ=3, dil1=1, dil2=1, dropout=dropout)
        self.out_ch = c1

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        s1 = self.stem(x)
        d2_in = self.down1(s1)
        e2 = self.enc2(d2_in)
        d3_in = self.down2(e2)
        e3 = self.enc3(d3_in)
        d4_in = self.down3(e3)
        b = self.bott(d4_in)
        u3 = self.up3(b, e3)
        u3 = self.dec3(u3)
        u2 = self.up2(u3, e2)
        u2 = self.dec2(u2)
        u1 = self.up1(u2, s1)
        u1 = self.dec1(u1)
        return u1
