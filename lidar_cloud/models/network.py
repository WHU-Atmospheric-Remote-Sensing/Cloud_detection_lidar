"""Inference architecture; the optional R1 head enables weight reuse."""
import torch
from torch import nn
from .backbone import ZBackboneV3

class SegHead(nn.Module):

    def __init__(self, in_ch, out_ch=1, bg_bias=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.bg_bias = nn.Parameter(torch.tensor(bg_bias))

    def forward(self, feat):
        logits = self.conv(feat) - self.bg_bias
        logits = logits.permute(0, 2, 3, 1).contiguous()
        return logits

class ReconHead(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, feat):
        x_rec = self.conv(feat)
        x_rec = x_rec.permute(0, 2, 3, 1).contiguous()
        return x_rec

class ZMultiTaskNetL(nn.Module):

    def __init__(self, in_ch=5, base_ch=48, kZ=5, n_recon_ch=0, n_seg_out=1, bg_bias=0.2, dropout=0.0):
        super().__init__()
        self.in_ch = in_ch
        self.n_recon_ch = n_recon_ch
        self.n_seg_out = n_seg_out
        self.backbone = ZBackboneV3(in_ch=in_ch, base_ch=base_ch, kZ=kZ, dropout=dropout)
        self.seg_head = SegHead(in_ch=self.backbone.out_ch, out_ch=n_seg_out, bg_bias=bg_bias)
        if n_recon_ch > 0:
            self.recon_head = ReconHead(in_ch=self.backbone.out_ch, out_ch=n_recon_ch)
        else:
            self.recon_head = None

    def forward(self, x, task='seg'):
        feat = self.backbone(x)
        if task == 'seg':
            return self.seg_head(feat)
        elif task == 'recon':
            assert self.recon_head is not None, 'ReconHead not initialized!'
            return self.recon_head(feat)
        elif task == 'both':
            seg = self.seg_head(feat)
            rec = self.recon_head(feat) if self.recon_head is not None else None
            return (seg, rec)
        else:
            raise ValueError(f"Unknown task: {task}. Use 'seg', 'recon', or 'both'.")
