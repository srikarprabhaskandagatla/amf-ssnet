import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


class BoundaryRefine(nn.Module):
    def __init__(self, channels, wave="haar"):
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wave, mode="periodization")
        self.proc = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 1e-5))

    def forward(self, x):
        _, yh = self.dwt(x)
        high = yh[0]
        B, C, _, h, w = high.shape
        e = self.proc(high.reshape(B, C * 3, h, w))
        e = F.interpolate(e, size=x.shape[-2:], mode="bilinear",
                          align_corners=False)
        return x + self.gamma * self.gate(x) * e


class BoundaryDecoder(nn.Module):
    def __init__(self, channels, stages=("up3", "up4"), wave="haar"):
        super().__init__()
        self.stages = tuple(stages)
        self.refiners = nn.ModuleDict({
            s: BoundaryRefine(channels[s], wave=wave) for s in self.stages
        })

    def forward(self, name, x):
        if name in self.refiners:
            return self.refiners[name](x)
        return x
