"""
In Module 2 the spatial Mamba branch and the frequency (DWT->Mamba->IDWT) branch
are added back as INDEPENDENT residuals - they never see each other. This module
COUPLES them: the spatial features guide the frequency branch and the frequency
features guide the spatial branch, before either is written back to the trunk.

The coupling is a bidirectional, per-pixel multiplicative gate:

    f_out = f * (1 + tanh(s2f(s)))      # spatial guides frequency
    s_out = s * (1 + tanh(f2s(f)))      # frequency guides spatial

NON-OBVIOUS CONSTRAINT: the SECOND conv of each gate is zero-initialized, so at
init s2f(s)=f2s(f)=0 -> tanh(0)=0 -> the gates are EXACTLY 1 -> (s_out,f_out)==(s,f).
The unit is therefore an exact identity coupling at init and only diverges once
training rewards it (matches the LayerScale no-op-at-init philosophy of Module 2).
tanh keeps each multiplier in (0,2): bounded, stable, can both suppress and amplify.
"""

import torch
import torch.nn as nn


class CrossDomainFusion(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.s2f = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.f2s = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        for gate in (self.s2f, self.f2s):
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)

    def forward(self, s, f):
        f_out = f * (1.0 + torch.tanh(self.s2f(s)))
        s_out = s * (1.0 + torch.tanh(self.f2s(f)))
        return s_out, f_out


if __name__ == "__main__":
    fus = CrossDomainFusion(64)
    s = torch.randn(2, 64, 28, 28)
    f = torch.randn(2, 64, 28, 28)
    so, fo = fus(s, f)
    print("identity at init:",
          torch.allclose(so, s, atol=1e-6), torch.allclose(fo, f, atol=1e-6),
          "| params", sum(p.numel() for p in fus.parameters()))
