import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

from .mamba_block import VisionMambaBlock
from .fusion import CrossDomainFusion


class FrequencyMamba(nn.Module):
    def __init__(self, channels, d_state=16, d_conv=4, expand=2, ls_init=1e-5,
                 use_fusion=False):
        super().__init__()
        self.dwt = DWTForward(J=1, wave="haar", mode="periodization")
        self.idwt = DWTInverse(wave="haar", mode="periodization")
        self.freq_in = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.freq = VisionMambaBlock(channels, d_state, d_conv, expand)
        self.freq_out = nn.Conv2d(channels, channels * 4, 1, bias=False)
        self.gamma_f = nn.Parameter(ls_init * torch.ones(channels))
        self.fusion = CrossDomainFusion(channels) if use_fusion else None

    def _freq_branch(self, x):
        B, C, H, W = x.shape
        LL, Yh = self.dwt(x)
        h2, w2 = LL.shape[-2:]
        hf = Yh[0].reshape(B, C * 3, h2, w2)
        z = torch.cat([LL, hf], dim=1)
        z = self.freq_in(z)
        z = self.freq(z)
        z = self.freq_out(z)
        LL2 = z[:, :C].contiguous()
        Yh2 = z[:, C:].contiguous().reshape(B, C, 3, h2, w2)
        freq = self.idwt((LL2, [Yh2]))
        if freq.shape[-2:] != (H, W):
            freq = F.interpolate(freq, size=(H, W), mode="bilinear", align_corners=True)
        return freq

    def forward(self, x):
        C = x.shape[1]
        f = self._freq_branch(x)
        if self.fusion is not None:
            x, f = self.fusion(x, f)
        return x + self.gamma_f.view(1, C, 1, 1) * f


if __name__ == "__main__":
    for uf in (False, True):
        m = FrequencyMamba(64, use_fusion=uf).cuda()
        x = torch.randn(2, 64, 28, 28).cuda()
        y = m(x)
        print(f"FrequencyMamba(use_fusion={uf}):", tuple(y.shape),
              "| params", sum(p.numel() for p in m.parameters()))
