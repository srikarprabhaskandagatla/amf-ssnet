"""
The baseline EW-ViT uses ONLY the fixed Haar wavelet. Here each feature map is decomposed
with FOUR wavelets (Haar, Daubechies-4, Symlet-4, Coiflet-4) and let
a small gating network predict, per input, how much to trust each wavelet. The
network thus learns which wavelet basis best suits each dataset / region.

IMPORTANT: mode='periodization' is REQUIRED so all four wavelets output the same
spatial size (N/2) and can be combined. Other modes give different sizes per
wavelet (longer filters -> larger output) and cannot be summed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


WAVELET_SET = ["haar", "db4", "sym4", "coif4"]


class AdaptiveWaveletBank(nn.Module): # Decompose into several wavelets and adaptively combine the sub-bands
    """
    Input  : x         (B, C, H, W)
    Returns: LL        (B, C, H/2, W/2)        low-frequency (approximation)
             high      (B, C, 3, H/2, W/2)     high-freq bands (LH, HL, HH)
             weights   (B, K)                  per-sample wavelet weights (softmax)
    """

    def __init__(self, in_channels, wavelets=WAVELET_SET, reduction=4, temperature=1.0):
        super().__init__()
        self.wavelets = wavelets
        self.K = len(wavelets)
        self.temperature = temperature

        # one fixed (non-learnable) DWT per wavelet
        self.dwts = nn.ModuleList([
            DWTForward(J=1, wave=w, mode="periodization") for w in wavelets
        ])

        # gating network: predicts a weight per wavelet from global context
        hidden = max(in_channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                 # (B, C, 1, 1)
            nn.Conv2d(in_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.K, 1),            # (B, K, 1, 1)
        )

    def forward(self, x):
        B = x.size(0)

        # per-sample wavelet weights
        logits = self.gate(x).view(B, self.K)        # (B, K)
        weights = F.softmax(logits / self.temperature, dim=1)

        # decompose with each wavelet
        LLs, highs = [], []
        for dwt in self.dwts:
            yl, yh = dwt(x)            # yl: (B,C,H/2,W/2)  yh[0]: (B,C,3,H/2,W/2)
            LLs.append(yl)
            highs.append(yh[0])

        LL_stack = torch.stack(LLs, dim=1)      # (B, K, C, H/2, W/2)
        high_stack = torch.stack(highs, dim=1)  # (B, K, C, 3, H/2, W/2)

        # weighted combination across wavelets
        w_ll = weights.view(B, self.K, 1, 1, 1)
        w_hi = weights.view(B, self.K, 1, 1, 1, 1)
        LL = (LL_stack * w_ll).sum(dim=1)        # (B, C, H/2, W/2)
        high = (high_stack * w_hi).sum(dim=1)    # (B, C, 3, H/2, W/2)

        return LL, high, weights


class WaveletDown(nn.Module): # Downsampling block built on the wavelet bank
    """
    Replaces MaxPool2d: instead of throwing away information when halving the
    resolution, keep BOTH the low-frequency approximation (LL) and the three
    high-frequency detail bands, fuse them, then apply a conv block.

    x (B, in_ch, H, W) -> (B, out_ch, H/2, W/2),  also returns wavelet weights.
    """

    def __init__(self, in_ch, out_ch, wavelets=WAVELET_SET):
        super().__init__()
        self.bank = AdaptiveWaveletBank(in_ch, wavelets)
        
        # fuse LL (in_ch) + 3 high bands (3*in_ch) = 4*in_ch  ->  in_ch
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch * 4, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        LL, high, weights = self.bank(x)              # LL (B,C,h,w), high (B,C,3,h,w)
        B, C, _, h, w = high.shape
        high_flat = high.reshape(B, C * 3, h, w)      # (B, 3C, h, w)
        fused = self.fuse(torch.cat([LL, high_flat], dim=1))
        return self.conv(fused), weights


if __name__ == "__main__":
    x = torch.randn(2, 16, 64, 64)
    bank = AdaptiveWaveletBank(16)
    LL, high, w = bank(x)
    print("bank:", LL.shape, high.shape, w.shape, "weights sum:", w.sum(1))
    down = WaveletDown(16, 32)
    y, w = down(x)
    print("down:", y.shape)
