import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet import WaveletDown, WAVELET_SET
from .mamba_block import DualDomainMamba


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MaxPoolDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x), None


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class AMFSSNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=9, base=64,
                 wavelets=WAVELET_SET,
                 use_wavelet=True, use_mamba=False,
                 use_fusion=False, use_proto=False,
                 mamba_stages=(3, 4), mamba_freq=True):
        super().__init__()
        self.use_wavelet = use_wavelet
        self.use_mamba = use_mamba
        self.use_fusion = use_fusion
        self.use_proto = use_proto
        self.mamba_freq = bool(mamba_freq)
        # which deep stages get a Mamba unit (only meaningful when use_mamba)
        self.mamba_stages = tuple(mamba_stages) if use_mamba else ()

        Down = (lambda i, o: WaveletDown(i, o, wavelets)) if use_wavelet else MaxPoolDown

        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 8)

        # Mamba branches at the two deepest stages (created ONLY when use_mamba,
        # so use_mamba=False keeps the exact Module-1 parameter set).
        if use_mamba:
            self.mamba3 = (DualDomainMamba(base * 8, use_freq=self.mamba_freq,
                                           use_fusion=use_fusion)
                           if 3 in self.mamba_stages else None)
            self.mamba4 = (DualDomainMamba(base * 8, use_freq=self.mamba_freq,
                                           use_fusion=use_fusion)
                           if 4 in self.mamba_stages else None)

        self.up1 = Up(base * 8 + base * 8, base * 4)
        self.up2 = Up(base * 4 + base * 4, base * 2)
        self.up3 = Up(base * 2 + base * 2, base)
        self.up4 = Up(base + base, base)
        self.outc = nn.Conv2d(base, num_classes, 1)

    def forward(self, x, return_aux=False):
        wavelet_weights = []

        x1 = self.inc(x)
        x2, w = self.down1(x1); wavelet_weights.append(w)
        x3, w = self.down2(x2); wavelet_weights.append(w)
        x4, w = self.down3(x3); wavelet_weights.append(w)
        if self.use_mamba and self.mamba3 is not None:
            x4 = self.mamba3(x4)                 # stage-3 dual-domain Mamba (28x28)
        x5, w = self.down4(x4); wavelet_weights.append(w)
        if self.use_mamba and self.mamba4 is not None:
            x5 = self.mamba4(x5)                 # bottleneck dual-domain Mamba (14x14)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        if return_aux:
            return logits, {"wavelet_weights": wavelet_weights}
        return logits


def build_amfssnet(cfg):
    return AMFSSNet(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        use_wavelet=getattr(cfg, "use_wavelet", True),
        use_mamba=getattr(cfg, "use_mamba", False),
        use_fusion=getattr(cfg, "use_fusion", False),
        use_proto=getattr(cfg, "use_proto", False),
        mamba_stages=getattr(cfg, "mamba_stages", (3, 4)),
        mamba_freq=getattr(cfg, "mamba_freq", True),
    )


if __name__ == "__main__":
    for um in [False, True]:
        for mf in ([False, True] if um else [True]):
            m = AMFSSNet(1, 4, use_wavelet=True, use_mamba=um, mamba_freq=mf)
            x = torch.randn(2, 1, 224, 224)
            y = m(x)
            n = sum(p.numel() for p in m.parameters())
            tag = f"use_mamba={um}" + (f" mamba_freq={mf}" if um else "")
            print(f"{tag}: out={tuple(y.shape)} params={n/1e6:.2f}M")
