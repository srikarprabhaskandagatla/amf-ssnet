import torch
import torch.nn as nn

from .amfssnet import DoubleConv, Up
from .wavelet import WaveletDown, WAVELET_SET
from .vmamba_encoder import VMambaEncoder
from .freq_mamba import FrequencyMamba
from .prototype import FrequencyPrototypeHead, BoundaryAwarePrototypeHead
from .sdf import SDFHead


class WaveletStem(nn.Module):
    def __init__(self, in_ch, c0, c1, wavelets=WAVELET_SET):
        super().__init__()
        self.stem = DoubleConv(in_ch, c0)
        self.down = WaveletDown(c0, c1, wavelets)

    def forward(self, x):
        s0 = self.stem(x)
        s1, w = self.down(s0)
        return s0, s1, w


class AMFSSNetVM(nn.Module):
    def __init__(self, in_channels=1, num_classes=9,
                 wavelets=WAVELET_SET,
                 use_wavelet=True, use_mamba=True, use_fusion=True,
                 use_proto=False, use_mamba_dec=False,
                 freq_stages=(2, 3, 4), stem_channels=(48, 96),
                 vmamba_ckpt=None, vmamba_size="tiny", drop_path_rate=0.2,
                 proto_dim=128, proto_tau=0.1,
                 proto_boundary=False, use_sdf=False, use_ds=False):
        super().__init__()
        self.use_wavelet = use_wavelet
        self.use_mamba = use_mamba
        self.use_fusion = use_fusion
        self.use_proto = use_proto
        self.proto_boundary = proto_boundary
        self.use_sdf = use_sdf
        self.use_ds = use_ds
        self.use_mamba_dec = use_mamba_dec
        self.freq_stages = tuple(freq_stages) if use_mamba else ()

        self.encoder = VMambaEncoder(in_channels, pretrained=vmamba_ckpt,
                                     size=vmamba_size, drop_path_rate=drop_path_rate)
        d0, d1, d2, d3 = self.encoder.dims

        c0, c1 = stem_channels
        self.stem = WaveletStem(in_channels, c0, c1, wavelets)

        stage_ch = {1: d0, 2: d1, 3: d2, 4: d3}
        for s in (1, 2, 3, 4):
            built = (FrequencyMamba(stage_ch[s], use_fusion=use_fusion)
                     if (use_mamba and s in self.freq_stages) else None)
            setattr(self, f"freq{s}", built)

        self.up3 = Up(d3 + d2, d2)
        self.up2 = Up(d2 + d1, d1)
        self.up1 = Up(d1 + d0, d0)
        self.up_s1 = Up(d0 + c1, c1)
        self.up_s0 = Up(c1 + c0, c0)
        self.outc = nn.Conv2d(c0, num_classes, 1)

        if use_mamba_dec:
            from .mamba_block import VisionMambaBlock
            self.dec_mamba3 = VisionMambaBlock(d2)
            self.dec_mamba2 = VisionMambaBlock(d1)
            self.dec_gamma3 = nn.Parameter(1e-5 * torch.ones(d2))
            self.dec_gamma2 = nn.Parameter(1e-5 * torch.ones(d1))

        if use_proto and proto_boundary:
            self.proto_head = BoundaryAwarePrototypeHead(d0, num_classes,
                                                          dim=proto_dim, tau=proto_tau)
        elif use_proto:
            self.proto_head = FrequencyPrototypeHead(d0, num_classes,
                                                      dim=proto_dim, tau=proto_tau)
        else:
            self.proto_head = None

        self.sdf_head = SDFHead(c0, num_classes) if use_sdf else None

        if use_ds:
            self.ds_head0 = nn.Conv2d(c1, num_classes, 1)
            self.ds_head1 = nn.Conv2d(d0, num_classes, 1)
            self.ds_head2 = nn.Conv2d(d1, num_classes, 1)

    def _apply_freq(self, s, feat):
        m = getattr(self, f"freq{s}")
        return m(feat) if m is not None else feat

    def forward(self, x, return_aux=False):
        s0, s1, w_stem = self.stem(x)
        f0, f1, f2, f3 = self.encoder(x)

        f1 = self._apply_freq(2, f1)
        f2 = self._apply_freq(3, f2)
        f3 = self._apply_freq(4, f3)
        f0 = self._apply_freq(1, f0)

        d = self.up3(f3, f2)
        if self.use_mamba_dec:
            d = d + self.dec_gamma3.view(1, -1, 1, 1) * self.dec_mamba3(d)
        d = self.up2(d, f1)
        if self.use_mamba_dec:
            d = d + self.dec_gamma2.view(1, -1, 1, 1) * self.dec_mamba2(d)
        ds_c2 = d
        d = self.up1(d, f0)
        ds_c1 = d
        d = self.up_s1(d, s1)
        ds_c0 = d
        d = self.up_s0(d, s0)
        logits = self.outc(d)

        if return_aux:
            aux = {"wavelet_weights": [w_stem]}
            if self.use_ds:
                aux["ds_logits"] = [self.ds_head0(ds_c0),
                                    self.ds_head1(ds_c1),
                                    self.ds_head2(ds_c2)]
            if self.use_proto and self.proto_head is not None:
                proto_seg, proto_embed = self.proto_head(f0)
                aux["proto_seg"] = proto_seg
                aux["proto_embed"] = proto_embed
                if self.proto_boundary:
                    aux["centers"] = self.proto_head.centers
                    aux["boundaries"] = self.proto_head.boundaries
                else:
                    aux["prototypes"] = self.proto_head.prototypes
            if self.use_sdf and self.sdf_head is not None:
                aux["sdf_pred"] = self.sdf_head(d)
            return logits, aux
        return logits


def build_amfssnet_vm(cfg):
    return AMFSSNetVM(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        use_wavelet=getattr(cfg, "use_wavelet", True),
        use_mamba=getattr(cfg, "use_mamba", True),
        use_fusion=getattr(cfg, "use_fusion", True),
        use_proto=getattr(cfg, "use_proto", False),
        use_mamba_dec=getattr(cfg, "use_mamba_dec", False),
        freq_stages=getattr(cfg, "freq_stages", (2, 3, 4)),
        vmamba_ckpt=getattr(cfg, "vmamba_ckpt", None),
        vmamba_size=getattr(cfg, "vmamba_size", "tiny"),
        drop_path_rate=getattr(cfg, "drop_path_rate", 0.2),
        proto_dim=getattr(cfg, "proto_dim", 128),
        proto_tau=getattr(cfg, "proto_tau", 0.1),
        proto_boundary=getattr(cfg, "proto_boundary", False),
        use_sdf=getattr(cfg, "use_sdf", False),
        use_ds=getattr(cfg, "use_ds", False),
    )
