"""
Model registry: Selects the architecture from cfg.arch.

  arch='unet': plain U-Net baseline (Phase 3)
  arch='amfssnet': AMF-SSNet (Phase 4), with module flags use_wavelet/use_mamba/...
"""

from .unet import UNet
from .amfssnet import AMFSSNet, build_amfssnet


def build_model(cfg):
    arch = getattr(cfg, "arch", "unet").lower()
    if arch == "unet":
        return UNet(in_channels=cfg.in_channels, num_classes=cfg.num_classes)
    if arch == "amfssnet":
        return build_amfssnet(cfg)
    if arch == "amfssnet_vm":
        from .amfssnet_vm import build_amfssnet_vm
        return build_amfssnet_vm(cfg)
    raise ValueError(f"Unknown arch '{arch}'. Choose 'unet', 'amfssnet', 'amfssnet_vm'.")
