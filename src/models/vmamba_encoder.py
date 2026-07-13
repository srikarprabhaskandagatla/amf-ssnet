import torch.nn as nn

from .vmamba_src import Backbone_VSSM


_COMMON = dict(
    patch_size=4, in_chans=3, num_classes=1000,
    ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
    ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
    ssm_init="v0", forward_type="v05_noz",
    mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
    patch_norm=True, norm_layer="ln2d",
    downsample_version="v3", patchembed_version="v2",
    use_checkpoint=False, posembed=False, imgsize=224,
)

_URL = "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/"

_CONFIGS = {
    "tiny":  dict(depths=[2, 2, 8, 2],  dims=96,
                  ckpt="vssm1_tiny_0230s_ckpt_epoch_264.pth"),
    "small": dict(depths=[2, 2, 20, 2], dims=96,
                  ckpt="vssm1_small_0229s_ckpt_epoch_240.pth"),
    "base":  dict(depths=[2, 2, 20, 2], dims=128,
                  ckpt="vssm1_base_0229s_ckpt_epoch_225.pth"),
}


def vmamba_ckpt_url(size):
    return _URL + _CONFIGS[size]["ckpt"]


class VMambaEncoder(nn.Module):
    def __init__(self, in_channels=3, pretrained=None, drop_path_rate=0.2,
                 size="tiny", out_indices=(0, 1, 2, 3)):
        super().__init__()
        if size not in _CONFIGS:
            raise ValueError(f"Unknown VMamba size '{size}'. "
                             f"Choose from {list(_CONFIGS)}.")
        self.in_channels = in_channels
        self.size = size
        spec = _CONFIGS[size]
        cfg = dict(_COMMON)
        cfg["depths"] = list(spec["depths"])
        cfg["dims"] = spec["dims"]
        cfg["drop_path_rate"] = drop_path_rate
        self.backbone = Backbone_VSSM(out_indices=out_indices,
                                      pretrained=pretrained, **cfg)
        self.dims = [spec["dims"] * (2 ** i) for i in range(4)]

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)
