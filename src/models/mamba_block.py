import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

from .fusion import CrossDomainFusion

def _get_mamba_cls(): # It requires a CUDA GPU even to import
    from mamba_ssm import Mamba
    return Mamba


class VisionMambaBlock(nn.Module):
    """
    VSS-style Mamba block with a 4-DIRECTIONAL (SS2D) weight-tied scan.
    Output shape == input shape (B, C, H, W). No residual here - the caller wraps
    it in a LayerScale-gated residual.
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        Mamba = _get_mamba_cls()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)  
        self.act = nn.SiLU()
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm2 = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _bidir(self, seq):
        f = self.mamba(seq)
        b = torch.flip(self.mamba(torch.flip(seq, dims=[1])), dims=[1])
        return f + b

    def forward(self, x):
        B, C, H, W = x.shape

        xn = x.permute(0, 2, 3, 1).contiguous()              
        xn = self.norm(xn)

        g = self.act(self.gate_proj(xn))                    

        xm = self.in_proj(xn).permute(0, 3, 1, 2).contiguous()   
        xm = self.act(self.dwconv(xm))                         

        # ---- 4-directional weight-tied SS2D scan ----
        # row-major (raster): neighbours along W are adjacent in the sequence
        row = xm.flatten(2).transpose(1, 2).contiguous()                 # B, H*W, C
        row = self._bidir(row).transpose(1, 2).reshape(B, C, H, W)

        # column-major: neighbours along H are adjacent in the sequence
        col = xm.transpose(2, 3).flatten(2).transpose(1, 2).contiguous() # B, W*H, C
        col = self._bidir(col).transpose(1, 2).reshape(B, C, W, H).transpose(2, 3).contiguous()

        xm = row + col                                        # B,C,H,W  (4 directions)
        # ---------------------------------------------

        xm = xm.permute(0, 2, 3, 1).contiguous()              # B,H,W,C
        xm = self.norm2(xm)

        out = xm * g                                          # gated merge
        out = self.out_proj(out)                              # B,H,W,C
        return out.permute(0, 3, 1, 2).contiguous()           # B,C,H,W


class DualDomainMamba(nn.Module):
    """
    SPATIAL branch  : 4-directional Mamba on the feature map (long-range spatial
                      context, both axes).
    FREQUENCY branch: DWT the feature map, run Mamba over the low+high sub-bands
                      (the "Frequency Mamba", at half resolution -> cheap), then
                      IDWT back to full resolution. Optional via `use_freq`.
    """

    def __init__(self, channels, d_state=16, d_conv=4, expand=2, ls_init=1e-5,
                 use_freq=True, use_fusion=False):
        super().__init__()
        self.use_freq = bool(use_freq)

        # Spatial branch
        self.spatial = VisionMambaBlock(channels, d_state, d_conv, expand)
        self.gamma_s = nn.Parameter(ls_init * torch.ones(channels))

        # frequency branch (optional)
        if self.use_freq:
            self.dwt = DWTForward(J=1, wave="haar", mode="periodization")
            self.idwt = DWTInverse(wave="haar", mode="periodization")
            self.freq_in = nn.Sequential(                    # 4C (LL+3 high) -> C
                nn.Conv2d(channels * 4, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(),
            )
            self.freq = VisionMambaBlock(channels, d_state, d_conv, expand)
            self.freq_out = nn.Conv2d(channels, channels * 4, 1, bias=False)  # C -> 4C for IDWT
            self.gamma_f = nn.Parameter(ls_init * torch.ones(channels))

        # --- cross-domain fusion (Module 3, optional; created LAST so use_fusion=False
        #     stays byte-for-byte identical to Module 2). Needs both branches to exist. ---
        self.fusion = (CrossDomainFusion(channels)
                       if (use_fusion and self.use_freq) else None)

    def _freq_branch(self, x):
        B, C, H, W = x.shape
        LL, Yh = self.dwt(x)                   
        h2, w2 = LL.shape[-2:]
        hf = Yh[0].reshape(B, C * 3, h2, w2)          # B, 3C, h, w
        z = torch.cat([LL, hf], dim=1)                # B, 4C, h, w
        z = self.freq_in(z)                           # B, C, h, w
        z = self.freq(z)                              # B, C, h, w  (Mamba in wavelet domain)
        z = self.freq_out(z)                          # B, 4C, h, w
        LL2 = z[:, :C].contiguous()                   # B, C, h, w
        Yh2 = z[:, C:].contiguous().reshape(B, C, 3, h2, w2)
        freq = self.idwt((LL2, [Yh2]))                # B, C, H, W  (inverse wavelet)
        if freq.shape[-2:] != (H, W):                 # guard odd-size rounding
            freq = F.interpolate(freq, size=(H, W), mode="bilinear", align_corners=True)
        return freq

    def forward(self, x):
        C = x.shape[1]
        gs = self.gamma_s.view(1, C, 1, 1)

        if self.fusion is None:
            x = x + gs * self.spatial(x)
            if self.use_freq:
                x = x + self.gamma_f.view(1, C, 1, 1) * self._freq_branch(x)
            return x

        s = self.spatial(x)
        f = self._freq_branch(x)
        s, f = self.fusion(s, f)
        return x + gs * s + self.gamma_f.view(1, C, 1, 1) * f


if __name__ == "__main__":
    for uf, ufus in ((False, False), (True, False), (True, True)):
        m = DualDomainMamba(64, use_freq=uf, use_fusion=ufus).cuda()
        x = torch.randn(2, 64, 28, 28).cuda()
        y = m(x)
        print(f"DualDomainMamba(use_freq={uf}, use_fusion={ufus}):", tuple(y.shape),
              "| params", sum(p.numel() for p in m.parameters()))
