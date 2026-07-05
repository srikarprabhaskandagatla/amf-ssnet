import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyPrototypeHead(nn.Module):
    def __init__(self, in_ch, num_classes, dim=128, tau=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.tau = tau
        # 1x1 projection of the frequency feature -> embedding space
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        # one learnable prototype per class (L2-normalized where used)
        self.prototypes = nn.Parameter(torch.randn(num_classes, dim))

    def embed(self, feat): # feat (B, in_ch, h, w) -> unit-norm per-pixel embedding (B, dim, h, w)
        return F.normalize(self.proj(feat), dim=1)

    def forward(self, feat):
        """
        Returns:
          seg (B, K, h, w): per-pixel cosine-similarity class logits (/ tau) — the
                            deep-supervised prototype segmentation map.
          e   (B, dim, h, w): the unit-norm embedding (for analysis / tests).
        """
        e = self.embed(feat)
        P = F.normalize(self.prototypes, dim=1)               # (K, dim)
        seg = torch.einsum("bdhw,kd->bkhw", e, P) / self.tau  # (B, K, h, w)
        return seg, e


if __name__ == "__main__":
    head = FrequencyPrototypeHead(256, 9, dim=128)   # x3 = base*4 = 256 ch, 56x56
    x = torch.randn(2, 256, 56, 56)
    seg, e = head(x)
    norms = e.norm(dim=1)
    print("seg", tuple(seg.shape), "| embed", tuple(e.shape),
          "| protos", tuple(head.prototypes.shape),
          "| unit-norm:", torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
          "| params", sum(p.numel() for p in head.parameters()))
