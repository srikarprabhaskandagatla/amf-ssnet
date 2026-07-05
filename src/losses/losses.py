import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module): # Expects logits (B, C, H, W) and target (B, H, W)
    def __init__(self, n_classes, smooth=1e-5, ignore_background=False):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        self.ignore_background = ignore_background

    def _one_hot(self, target):
        # target (B, H, W) to (B, C, H, W)
        return F.one_hot(target.long(), self.n_classes).permute(0, 3, 1, 2).float()

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_oh = self._one_hot(target)

        start = 1 if self.ignore_background else 0
        dice_per_class = []
        for c in range(start, self.n_classes):
            p = probs[:, c]
            t = target_oh[:, c]
            inter = (p * t).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            dice = (2 * inter + self.smooth) / (union + self.smooth)
            dice_per_class.append(1 - dice)
        return torch.stack(dice_per_class, dim=0).mean()


class DiceCELoss(nn.Module):
    def __init__(self, n_classes, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss(n_classes)
        self.ce = nn.CrossEntropyLoss()
        self.dw = dice_weight
        self.cw = ce_weight

    def forward(self, logits, target):
        return self.cw * self.ce(logits, target.long()) + self.dw * self.dice(logits, target)


class DiceBCELoss(nn.Module): 
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss(n_classes=2)
        self.ce = nn.CrossEntropyLoss()
        self.dw = dice_weight
        self.cw = ce_weight

    def forward(self, logits, target):
        return self.cw * self.ce(logits, target.long()) + self.dw * self.dice(logits, target)


class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes, alpha_cap=0.8, smooth=1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.alpha_cap = alpha_cap
        self.smooth = smooth

    def _one_hot(self, target):
        return F.one_hot(target.long(), self.n_classes).permute(0, 3, 1, 2).float()

    def boundary_alpha(self, t):
        C, s = self.n_classes, self.smooth
        
        # 4-connected boundary via a depthwise cross-shaped neighbour sum: an
        # interior pixel equals self + 4 neighbours == 5, everything else is boundary.
        cross = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],
                             device=t.device, dtype=t.dtype).view(1, 1, 3, 3)
        kernel = cross.repeat(C, 1, 1, 1)           
        neigh = F.conv2d(t, kernel, padding=1, groups=C)
        boundary = (neigh < 5).to(t.dtype) * t        

        dims = (0, 2, 3)
        Cb = boundary.sum(dim=dims)                  
        S = t.sum(dim=dims)                         
        alpha = torch.clamp(1.0 - (Cb + s) / (S + s), max=self.alpha_cap)  
        return alpha, Cb, S

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)         
        t = self._one_hot(target)                    
        s = self.smooth
        alpha, _, _ = self.boundary_alpha(t)        

        dims = (0, 2, 3)
        inter = (probs * t).sum(dim=dims)            
        y_sum = (t * t).sum(dim=dims)               
        z_sum = (probs * probs).sum(dim=dims)       
        loss_c = (z_sum + y_sum - 2 * inter + s) / \
                 (z_sum + y_sum - (1 + alpha) * inter + s)
        return loss_c.mean()


def build_loss(name, n_classes):
    name = name.lower()
    if name == "dice_ce":
        return DiceCELoss(n_classes)
    if name == "dice_bce":
        return DiceBCELoss()
    if name == "dice":
        return DiceLoss(n_classes)
    raise ValueError(f"Unknown loss '{name}'")


class PrototypeSegLoss(nn.Module):
    def __init__(self, n_classes, sep_weight=1.0, sep_margin=-0.2,
                 dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.n_classes = n_classes
        self.sep_weight = sep_weight
        self.sep_margin = sep_margin
        self.dice = DiceLoss(n_classes)          # multi-class soft Dice (class-balanced)
        self.ce = nn.CrossEntropyLoss()
        self.dw = dice_weight
        self.cw = ce_weight

    def forward(self, proto_seg, prototypes, target):
        B, K, h, w = proto_seg.shape
        # label down to the prototype-map resolution (nearest preserves class ids)
        t = F.interpolate(target.unsqueeze(1).float(), size=(h, w),
                          mode="nearest").squeeze(1).long()          # B, h, w

        seg = self.cw * self.ce(proto_seg, t) + self.dw * self.dice(proto_seg, t)

        # explicit prototype separation (off-diagonal cosine)
        P = F.normalize(prototypes, dim=1)                          # K, D
        gram = P @ P.t()                                            # K, K
        off = gram[~torch.eye(K, dtype=torch.bool, device=P.device)]
        sep = F.relu(off - self.sep_margin).mean()

        total = seg + self.sep_weight * sep
        return total, {"seg": seg.detach(), "sep": sep.detach()}


def build_proto_loss(cfg):
    return PrototypeSegLoss(
        n_classes=cfg.num_classes,
        sep_weight=getattr(cfg, "proto_sep_weight", 1.0),
        sep_margin=getattr(cfg, "proto_sep_margin", -0.2),
    )


def build_boundary_loss(cfg):
    return BoundaryDoULoss(
        n_classes=cfg.num_classes,
        alpha_cap=getattr(cfg, "boundary_alpha_cap", 0.8),
    )
