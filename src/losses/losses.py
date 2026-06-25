"""
Define Segmentation losses,
- DiceLoss        : multi-class soft Dice
- DiceCELoss      : Dice + CrossEntropy (Synapse, ACDC)
- DiceBCELoss     : Dice + BCE for binary masks (ISIC)

NOTE: the baseline paper uses BDoU (Boundary Difference over Union) for
Synapse/ISIC. This research starts with the standard Dice+CE/BCE baseline; BDoU can be
swapped in later as a refinement.
"""

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


def build_loss(name, n_classes):
    name = name.lower()
    if name == "dice_ce":
        return DiceCELoss(n_classes)
    if name == "dice_bce":
        return DiceBCELoss()
    if name == "dice":
        return DiceLoss(n_classes)
    raise ValueError(f"Unknown loss '{name}'")
