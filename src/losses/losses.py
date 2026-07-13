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


class TverskyLoss(nn.Module):
    def __init__(self, n_classes, alpha=0.3, beta=0.7, class_weights=None, smooth=1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.register_buffer("class_weights",
                             torch.ones(n_classes) if class_weights is None
                             else torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        t = F.one_hot(target.long(), self.n_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        tp = (probs * t).sum(dims)
        fp = (probs * (1 - t)).sum(dims)
        fn = ((1 - probs) * t).sum(dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        w = self.class_weights.to(logits.device)
        return ((1 - tversky) * w).sum() / w.sum()


class FocalLoss(nn.Module):
    def __init__(self, n_classes, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("class_weights",
                             torch.ones(n_classes) if class_weights is None
                             else torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, logits, target):
        logpt = F.log_softmax(logits, dim=1)
        logpt_t = logpt.gather(1, target.long().unsqueeze(1)).squeeze(1)
        pt = logpt_t.exp()
        w = self.class_weights.to(logits.device)[target.long()]
        return (-w * (1 - pt) ** self.gamma * logpt_t).mean()


class SmallOrganLoss(nn.Module):
    def __init__(self, n_classes, tversky_weight=0.0, focal_weight=0.0,
                 alpha=0.3, beta=0.7, gamma=2.0, class_weights=None):
        super().__init__()
        self.tw = tversky_weight
        self.fw = focal_weight
        self.tversky = TverskyLoss(n_classes, alpha, beta, class_weights) if tversky_weight > 0 else None
        self.focal = FocalLoss(n_classes, gamma, class_weights) if focal_weight > 0 else None

    def forward(self, logits, target):
        loss = logits.new_zeros(())
        if self.tversky is not None:
            loss = loss + self.tw * self.tversky(logits, target)
        if self.focal is not None:
            loss = loss + self.fw * self.focal(logits, target)
        return loss


def build_small_organ_loss(cfg):
    tw = getattr(cfg, "tversky_weight", 0.0)
    fw = getattr(cfg, "focal_weight", 0.0)
    if tw <= 0 and fw <= 0:
        return None
    n = cfg.num_classes
    weights = torch.ones(n)
    factor = getattr(cfg, "small_organ_weight", 2.0)
    for c in getattr(cfg, "small_organ_classes", ()):
        if 0 <= c < n:
            weights[c] = factor
    return SmallOrganLoss(n, tversky_weight=tw, focal_weight=fw,
                          alpha=getattr(cfg, "tversky_alpha", 0.3),
                          beta=getattr(cfg, "tversky_beta", 0.7),
                          gamma=getattr(cfg, "focal_gamma", 2.0),
                          class_weights=weights)


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


def _boundary_mask(t, n_classes):
    oh = F.one_hot(t.long(), n_classes).permute(0, 3, 1, 2).float()
    cross = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],
                         device=t.device, dtype=oh.dtype).view(1, 1, 3, 3)
    kernel = cross.repeat(n_classes, 1, 1, 1)
    neigh = F.conv2d(oh, kernel, padding=1, groups=n_classes)
    is_bnd = ((neigh < 5) & (oh > 0)).any(dim=1)
    return is_bnd


class BoundaryPrototypeLoss(nn.Module):
    def __init__(self, n_classes, sep_weight=1.0, sep_margin=-0.2,
                 align_weight=1.0, cons_weight=0.5, cons_tau=0.5,
                 dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.n_classes = n_classes
        self.sep_weight = sep_weight
        self.sep_margin = sep_margin
        self.align_weight = align_weight
        self.cons_weight = cons_weight
        self.cons_tau = cons_tau
        self.dice = DiceLoss(n_classes)
        self.ce = nn.CrossEntropyLoss()
        self.dw = dice_weight
        self.cw = ce_weight

    def _balanced_align(self, penalty, mask, t, K):
        total = penalty.new_zeros(())
        count = 0
        for c in range(K):
            m = mask & (t == c)
            s = m.sum()
            if s > 0:
                total = total + (penalty * m).sum() / s
                count += 1
        return total / max(count, 1)

    def forward(self, proto_seg, embed, centers, boundaries, logits, target):
        B, K, h, w = proto_seg.shape
        t = F.interpolate(target.unsqueeze(1).float(), size=(h, w),
                          mode="nearest").squeeze(1).long()

        seg = self.cw * self.ce(proto_seg, t) + self.dw * self.dice(proto_seg, t)

        C = F.normalize(centers, dim=1)
        Bn = F.normalize(boundaries, dim=1)
        sc = torch.einsum("bdhw,kd->bkhw", embed, C)
        sb = torch.einsum("bdhw,kd->bkhw", embed, Bn)
        cos_c = sc.gather(1, t.unsqueeze(1)).squeeze(1)
        cos_b = sb.gather(1, t.unsqueeze(1)).squeeze(1)

        bnd = _boundary_mask(t, K)
        interior = ~bnd
        align = self._balanced_align(1 - cos_c, interior, t, K) \
              + self._balanced_align(1 - cos_b, bnd, t, K)

        allp = torch.cat([C, Bn], dim=0)
        cls = torch.arange(K, device=allp.device).repeat(2)
        cross = cls.unsqueeze(0) != cls.unsqueeze(1)
        gram = allp @ allp.t()
        sep = F.relu(gram[cross] - self.sep_margin).mean()

        proto_soft = torch.maximum(sc, sb) / self.cons_tau
        main_down = F.interpolate(logits, size=(h, w), mode="bilinear",
                                  align_corners=False)
        cons = F.mse_loss(F.softmax(main_down, dim=1), F.softmax(proto_soft, dim=1))

        total = seg + self.align_weight * align + self.sep_weight * sep \
              + self.cons_weight * cons
        return total, {"seg": seg.detach(), "align": align.detach(),
                       "sep": sep.detach(), "cons": cons.detach()}


def build_bproto_loss(cfg):
    return BoundaryPrototypeLoss(
        n_classes=cfg.num_classes,
        sep_weight=getattr(cfg, "proto_sep_weight", 1.0),
        sep_margin=getattr(cfg, "proto_sep_margin", -0.2),
        align_weight=getattr(cfg, "proto_align_weight", 1.0),
        cons_weight=getattr(cfg, "proto_cons_weight", 0.5),
        cons_tau=getattr(cfg, "proto_cons_tau", 0.5),
    )


class SDFLoss(nn.Module):
    def __init__(self, n_classes, sigma=0.1, reg_weight=1.0, focus_weight=1.0,
                 max_dist=16):
        super().__init__()
        self.n_classes = n_classes
        self.sigma = sigma
        self.reg_weight = reg_weight
        self.focus_weight = focus_weight
        self.max_dist = int(max_dist)
        self.l1 = nn.L1Loss()

    @torch.no_grad()
    def _dist_to(self, src, n):
        reached = src > 0.5
        dist = torch.where(reached, torch.zeros_like(src),
                           torch.full_like(src, float(n)))
        cur = reached
        for i in range(1, n + 1):
            nxt = F.max_pool2d(cur.float(), 3, stride=1, padding=1) > 0.5
            newly = nxt & (~cur)
            dist = torch.where(newly, torch.full_like(dist, float(i)), dist)
            cur = nxt
        return dist

    @torch.no_grad()
    def _gt_sdf(self, target):
        oh = F.one_hot(target.long(), self.n_classes).permute(0, 3, 1, 2).float()
        n = self.max_dist
        out_d = self._dist_to(oh, n)
        in_d = self._dist_to(1.0 - oh, n)
        inside = oh > 0.5
        return torch.where(inside, -in_d, out_d) / n

    def forward(self, sdf_pred, logits, target):
        gt = self._gt_sdf(target)
        reg = self.l1(sdf_pred, gt)

        own = gt.gather(1, target.long().unsqueeze(1)).squeeze(1).abs()
        wmap = torch.exp(-own / self.sigma)
        ce_pix = F.cross_entropy(logits, target.long(), reduction="none")
        focus = (wmap * ce_pix).mean()

        total = self.reg_weight * reg + self.focus_weight * focus
        return total, {"sdf_reg": reg.detach(), "sdf_focus": focus.detach()}


def build_sdf_loss(cfg):
    return SDFLoss(
        n_classes=cfg.num_classes,
        sigma=getattr(cfg, "sdf_sigma", 0.1),
        reg_weight=getattr(cfg, "sdf_reg_weight", 1.0),
        focus_weight=getattr(cfg, "sdf_focus_weight", 1.0),
        max_dist=getattr(cfg, "sdf_max_dist", 16),
    )


def deep_supervision_loss(criterion, ds_logits, target, level_weights=None):
    n = len(ds_logits)
    if level_weights is None:
        w = [1.0 / (2 ** i) for i in range(n)]
        s = sum(w)
        level_weights = [x / s for x in w]
    total = ds_logits[0].new_zeros(())
    for lw, logit in zip(level_weights, ds_logits):
        up = F.interpolate(logit, size=target.shape[-2:], mode="bilinear",
                           align_corners=False)
        total = total + lw * criterion(up, target)
    return total
