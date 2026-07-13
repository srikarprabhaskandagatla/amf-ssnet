import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom, label as cc_label
from medpy import metric


@torch.no_grad()
def predict_softmax(model, inp, tta=False):
    prob = torch.softmax(model(inp), dim=1)
    if not tta:
        return prob
    for dims in [(2,), (3,), (2, 3)]:
        prob = prob + torch.flip(torch.softmax(model(torch.flip(inp, dims=dims)), dim=1), dims=dims)
    return prob / 4.0


def keep_largest_cc(mask):
    lab, n = cc_label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def postprocess_largest_cc(prediction, num_classes):
    out = prediction.copy()
    for c in range(1, num_classes):
        m = prediction == c
        if m.sum() == 0:
            continue
        out[m & (~keep_largest_cc(m))] = 0
    return out


def calculate_metric_per_case(pred, gt, dice_only=False):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = float("nan") if dice_only else metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0          # both empty -> perfect
    else:
        return 0.0, 0.0          # one empty -> worst (dice 0)


@torch.no_grad()
def test_single_volume(image, label, model, num_classes, patch_size,
                       device="cuda", z_spacing=1, tta=False, postproc=False,
                       val_batch=16, dice_only=False):
    """
    image, label : torch tensors (D, H, W) for one 3D volume.
    Returns list of (dice, hd95) per foreground class.
    """
    model.eval()
    image = image.squeeze().cpu().numpy()
    label = label.squeeze().cpu().numpy()

    prediction = np.zeros_like(label)

    if image.ndim == 3:
        D, h, w = image.shape
        need_resize = (h, w) != tuple(patch_size)
        for start in range(0, D, val_batch):
            sl = image[start:start + val_batch]
            if need_resize:
                sl = np.stack([zoom(s, (patch_size[0] / h, patch_size[1] / w),
                                    order=3) for s in sl])
            inp = torch.from_numpy(np.ascontiguousarray(sl)).unsqueeze(1).float().to(device)
            out = torch.argmax(predict_softmax(model, inp, tta), dim=1).cpu().numpy()
            for i in range(out.shape[0]):
                o = out[i]
                if need_resize:
                    o = zoom(o, (h / patch_size[0], w / patch_size[1]), order=0)
                prediction[start + i] = o
    else:
        # single 2D image (e.g. ISIC)
        h, w = image.shape
        slice_2d = image
        if (h, w) != tuple(patch_size):
            slice_2d = zoom(slice_2d, (patch_size[0] / h, patch_size[1] / w), order=3)
        inp = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).float().to(device)
        out = torch.argmax(predict_softmax(model, inp, tta), dim=1).squeeze(0).cpu().numpy()
        if (h, w) != tuple(patch_size):
            out = zoom(out, (h / patch_size[0], w / patch_size[1]), order=0)
        prediction = out

    if postproc:
        prediction = postprocess_largest_cc(prediction, num_classes)

    metrics = []
    for c in range(1, num_classes):
        metrics.append(calculate_metric_per_case(prediction == c, label == c,
                                                 dice_only=dice_only))
    return metrics


@torch.no_grad()
def evaluate_isic(model, loader, device="cuda", tta=False, postproc=False):
    model.eval()
    dices, ious = [], []
    for batch in loader:
        img = batch["image"].to(device)
        lbl = batch["label"].cpu().numpy()
        pred = torch.argmax(predict_softmax(model, img, tta), dim=1).cpu().numpy()
        for i in range(pred.shape[0]):
            p = (pred[i] == 1)
            if postproc and p.sum() > 0:
                p = keep_largest_cc(p)
            g = (lbl[i] == 1)
            if p.sum() > 0 and g.sum() > 0:
                dices.append(metric.binary.dc(p, g))
                ious.append(metric.binary.jc(p, g))
            elif p.sum() == 0 and g.sum() == 0:
                dices.append(1.0); ious.append(1.0)
            else:
                dices.append(0.0); ious.append(0.0)
    return float(np.mean(dices)), float(np.mean(ious))
