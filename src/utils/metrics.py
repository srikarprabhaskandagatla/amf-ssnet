"""
Evaluation metrics matching the baseline paper: Dice (DSC) and 95% Hausdorff
Distance (HD95), computed per organ/class and averaged.

The key function is `test_single_volume`, which segments a 3D volume slice by
slice (2D model) and reports per-class DSC/HD95, exactly the TransUNet/EW-ViT
evaluation protocol.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from medpy import metric


def calculate_metric_per_case(pred, gt): # Dice + HD95 for one binary mask pair
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0 # both empty which is perfect
    else:
        return 0.0, 0.0 # one empty which is worst (dice 0)


@torch.no_grad()
def test_single_volume(image, label, model, num_classes, patch_size,
                       device="cuda", z_spacing=1):
    model.eval()
    image = image.squeeze().cpu().numpy()
    label = label.squeeze().cpu().numpy()

    prediction = np.zeros_like(label)

    # 3D volume: loop over depth slices
    if image.ndim == 3:
        for d in range(image.shape[0]):
            slice_2d = image[d]
            h, w = slice_2d.shape 
            
            if (h, w) != tuple(patch_size):
                # resize slice to model input size
                slice_2d = zoom(slice_2d, (patch_size[0] / h, patch_size[1] / w), order=3) 
            inp = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).float().to(device)
            logits = model(inp)
            out = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0)
            out = out.cpu().numpy() 
            
            if (h, w) != tuple(patch_size):
                # resize prediction back to original slice size
                out = zoom(out, (h / patch_size[0], w / patch_size[1]), order=0)
            prediction[d] = out
    else:
        h, w = image.shape # single 2D image
        slice_2d = image
        if (h, w) != tuple(patch_size):
            slice_2d = zoom(slice_2d, (patch_size[0] / h, patch_size[1] / w), order=3)
        inp = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).float().to(device)
        logits = model(inp)
        out = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()
        if (h, w) != tuple(patch_size):
            out = zoom(out, (h / patch_size[0], w / patch_size[1]), order=0)
        prediction = out

    metrics = []
    for c in range(1, num_classes):
        metrics.append(calculate_metric_per_case(prediction == c, label == c))
    return metrics


@torch.no_grad()
def evaluate_isic(model, loader, device="cuda"):
    # Binary Dice/IoU for ISIC (single 2D images)
    model.eval()
    dices, ious = [], []
    for batch in loader:
        img = batch["image"].to(device)
        lbl = batch["label"].cpu().numpy()
        logits = model(img)
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()
        for i in range(pred.shape[0]):
            p = (pred[i] == 1)
            g = (lbl[i] == 1)
            if p.sum() > 0 and g.sum() > 0:
                dices.append(metric.binary.dc(p, g))
                ious.append(metric.binary.jc(p, g))
            elif p.sum() == 0 and g.sum() == 0:
                dices.append(1.0); ious.append(1.0)
            else:
                dices.append(0.0); ious.append(0.0)
    return float(np.mean(dices)), float(np.mean(ious))
