import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from medpy import metric


def calculate_metric_per_case(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0          # both empty -> perfect
    else:
        return 0.0, 0.0          # one empty -> worst (dice 0)


@torch.no_grad()
def test_single_volume(image, label, model, num_classes, patch_size,
                       device="cuda", z_spacing=1):
    """
    image, label : torch tensors (D, H, W) for one 3D volume.
    Returns list of (dice, hd95) per foreground class.
    """
    model.eval()
    image = image.squeeze().cpu().numpy()
    label = label.squeeze().cpu().numpy()

    prediction = np.zeros_like(label)

    # 3D volume: loop over depth slices
    if image.ndim == 3:
        for d in range(image.shape[0]):
            slice_2d = image[d]
            h, w = slice_2d.shape

            # resize slice to model input size
            if (h, w) != tuple(patch_size):
                slice_2d = zoom(slice_2d, (patch_size[0] / h, patch_size[1] / w), order=3)
            inp = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).float().to(device)
            logits = model(inp)
            out = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0)
            out = out.cpu().numpy()

            # resize prediction back to original slice size
            if (h, w) != tuple(patch_size):
                out = zoom(out, (h / patch_size[0], w / patch_size[1]), order=0)
            prediction[d] = out
    else:
        # single 2D image (e.g. ISIC)
        h, w = image.shape
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
