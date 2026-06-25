"""
Evaluate a trained checkpoint and print per-class DSC and HD95.

Usage:
    python test.py --dataset acdc    --ckpt experiments/acdc_unet_baseline/best.pth
    python test.py --dataset synapse --ckpt experiments/synapse_unet_baseline/best.pth
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import get_config
from src.utils.misc import get_logger
from src.models.unet import build_model
from src.data.dataset_synapse import SynapseDataset
from src.data.dataset_acdc import ACDCDataset
from src.data.dataset_isic import ISICDataset
from src.utils.metrics import test_single_volume, evaluate_isic


def build_test_loader(cfg):
    if cfg.dataset == "synapse":
        ds = SynapseDataset(base_dir=cfg.test_dir, list_dir=cfg.list_dir, split="test_vol")
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)
    if cfg.dataset == "acdc":
        ds = ACDCDataset(cfg, split="test")
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)
    if cfg.dataset == "isic":
        ds = ISICDataset(cfg.test_img_dir, cfg.test_mask_dir, cfg.img_size, split="test")
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["synapse", "acdc", "isic"])
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    cfg = get_config(args.dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_logger("experiments", name=f"test_{cfg.dataset}")

    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    logger.info(f"Loaded checkpoint: {args.ckpt}")

    loader = build_test_loader(cfg)

    if cfg.dataset == "isic":
        dice, iou = evaluate_isic(model, loader, device)
        logger.info(f"ISIC Test  ->  Dice: {dice:.4f}   IoU: {iou:.4f}")
        return

    # Synapse / ACDC: per-class DSC + HD95
    patch = [cfg.img_size, cfg.img_size]
    n_fg = cfg.num_classes - 1
    dsc = np.zeros(n_fg)
    hd = np.zeros(n_fg)
    n = 0

    for batch in loader:
        image = batch["image"][0]
        label = batch["label"][0]
        res = test_single_volume(image, label, model, cfg.num_classes, patch,
                                 device=device, z_spacing=cfg.z_spacing)
        for c, (d, h) in enumerate(res):
            dsc[c] += d
            hd[c] += h
        n += 1

    dsc /= n
    hd /= n
    names = cfg.class_names[1:]   # skip background
    logger.info(f"\n{'Class':<14}{'DSC':>8}{'HD95':>10}")
    for i, name in enumerate(names):
        logger.info(f"{name:<14}{dsc[i]*100:>8.2f}{hd[i]:>10.2f}")
    logger.info(f"{'MEAN':<14}{dsc.mean()*100:>8.2f}{hd.mean():>10.2f}")


if __name__ == "__main__":
    main()