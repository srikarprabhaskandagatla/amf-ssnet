"""
Main training script for all three datasets.

Usage:
    python train.py --dataset acdc
    python train.py --dataset synapse --batch_size 12 --max_epochs 300
    python train.py --dataset isic
"""

import os
import argparse
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import get_config
from src.utils.misc import set_seed, get_logger, AverageMeter, save_checkpoint, count_params
from src.losses.losses import build_loss
from src.models import build_model

from src.data.transforms import RandomGenerator
from src.data.dataset_synapse import SynapseDataset
from src.data.dataset_acdc import ACDCDataset
from src.data.dataset_isic import ISICDataset
from src.utils.metrics import test_single_volume, evaluate_isic


def build_dataloaders(cfg):
    if cfg.dataset == "synapse":
        train_ds = SynapseDataset(
            base_dir=cfg.train_dir, list_dir=cfg.list_dir, split="train",
            transform=transforms.Compose([RandomGenerator(cfg.img_size)]))
        val_ds = SynapseDataset(
            base_dir=cfg.test_dir, list_dir=cfg.list_dir, split="test_vol")
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    elif cfg.dataset == "acdc":
        train_ds = ACDCDataset(cfg, split="train",
                               transform=transforms.Compose([RandomGenerator(cfg.img_size)]))
        val_ds = ACDCDataset(cfg, split="val")
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    elif cfg.dataset == "isic":
        train_ds = ISICDataset(cfg.train_img_dir, cfg.train_mask_dir, cfg.img_size, split="train")
        val_ds = ISICDataset(cfg.val_img_dir, cfg.val_mask_dir, cfg.img_size, split="val")
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.num_workers)
    else:
        raise ValueError(cfg.dataset)

    return train_loader, val_loader


def build_optimizer(cfg, model):
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.base_lr,
                               momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=cfg.base_lr,
                            weight_decay=cfg.weight_decay)


@torch.no_grad()
def validate(cfg, model, val_loader, device):
    if cfg.dataset == "isic":
        dice, iou = evaluate_isic(model, val_loader, device)
        return dice

    # Synapse / ACDC: per-volume per-class Dice
    all_dice = []
    patch = [cfg.img_size, cfg.img_size]
    for batch in val_loader:
        image = batch["image"][0]   # (D, H, W)
        label = batch["label"][0]
        res = test_single_volume(image, label, model, cfg.num_classes, patch,
                                 device=device, z_spacing=cfg.z_spacing)
        case_dice = sum(d for d, h in res) / len(res)
        all_dice.append(case_dice)
    return sum(all_dice) / len(all_dice)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["synapse", "acdc", "isic"])
    ap.add_argument("--arch", type=str, default=None, choices=["unet", "amfssnet"])
    ap.add_argument("--use_wavelet", type=int, default=None, help="1/0 toggle")
    ap.add_argument("--use_mamba", type=int, default=None, help="1/0 toggle")
    ap.add_argument("--use_fusion", type=int, default=None, help="1/0 toggle")
    ap.add_argument("--use_proto", type=int, default=None, help="1/0 toggle")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--base_lr", type=float, default=None)
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--tag", type=str, default=None, help="extra suffix for the run folder")
    ap.add_argument("--smoke_test", action="store_true",
                    help="Run 1 epoch on a tiny subset to verify the pipeline.")
    args = ap.parse_args()

    cfg = get_config(args.dataset)

    # apply CLI overrides
    for k in ["arch", "batch_size", "base_lr", "max_epochs", "img_size",
              "output_dir", "num_workers"]:
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
            
    # boolean module toggles
    for k in ["use_wavelet", "use_mamba", "use_fusion", "use_proto"]:
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, bool(v))

    run_name = f"{cfg.dataset}_{cfg.arch}"
    if args.tag:
        run_name += f"_{args.tag}"
    exp_dir = os.path.join(getattr(cfg, "output_dir", "experiments"), run_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = get_logger(exp_dir)
    set_seed(cfg.seed, cfg.deterministic)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Config: {vars(cfg)}")

    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(f"Train batches: {len(train_loader)} | Val items: {len(val_loader)}")

    model = build_model(cfg).to(device)
    total, trainable = count_params(model)
    logger.info(f"Model params: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")

    criterion = build_loss(cfg.loss, cfg.num_classes)
    optimizer = build_optimizer(cfg, model)

    max_epochs = 1 if args.smoke_test else cfg.max_epochs
    best_dice = 0.0
    n_iter = 0

    for epoch in range(max_epochs):
        model.train()
        loss_meter = AverageMeter()
        t0 = time.time()

        for i, batch in enumerate(train_loader):
            image = batch["image"].to(device)
            label = batch["label"].to(device)

            logits = model(image)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), image.size(0))
            n_iter += 1

            # poly LR decay (TransUNet-style)
            lr = cfg.base_lr * (1.0 - n_iter / (max_epochs * len(train_loader))) ** 0.9
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if args.smoke_test and i >= 5:
                break

        logger.info(f"Epoch {epoch+1}/{max_epochs}  loss={loss_meter.avg:.4f}  "
                    f"time={time.time()-t0:.1f}s")

        # validation
        if (epoch + 1) % cfg.val_every == 0 or epoch + 1 == max_epochs or args.smoke_test:
            dice = validate(cfg, model, val_loader, device)
            logger.info(f"  >> Val mean Dice: {dice:.4f}")
            if dice > best_dice:
                best_dice = dice
                save_checkpoint({"epoch": epoch, "model": model.state_dict(),
                                 "dice": dice, "config": vars(cfg)},
                                os.path.join(exp_dir, "best.pth"))
                logger.info(f"  >> New best! saved (Dice={dice:.4f})")

        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint({"epoch": epoch, "model": model.state_dict()},
                            os.path.join(exp_dir, f"epoch_{epoch+1}.pth"))

    logger.info(f"Training done. Best val Dice: {best_dice:.4f}")
    if args.smoke_test:
        logger.info("SMOKE TEST PASSED — pipeline works end to end.")


if __name__ == "__main__":
    main()
