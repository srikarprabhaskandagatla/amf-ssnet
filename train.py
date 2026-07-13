"""
Main training script for all three datasets.

Usage:
    python train.py --dataset "$DATASET" --arch amfssnet \
        --use_wavelet 1 --use_mamba 1 --use_fusion 1 --use_proto 1 \
        --use_boundary 1 --mamba_freq 1 --mamba_stages 2,3,4 \
        --tag full_model
"""

import os
import argparse
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import get_config
from src.utils.misc import set_seed, get_logger, AverageMeter, save_checkpoint, count_params
from src.losses.losses import (build_loss, build_proto_loss, build_bproto_loss,
                               build_boundary_loss, build_small_organ_loss,
                               build_sdf_loss, deep_supervision_loss)
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
    name = getattr(cfg, "optimizer", "sgd").lower()
    wd = cfg.weight_decay
    mult = getattr(cfg, "backbone_lr_mult", 1.0)

    if getattr(cfg, "arch", "") == "amfssnet_vm" and mult != 1.0:
        enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
        rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
        groups = [{"params": rest, "lr": cfg.base_lr},
                  {"params": enc, "lr": cfg.base_lr * mult}]
    else:
        groups = [{"params": list(model.parameters()), "lr": cfg.base_lr}]

    if name == "sgd":
        opt = torch.optim.SGD(groups, lr=cfg.base_lr, momentum=cfg.momentum, weight_decay=wd)
    elif name == "adamw":
        opt = torch.optim.AdamW(groups, lr=cfg.base_lr, weight_decay=wd)
    else:
        opt = torch.optim.Adam(groups, lr=cfg.base_lr, weight_decay=wd)
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    return opt


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
                                 device=device, z_spacing=cfg.z_spacing, dice_only=True)
        case_dice = sum(d for d, h in res) / len(res)
        all_dice.append(case_dice)
    return sum(all_dice) / len(all_dice)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["synapse", "acdc", "isic"])
    ap.add_argument("--arch", type=str, default=None, choices=["unet", "amfssnet", "amfssnet_vm"])
    ap.add_argument("--use_wavelet", type=int, default=None, help="1/0 toggle")
    ap.add_argument("--use_mamba", type=int, default=None, help="1/0 toggle")
    ap.add_argument("--use_fusion", type=int, default=None, help="1/0 toggle")
    ap.add_argument("--use_proto", type=int, default=None, help="1/0 toggle")
    ap.add_argument("--use_boundary", type=int, default=None, help="1/0 toggle")
    ap.add_argument("--mamba_freq", type=int, default=None,
                    help="1/0: include frequency branch in Mamba unit (default 1)")
    ap.add_argument("--mamba_stages", type=str, default=None,
                    help="comma-separated encoder stages to place Mamba at, e.g. '2,3,4'")
    ap.add_argument("--freq_stages", type=str, default=None,
                    help="amfssnet_vm: comma-separated encoder stages for FrequencyMamba, e.g. '2,3,4'")
    ap.add_argument("--use_mamba_dec", type=int, default=None, help="amfssnet_vm: VSS decoder blocks 1/0")
    ap.add_argument("--vmamba_ckpt", type=str, default=None,
                    help="amfssnet_vm: path to pretrained VMamba checkpoint")
    ap.add_argument("--vmamba_size", type=str, default=None,
                    choices=["tiny", "small", "base"],
                    help="amfssnet_vm: VMamba encoder size (default tiny)")
    ap.add_argument("--drop_path_rate", type=float, default=None,
                    help="amfssnet_vm: stochastic-depth rate for the VMamba encoder")
    ap.add_argument("--optimizer", type=str, default=None, choices=["sgd", "adam", "adamw"])
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--backbone_lr_mult", type=float, default=None,
                    help="amfssnet_vm: lr multiplier for the pretrained encoder (e.g. 0.1)")
    ap.add_argument("--warmup_epochs", type=int, default=None,
                    help="linear LR warmup over N epochs before poly decay")
    ap.add_argument("--tversky_weight", type=float, default=None,
                    help="weight of the class-weighted Tversky term (0 = off)")
    ap.add_argument("--focal_weight", type=float, default=None,
                    help="weight of the focal term (0 = off)")
    ap.add_argument("--small_organ_classes", type=str, default=None,
                    help="comma-separated class ids up-weighted by Tversky/focal, e.g. '2,6'")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--base_lr", type=float, default=None)
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--proto_weight", type=float, default=None,
                    help="Module 4: weight of the prototype loss in the total loss")
    ap.add_argument("--proto_dim", type=int, default=None,
                    help="Module 4: prototype/embedding dimension")
    ap.add_argument("--proto_tau", type=float, default=None,
                    help="Module 4: temperature for the alignment CE")
    ap.add_argument("--proto_sep_margin", type=float, default=None,
                    help="Module 4: margin for the off-diagonal separation term "
                         "(must be negative to have any effect at init)")
    ap.add_argument("--proto_warmup_epochs", type=int, default=None,
                    help="Module 4: linearly ramp proto_weight 0->target over N epochs")
    ap.add_argument("--boundary_weight", type=float, default=None,
                    help="Weight of the BDoU boundary loss added to the region loss "
                         "(0 = off, byte-identical to the region-only baseline).")
    ap.add_argument("--proto_boundary", type=int, default=None,
                    help="amfssnet_vm + use_proto: 1 = center+boundary sub-prototypes "
                         "(2026-style), 0 = original single prototype per class")
    ap.add_argument("--proto_align_weight", type=float, default=None,
                    help="weight of the center/boundary alignment term (proto_boundary only)")
    ap.add_argument("--use_sdf", type=int, default=None,
                    help="amfssnet_vm: 1 = add a signed-distance-function boundary head "
                         "(FocusSDF-style), 0 = off")
    ap.add_argument("--sdf_weight", type=float, default=None,
                    help="weight of the SDF loss in the total loss")
    ap.add_argument("--sdf_sigma", type=float, default=None,
                    help="exponential-decay bandwidth for the SDF boundary-focus term")
    ap.add_argument("--use_ds", type=int, default=None,
                    help="amfssnet_vm: 1 = deep supervision (aux decoder heads, "
                         "dropped at inference), 0 = off")
    ap.add_argument("--ds_weight", type=float, default=None,
                    help="weight of the deep-supervision loss in the total loss")
    ap.add_argument("--tag", type=str, default=None, help="extra suffix for the run folder")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint (.pth) to resume training from.")
    ap.add_argument("--smoke_test", action="store_true",
                    help="Run 1 epoch on a tiny subset to verify the pipeline.")
    args = ap.parse_args()

    cfg = get_config(args.dataset)

    # apply CLI overrides
    for k in ["arch", "batch_size", "base_lr", "max_epochs", "img_size",
              "output_dir", "num_workers", "proto_weight", "proto_dim", "proto_tau",
              "proto_sep_margin", "proto_warmup_epochs", "boundary_weight",
              "vmamba_ckpt", "vmamba_size", "drop_path_rate",
              "tversky_weight", "focal_weight",
              "optimizer", "weight_decay", "backbone_lr_mult", "warmup_epochs",
              "proto_align_weight", "sdf_weight", "sdf_sigma", "ds_weight"]:
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)

    # boolean module toggles
    for k in ["use_wavelet", "use_mamba", "use_fusion", "use_proto",
              "use_boundary", "mamba_freq", "use_mamba_dec",
              "proto_boundary", "use_sdf", "use_ds"]:
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, bool(v))

    if args.mamba_stages is not None:
        cfg.mamba_stages = tuple(int(s) for s in args.mamba_stages.split(",") if s.strip())
    if args.freq_stages is not None:
        cfg.freq_stages = tuple(int(s) for s in args.freq_stages.split(",") if s.strip())
    if args.small_organ_classes is not None:
        cfg.small_organ_classes = tuple(int(s) for s in args.small_organ_classes.split(",") if s.strip())

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

    # Module 4: auxiliary frequency-prototype loss (only when use_proto is on)
    use_proto = getattr(cfg, "use_proto", False) and cfg.arch in ("amfssnet", "amfssnet_vm")
    proto_boundary = use_proto and getattr(cfg, "proto_boundary", False)
    if proto_boundary:
        proto_criterion = build_bproto_loss(cfg)
    elif use_proto:
        proto_criterion = build_proto_loss(cfg)
    else:
        proto_criterion = None
    proto_weight = getattr(cfg, "proto_weight", 0.1)
    proto_warmup_epochs = getattr(cfg, "proto_warmup_epochs", 0)
    if use_proto:
        logger.info(f"Prototype loss ON: boundary={proto_boundary} weight={proto_weight} "
                    f"warmup_epochs={proto_warmup_epochs} "
                    f"tau={getattr(cfg, 'proto_tau', 0.1)} "
                    f"sep_weight={getattr(cfg, 'proto_sep_weight', 1.0)} "
                    f"sep_margin={getattr(cfg, 'proto_sep_margin', 0.0)} "
                    f"align_weight={getattr(cfg, 'proto_align_weight', 1.0)} "
                    f"dim={getattr(cfg, 'proto_dim', 128)}")

    # Boundary loss (BDoU): additive refinement term on the main seg logits.
    # weight 0.0 (default) to criterion is None -> training is unchanged.
    boundary_weight = getattr(cfg, "boundary_weight", 0.0)
    boundary_criterion = build_boundary_loss(cfg) if boundary_weight > 0 else None
    if boundary_criterion is not None:
        logger.info(f"Boundary loss (BDoU) ON: weight={boundary_weight} "
                    f"alpha_cap={getattr(cfg, 'boundary_alpha_cap', 0.8)}")

    # SDF boundary head (FocusSDF-style): only when use_sdf is on.
    use_sdf = getattr(cfg, "use_sdf", False) and cfg.arch == "amfssnet_vm"
    sdf_criterion = build_sdf_loss(cfg) if use_sdf else None
    sdf_weight = getattr(cfg, "sdf_weight", 0.1)
    if use_sdf:
        logger.info(f"SDF loss ON: weight={sdf_weight} sigma={getattr(cfg, 'sdf_sigma', 0.1)} "
                    f"reg_weight={getattr(cfg, 'sdf_reg_weight', 1.0)} "
                    f"focus_weight={getattr(cfg, 'sdf_focus_weight', 1.0)}")

    use_ds = getattr(cfg, "use_ds", False) and cfg.arch == "amfssnet_vm"
    ds_weight = getattr(cfg, "ds_weight", 0.4)
    if use_ds:
        logger.info(f"Deep supervision ON: weight={ds_weight}")

    need_aux = use_proto or use_sdf or use_ds

    small_organ_criterion = build_small_organ_loss(cfg)
    if small_organ_criterion is not None:
        small_organ_criterion = small_organ_criterion.to(device)
        logger.info(f"Small-organ loss ON: tversky_w={getattr(cfg, 'tversky_weight', 0.0)} "
                    f"focal_w={getattr(cfg, 'focal_weight', 0.0)} "
                    f"classes={getattr(cfg, 'small_organ_classes', ())}")

    max_epochs = 1 if args.smoke_test else cfg.max_epochs
    best_dice = 0.0
    start_epoch = 0
    n_iter = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("dice", 0.0)
        # restore n_iter so poly-LR schedule is consistent with epoch
        n_iter = start_epoch * len(train_loader)
        logger.info(f"Resumed from {args.resume} — epoch {start_epoch}, best_dice so far {best_dice:.4f}")

    for epoch in range(start_epoch, max_epochs):
        model.train()
        loss_meter = AverageMeter()
        seg_meter = AverageMeter()
        sep_meter = AverageMeter()
        align_meter = AverageMeter()
        cons_meter = AverageMeter()
        bdou_meter = AverageMeter()
        sdf_meter = AverageMeter()
        ds_meter = AverageMeter()
        t0 = time.time()

        if use_proto:
            warmup_frac = min(1.0, (epoch + 1) / proto_warmup_epochs) if proto_warmup_epochs > 0 else 1.0
            cur_proto_weight = proto_weight * warmup_frac

        for i, batch in enumerate(train_loader):
            image = batch["image"].to(device)
            label = batch["label"].to(device)

            if need_aux:
                logits, aux = model(image, return_aux=True)
            else:
                logits = model(image)

            loss = criterion(logits, label)

            if use_proto:
                if proto_boundary:
                    p_loss, p_parts = proto_criterion(aux["proto_seg"], aux["proto_embed"],
                                                      aux["centers"], aux["boundaries"],
                                                      logits, label)
                    align_meter.update(p_parts["align"].item(), image.size(0))
                    cons_meter.update(p_parts["cons"].item(), image.size(0))
                else:
                    p_loss, p_parts = proto_criterion(aux["proto_seg"],
                                                      aux["prototypes"], label)
                loss = loss + cur_proto_weight * p_loss
                seg_meter.update(p_parts["seg"].item(), image.size(0))
                sep_meter.update(p_parts["sep"].item(), image.size(0))

            if boundary_criterion is not None:
                b_loss = boundary_criterion(logits, label)
                loss = loss + boundary_weight * b_loss
                bdou_meter.update(b_loss.item(), image.size(0))

            if use_sdf:
                s_loss, s_parts = sdf_criterion(aux["sdf_pred"], logits, label)
                loss = loss + sdf_weight * s_loss
                sdf_meter.update(s_loss.item(), image.size(0))

            if use_ds:
                ds_l = deep_supervision_loss(criterion, aux["ds_logits"], label)
                loss = loss + ds_weight * ds_l
                ds_meter.update(ds_l.item(), image.size(0))

            if small_organ_criterion is not None:
                loss = loss + small_organ_criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), image.size(0))
            n_iter += 1

            # poly LR decay (TransUNet-style) with optional linear warmup, per group
            total_iters = max_epochs * len(train_loader)
            warmup_iters = getattr(cfg, "warmup_epochs", 0) * len(train_loader)
            if warmup_iters > 0 and n_iter < warmup_iters:
                scale = n_iter / warmup_iters
            else:
                scale = (1.0 - (n_iter - warmup_iters) / max(1, total_iters - warmup_iters)) ** 0.9
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * scale

            if args.smoke_test and i >= 5:
                break

        msg = (f"Epoch {epoch+1}/{max_epochs}  loss={loss_meter.avg:.4f}  "
               f"time={time.time()-t0:.1f}s")
        if use_proto:
            if proto_boundary:
                msg += (f"  proto[seg={seg_meter.avg:.4f} align={align_meter.avg:.4f} "
                        f"cons={cons_meter.avg:.4f} sep={sep_meter.avg:.4f} w={cur_proto_weight:.4f}]")
            else:
                msg += f"  proto[seg={seg_meter.avg:.4f} sep={sep_meter.avg:.4f} w={cur_proto_weight:.4f}]"
        if boundary_criterion is not None:
            msg += f"  bdou={bdou_meter.avg:.4f}"
        if use_sdf:
            msg += f"  sdf={sdf_meter.avg:.4f}"
        if use_ds:
            msg += f"  ds={ds_meter.avg:.4f}"
        logger.info(msg)

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
