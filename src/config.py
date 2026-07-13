# Tune Hyperparameters for each dataset
from types import SimpleNamespace

# Shared defaults
_BASE = dict(
    data_root="datasets",
    output_dir="experiments",
    seed=1234,
    num_workers=4, # previously was 8 here
    deterministic=True,
    arch="unet", # model architecture: 'unet' (baseline) or 'amfssnet'
    # AMF-SSNet module switches (turned on one at a time across Phase 4)
    use_wavelet=True,
    use_mamba=False,
    use_fusion=False,
    use_proto=False,
    use_boundary=False,
    mamba_freq=True,
    mamba_stages=(3, 4),
    # VMamba-encoder variant (arch='amfssnet_vm')
    freq_stages=(2, 3, 4),
    use_mamba_dec=False,
    vmamba_ckpt=None,
    drop_path_rate=0.2,
    backbone_lr_mult=1.0,
    warmup_epochs=0,
    # compound small-organ loss (additive, 0 = off)
    tversky_weight=0.0,
    tversky_alpha=0.3,
    tversky_beta=0.7,
    focal_weight=0.0,
    focal_gamma=2.0,
    small_organ_classes=(),
    small_organ_weight=2.0,
    # Module 4 - Frequency Prototype Learning (reworked; only used when use_proto=True).
    # Prototypes act as a cosine-similarity segmentation head on the 56x56 wavelet
    # stage-3 feature (x3), DEEP-SUPERVISED with a class-balanced Dice+CE. See
    # src/models/prototype.py + PrototypeSegLoss in src/losses/losses.py.
    proto_dim=128,          
    proto_tau=0.1,           
    proto_sep_weight=1.0,    
    proto_sep_margin=-0.2,   
    proto_weight=0.1,
    proto_warmup_epochs=10,

    boundary_weight=0.0,
    boundary_alpha_cap=0.8,
    # Boundary-aware prototype (center+boundary sub-prototypes, 2026-style).
    # Only active when use_proto=True; adds an align term on top of PrototypeSegLoss.
    proto_boundary=False,
    proto_align_weight=1.0,
    proto_cons_weight=0.5,
    proto_cons_tau=0.5,
    # Signed-distance-function boundary head (FocusSDF-style, 2025).
    use_sdf=False,
    sdf_weight=0.1,
    sdf_sigma=0.1,
    sdf_reg_weight=1.0,
    sdf_focus_weight=1.0,
    sdf_max_dist=16,
    use_ds=False,
    ds_weight=0.4,
    save_every=50,
    val_every=10,
)

"""
NOTE: img_size, num_classes, in_channels, optimizer, lr, epochs follow the
      baseline paper (Synapse/ISIC: SGD or Adam; ACDC: Adam + CE+Dice).
"""

DATASET_CONFIGS = {
    "synapse": dict(
        dataset="synapse",
        list_dir="datasets/synapse/lists",
        train_dir="datasets/synapse/train_npz",
        test_dir="datasets/synapse/test_vol_h5",
        num_classes=9,            # 8 organs + background
        in_channels=1,            # grayscale CT
        img_size=224,
        batch_size=24,
        optimizer="sgd",
        base_lr=0.05,
        momentum=0.9,
        weight_decay=1e-4,
        max_epochs=400,
        loss="dice_ce",           # baseline uses BDoU; dice_ce is the safe baseline
        z_spacing=1,
        class_names=["background", "aorta", "gallbladder", "kidney_L",
                     "kidney_R", "liver", "pancreas", "spleen", "stomach"],
    ),

    "acdc": dict(
        dataset="acdc",
        slices_dir="datasets/ACDC/ACDC_training_slices",
        volumes_dir="datasets/ACDC/ACDC_training_volumes",
        num_classes=4,            # RV, Myo, LV + background
        in_channels=1,            # grayscale MRI
        img_size=224,
        batch_size=12,
        optimizer="adam",
        base_lr=1e-4,
        weight_decay=1e-4,
        max_epochs=150,
        loss="dice_ce",
        z_spacing=1,
        train_patients=(1, 70),   # patient001 to patient070
        val_patients=(71, 80),    # patient071 to patient080
        test_patients=(81, 100),  # patient081 to patient100
        class_names=["background", "RV", "Myo", "LV"],
    ),

    "isic": dict(
        dataset="isic",
        train_img_dir="datasets/ISIC/train/images",
        train_mask_dir="datasets/ISIC/train/masks",
        val_img_dir="datasets/ISIC/val/images",
        val_mask_dir="datasets/ISIC/val/masks",
        test_img_dir="datasets/ISIC/test/images",
        test_mask_dir="datasets/ISIC/test/masks",
        num_classes=2,            # lesion + background (binary)
        in_channels=3,            # RGB dermoscopy
        img_size=256,
        batch_size=16,
        optimizer="adam",
        base_lr=1e-4,
        weight_decay=1e-4,
        max_epochs=100,
        loss="dice_bce",
        class_names=["background", "lesion"],
    ),
}


def get_config(dataset):
    dataset = dataset.lower()
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'. "
                         f"Choose from {list(DATASET_CONFIGS.keys())}")
    cfg = dict(_BASE)
    cfg.update(DATASET_CONFIGS[dataset])
    return SimpleNamespace(**cfg)


if __name__ == "__main__":
    for ds in DATASET_CONFIGS:
        c = get_config(ds)
        print(f"\n=== {ds} ===")
        for k, v in vars(c).items():
            print(f"  {k}: {v}")
