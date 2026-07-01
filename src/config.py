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
    mamba_freq=True,         \
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
