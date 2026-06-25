# ACDC Dataset Loader

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
except ImportError:
    h5py = None


def _patient_id(fname):
    m = re.match(r"patient(\d+)", fname)
    return int(m.group(1)) if m else -1


def _read_h5(path):
    with h5py.File(path, "r") as f:
        return f["image"][:], f["label"][:]


class ACDCDataset(Dataset):
    """
    Volumes are split by patient number (configured in config.py).

    File format: patientXXX_frameYY.h5 into keys: 'image' (D, H, W) float32, 'label' (D, H, W) uint8.

    Train: yields individual 2D slices (H, W) with augmentation.
    Val/Test: yields full 3D volumes (D, H, W).
    """

    def __init__(self, cfg, split, transform=None):
        self.transform = transform
        self.split = split
        self.data_dir = cfg.volumes_dir 

        if split == "train":
            lo, hi = cfg.train_patients
        elif split == "val":
            lo, hi = cfg.val_patients
        else:
            lo, hi = cfg.test_patients

        all_files = sorted(os.listdir(self.data_dir))
        self.volume_list = [f for f in all_files if lo <= _patient_id(f) <= hi]

        # For training: pre-index every (volume, slice_idx) pair
        if split == "train":
            self.slice_index = []
            for vname in self.volume_list:
                with h5py.File(os.path.join(self.data_dir, vname), "r") as f:
                    n_slices = f["image"].shape[0]
                for s in range(n_slices):
                    self.slice_index.append((vname, s))

    def __len__(self):
        if self.split == "train":
            return len(self.slice_index)
        return len(self.volume_list)

    def __getitem__(self, idx):
        if self.split == "train":
            vname, s = self.slice_index[idx]
            path = os.path.join(self.data_dir, vname)
            with h5py.File(path, "r") as f:
                image = f["image"][s]   # (H, W) float32
                label = f["label"][s]   # (H, W) uint8
            sample = {"image": image.astype(np.float32),
                      "label": label.astype(np.float32)}
            if self.transform:
                sample = self.transform(sample)
            sample["case_name"] = vname.replace(".h5", "") + f"_s{s}"
        else:
            vname = self.volume_list[idx]
            image, label = _read_h5(os.path.join(self.data_dir, vname))
            sample = { 
                # (D, H, W)
                "image": torch.from_numpy(image.astype(np.float32)),   
                "label": torch.from_numpy(label.astype(np.float32)),   
                "case_name": vname.replace(".h5", ""),
            }
        return sample