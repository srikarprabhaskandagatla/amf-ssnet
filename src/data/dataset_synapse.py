# Synapse Dataset Loader

"""
FORMAT: 
Loader for the preprocessed Synapse multi-organ CT dataset (TransUNet format).

Train: .npz files, each one 2D slice with keys 'image' and 'label'.
Test : .npy.h5 (or .h5) files, each a full 3D volume with keys 'image','label'.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
except ImportError:
    h5py = None


class SynapseDataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None): # split: 'train' or 'test_vol'
        self.transform = transform
        self.split = split
        self.data_dir = base_dir
        list_file = os.path.join(list_dir, split + ".txt")
        with open(list_file) as f:
            self.sample_list = [ln.strip() for ln in f.readlines() if ln.strip()]

    def __len__(self):
        return len(self.sample_list)

    def _load_test_volume(self, name): # Handle both '<name>.npy.h5' and '<name>.h5' naming
        for ext in (".npy.h5", ".h5"):
            p = os.path.join(self.data_dir, name + ext)
            if os.path.exists(p):
                with h5py.File(p, "r") as f:
                    return f["image"][:], f["label"][:]
        raise FileNotFoundError(f"No test volume found for '{name}' in {self.data_dir}")

    def __getitem__(self, idx):
        name = self.sample_list[idx]

        if self.split == "train":
            path = os.path.join(self.data_dir, name + ".npz")
            data = np.load(path)
            image, label = data["image"], data["label"]
            sample = {"image": image, "label": label}
            if self.transform:
                sample = self.transform(sample)
        else:
            image, label = self._load_test_volume(name)
            # Return full 3D volume; metrics.py slices through it
            sample = { # (D, H, W)
                "image": torch.from_numpy(image.astype(np.float32)),  
                "label": torch.from_numpy(label.astype(np.float32)), 
            }

        sample["case_name"] = name
        return sample
