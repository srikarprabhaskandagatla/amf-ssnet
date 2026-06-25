# ISIC2018 Task-1 skin-lesion segmentation dataset Dataset Loader
"""
FORMAT:
images: RGB .jpg   (e.g. ISIC_0000000.jpg)
masks : binary .png (e.g. ISIC_0000000_segmentation.png, white=lesion)
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

import albumentations as A


def _list_images(img_dir):
    return sorted([f for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))
                   and not f.startswith("ATTRIBUTION")])


def _mask_name_for(img_name): # ISIC masks are '<stem>_segmentation.png'
    stem = os.path.splitext(img_name)[0]
    return stem + "_segmentation.png"


class ISICDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=256, split="train"):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.split = split
        self.images = _list_images(img_dir)

        if split == "train":
            self.tf = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=20, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.tf = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, _mask_name_for(img_name))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) # binarize to {0,1}

        out = self.tf(image=image, mask=mask)
        image, mask = out["image"], out["mask"]

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()  # (3, H, W)
        mask = torch.from_numpy(mask).long()                        # (H, W)

        return {"image": image, "label": mask, "case_name": os.path.splitext(img_name)[0]}