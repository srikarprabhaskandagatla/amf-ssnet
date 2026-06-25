"""
Augmentations shared by the Synapse and ACDC 2D-slice loaders.
Matches the baseline EW-ViT: random rotation + random flip, then resize to img_size.
"""

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator: # augment + resize to (output_size, output_size)

    def __init__(self, output_size):
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if np.random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif np.random.random() > 0.5:
            image, label = random_rotate(image, label)

        h, w = image.shape
        if (h, w) != self.output_size:
            image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w), order=3)
            label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)   # (1, H, W)
        label = torch.from_numpy(label.astype(np.float32)).long()      # (H, W)
        return {"image": image, "label": label}


class ResizeOnly: # Validation/inference transform for a single 2D slice (no augmentation)

    def __init__(self, output_size):
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        h, w = image.shape
        if (h, w) != self.output_size:
            image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w), order=3)
            label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32)).long()
        return {"image": image, "label": label}
