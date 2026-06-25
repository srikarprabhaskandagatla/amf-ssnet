# AMF-SSNet - Adaptive Multi-Wavelet Frequency-State Space Network

Medical image segmentation research extending [EW-ViT](https://doi.org/10.1109/WACV61041.2025.00889) with an adaptive multi-wavelet bank, Mamba state-space branches, cross-domain spatial-frequency fusion, and frequency-space prototype learning.

---

## Current Status

A U-Net has been implemented from scratch as the baseline. It is trained and evaluated on all three datasets. AMF-SSNet will be built on top of this pipeline in the next phase.

Test results are in `experiments/`:

**ACDC** 
| Class | DSC | HD95 |
|-------|-----|------|
| RV    | 85.74 | 1.97 |
| Myo   | 86.32 | 1.09 |
| LV    | 91.88 | 2.22 |
| **Mean** | **87.98** | **1.76** |

**Synapse**
| Class | DSC | HD95 |
|-------|-----|------|
| Aorta | 87.84 | 48.33 |
| Gallbladder | 55.86 | 75.22 |
| Kidney L | 82.51 | 67.30 |
| Kidney R | 77.06 | 51.79 |
| Liver | 94.86 | 18.96 |
| Pancreas | 61.55 | 14.43 |
| Spleen | 86.85 | 61.39 |
| Stomach | 76.18 | 31.89 |
| **Mean** | **77.84** | **46.16** |

**ISIC**
| Dice | IoU |
|------|-----|
| 0.8724 | 0.7945 |

---

## Configuration

All hyperparameters are in `src/config.py`. Each dataset has its own block. Key settings:

- **Synapse** - SGD, lr 0.05, 400 epochs, batch 24, Dice+CE loss
- **ACDC** - Adam, lr 1e-4, 150 epochs, batch 12, Dice+CE loss, patients 1–70 train / 71–80 val / 81–100 test
- **ISIC** - Adam, lr 1e-4, 100 epochs, batch 16, Dice+BCE loss

---

## Datasets

Datasets follow the same preprocessing as [TransUNet](https://github.com/Beckschen/TransUNet), sourced from the [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) repository.

- **Synapse/BTCV** - [Google Drive](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd)
- **ACDC** - [Google Drive](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4)
- **ISIC 2018** - [ISIC Archive](https://challenge.isic-archive.com/data/#2018)
