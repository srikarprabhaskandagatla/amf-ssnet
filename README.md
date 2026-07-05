# AMF-SSNet: Adaptive Multi-Wavelet Frequency-State Space Network

Medical image segmentation model built on top of the U-Net and compared with EW-ViT baseline, with few improvements.

**Current stage: Full model deployed - Modules 1-5 (Wavelet + Mamba + Cross-Domain Fusion + Frequency Prototypes + Boundary Refinement Decoder) plus a BDoU boundary loss, all trained/tested on all 3 datasets. See Results below.**

For full dataset details see [DATASETS.md](DATASETS.md).

---
## Environment

Conda env `amfssnet` (PyTorch 2.1.2 + CUDA 12.1). Always run on a GPU node:

```bash
srun --partition=gpu-preempt --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
module load conda/latest cuda/12.1
conda activate amfssnet
```

---

## How to Run

```bash
# Verify datasets
python scripts/inspect_data.py --data_root datasets

# Smoke test (1 epoch)
python train.py --dataset acdc --smoke_test

# Full training
sbatch scripts/train_amfssnet_acdc.sh

# Evaluate
python test.py --dataset acdc --arch amfssnet --ckpt experiments/acdc_amfssnet_wavelet/best.pth
```

---

## Metrics

- **DSC** - overlap between prediction and ground truth. Higher is better.
- **HD95** - boundary error in pixels (95th-percentile Hausdorff). Lower is better.
- **IoU** - intersection over union. Higher is better (ISIC only).

All computed per class and averaged, matching the EW-ViT evaluation protocol.

---

## Wavelet Decomposition (Module 1)

The key change from a plain U-Net is replacing MaxPool downsampling with an **Adaptive Multi-Wavelet Bank**.

### Why Not MaxPool?

MaxPool keeps only the largest value in each 2×2 region, throwing away ~75% of the information. Edges and textures - critical for organ/lesion boundaries - are lost.

### What a Wavelet Does

A wavelet splits a feature map into 4 frequency sub-bands at half the spatial size:

```
Feature map (B, C, H, W)  ->  wavelet decomposition

LL  (B, C, H/2, W/2)   -- smooth/global shape
LH  (B, C, H/2, W/2)   -- horizontal edges
HL  (B, C, H/2, W/2)   -- vertical edges
HH  (B, C, H/2, W/2)   -- diagonal edges + fine texture
```

Same output size as MaxPool, but nothing is thrown away.

### The 4-Wavelet Bank

AMF-SSNet runs 4 wavelets in parallel and blends them with a learned gate:

| Wavelet | Good At |
|---------|---------|
| Haar    | Sharp edges |
| db4     | Smooth curved boundaries |
| sym4    | Symmetric, less ringing |
| coif4   | Balance of local and global |

A small gating network (GlobalAvgPool → Conv → Softmax) predicts per-sample blend weights. The model can automatically prefer sharper wavelets for hard edges (ACDC heart walls) and smoother ones for soft boundaries (ISIC lesions).

### WaveletDown Block

```
x (B, C, H, W)
  -> 4 wavelets in parallel  ->  LL and [LH, HL, HH] each
  -> weighted blend
  -> cat(LL, high bands)  ->  (B, 4C, H/2, W/2)
  -> Conv(4C -> C)  ->  DoubleConv
Output: (B, out_ch, H/2, W/2)
```

### Ablation Flag

```python
model = AMFSSNet(in_channels=1, num_classes=9, use_wavelet=True)   # WaveletDown
model = AMFSSNet(in_channels=1, num_classes=9, use_wavelet=False)  # MaxPool (control)
```

Setting `use_wavelet=False` falls back to plain MaxPool using the same training code - clean internal ablation.

---

## Results

Ablation legend: **+Wavelet** = Module 1, **+Mamba** = Module 1+2 (dual-domain
Mamba, spatial + frequency branches, stages 3+4), **+Fusion** = Module 1+2+3
(cross-domain coupling), **+Proto** = full model with Module 4 (frequency
prototypes), **+BD** = Module 5 (Boundary Refinement Decoder), **+BDoU** =
boundary-aware BDoU loss added on top of Fusion/Mamba. All numbers are on the
held-out test set.

### ACDC (Cardiac MRI)

| Model | RV DSC | Myo DSC | LV DSC | Mean DSC | Mean HD95 |
|-------|--------|---------|--------|----------|-----------|
| U-Net baseline | 85.74 | 86.32 | 91.88 | 87.98 | 1.76 |
| + Wavelet | 86.86 | 86.56 | 92.13 | 88.52 | 1.92 |
| + Mamba | 85.58 | 86.37 | 91.31 | 87.75 | 2.52 |
| + Fusion (best unified base) | 88.60 | 86.41 | 91.21 | **88.74** | 2.06 |
| Full (+Proto) | - | - | - | 87.50 | 2.35 |
| + Fusion + BD | - | - | - | 88.57 | 2.46 |
| + Fusion + BDoU loss | - | - | - | 87.06 | 2.38 |
| EW-ViT (target, corrected) | - | - | - | 90.29 | - |

### Synapse (Abdominal CT)

| Model | Aorta | GB | KL | KR | Liver | Pancreas | Spleen | Stomach | Mean DSC | Mean HD95 |
|-------|-------|----|----|----|-------|----------|--------|---------|----------|-----------|
| U-Net baseline | 87.84 | 55.86 | 82.51 | 77.06 | 94.86 | 61.55 | 86.85 | 76.18 | 77.84 | 46.16 |
| + Wavelet | 89.27 | 56.74 | 82.94 | 71.41 | 94.58 | 56.47 | 87.53 | 76.45 | 76.92 | 37.40 |
| + Mamba | 88.36 | 51.57 | 84.77 | 75.88 | 95.00 | 55.48 | 86.95 | 78.27 | 77.03 | 28.69 |
| + Fusion (best unified base) | 89.60 | 56.07 | 82.91 | 75.80 | 94.46 | 57.15 | 91.14 | 77.06 | 78.02 | 29.66 |
| Full (+Proto) | - | 56.12 | - | - | - | 55.94 | - | - | 76.52 | 34.34 |
| + Fusion + BD | - | 55.34 | - | - | - | 60.19 | - | - | 77.11 | 30.05 |
| + Fusion + BDoU loss (best) | - | **59.52** | - | - | - | **63.07** | - | - | **78.76** | 29.17 |
| EW-ViT (target, corrected) | - | - | - | - | - | - | - | - | **83.51** | **16.68** |

### ISIC 2018 (Skin Lesion)

| Model | Dice | IoU |
|-------|------|-----|
| U-Net baseline | 87.24 | 79.45 |
| + Wavelet | 87.41 | 79.81 |
| + Mamba (best unified base) | **87.88** | **80.44** |
| + Fusion | 86.46 | 78.63 |
| Full (+Proto) | 87.41 | 79.91 |
| + Mamba + BD | 87.86 | 80.63 |
| + Mamba + BDoU loss | 87.76 | 80.52 |
| EW-ViT (target, corrected) | 91.64 | - |

> GB = Gallbladder, KL = Left Kidney, KR = Right Kidney. EW-ViT targets above
> are corrected against the published paper's actual tables (paper reports no
> ACDC HD95). Fusion helps ACDC/Synapse but hurts ISIC; since EW-ViT reports one
> config for all datasets, our best single unified config is
> **Wavelet + Mamba + Fusion (stages 3,4)**. The Boundary Decoder (Module 5) and
> Frequency Prototypes (Module 4) were both trained and came out net-neutral;
> the **BDoU boundary loss** is the lever that actually improved results - it
> gives the best Synapse row and the best gallbladder/pancreas scores of any
> variant (the two organs responsible for the entire Synapse DSC gap).

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