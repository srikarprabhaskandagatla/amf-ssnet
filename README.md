# AMF-SSNet: Adaptive Multi-Wavelet Frequency-State Space Network

Medical image segmentation model built on top of the U-Net and compared with EW-ViT baseline, with few improvements.

**Current stage: Phase 4 — Wavelet Decomposition**

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

- **DSC** — overlap between prediction and ground truth. Higher is better.
- **HD95** — boundary error in pixels (95th-percentile Hausdorff). Lower is better.
- **IoU** — intersection over union. Higher is better (ISIC only).

All computed per class and averaged, matching the EW-ViT evaluation protocol.

---

## Wavelet Decomposition (Module 1)

The key change from a plain U-Net is replacing MaxPool downsampling with an **Adaptive Multi-Wavelet Bank**.

### Why Not MaxPool?

MaxPool keeps only the largest value in each 2×2 region, throwing away ~75% of the information. Edges and textures — critical for organ/lesion boundaries — are lost.

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

Setting `use_wavelet=False` falls back to plain MaxPool using the same training code — clean internal ablation.

---

## Results

### ACDC (Cardiac MRI)

| Model | RV DSC | Myo DSC | LV DSC | Mean DSC | Mean HD95 |
|-------|--------|---------|--------|----------|-----------|
| U-Net baseline | 85.74 | 86.32 | 91.88 | 87.98 | 1.76 |
| + Wavelet (ours) | 86.86 | 86.56 | 92.13 | **88.52** | 1.92 |
| EW-ViT (target) | — | — | — | 92.12 | **1.18** |

### Synapse (Abdominal CT)

| Model | Aorta | GB | KL | KR | Liver | Pancreas | Spleen | Stomach | Mean DSC | Mean HD95 |
|-------|-------|----|----|----|-------|----------|--------|---------|----------|-----------|
| U-Net baseline | 87.84 | 55.86 | 82.51 | 77.06 | 94.86 | 61.55 | 86.85 | 76.18 | 77.84 | 46.16 |
| + Wavelet (ours) | 89.27 | 56.74 | 82.94 | 71.41 | 94.58 | 56.47 | 87.53 | 76.45 | 76.92 | **37.40** |
| EW-ViT (target) | — | — | — | — | — | — | — | — | **82.38** | **14.28** |

### ISIC 2018 (Skin Lesion)

| Model | Dice | IoU |
|-------|------|-----|
| U-Net baseline | 87.24 | 79.45 |
| + Wavelet (ours) | **87.41** | **79.81** |
| EW-ViT (target) | 88.07 | — |

> GB = Gallbladder, KL = Left Kidney, KR = Right Kidney. All numbers are on the held-out test set.
