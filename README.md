# AMF-SSNet: Adaptive Multi-Wavelet Frequency-State Space Network

Medical image segmentation model is an standalone imrpoved version compared to the EW-ViT baseline with:
1. An adaptive multi-wavelet bank (replaces fixed Haar wavelet)
2. Mamba state-space branches
3. Cross-domain spatial-frequency fusion
4. Frequency-space prototype learning

**Current stage: Phase 4 -- Wavelet Decomposition (AMF-SSNet)** In this Phase 4 the
model is swapped with the AMF-SSNet.

***

## Project Structure

```
amf-ssnet/
├── train.py                 # train on any dataset
├── test.py                  # evaluate a checkpoint (DSC + HD95)
├── configs/                 # reserved for yaml configs if needed
├── datasets/                # raw data (already downloaded)
│   ├── synapse/{train_npz, test_vol_h5, lists}
│   ├── ACDC/{ACDC_training_slices, ACDC_training_volumes}
│   └── ISIC/{train,val,test}/{images,masks}
├── scripts/
│   ├── inspect_data.py      # verify dataset formats (run first!)
│   ├── train_acdc.sh        # SLURM job scripts
│   ├── train_synapse.sh
│   └── train_isic.sh
└── src/
    ├── config.py            # all hyperparameters, per dataset
    ├── data/                # one loader per dataset + shared transforms
    ├── models/
    │   ├── unet.py          # baseline model (to be replaced by AMF-SSNet)
    │   ├── wavelet.py       # AdaptiveWaveletBank + WaveletDown
    │   └── amfssnet.py      # full AMF-SSNet model
    ├── losses/losses.py     # Dice, Dice+CE, Dice+BCE
    └── utils/               # metrics (DSC/HD95), seeding, logging
```

***

## Environment Setup

Conda env `amfssnet` (PyTorch 2.1.2 + CUDA 12.1). Always request a GPU node first:

```bash
srun --partition=gpu-preempt --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
module load conda/latest cuda/12.1
conda activate amfssnet
```

***

## Phase 3: How to Run

**Step 1. Verify dataset formats:**
```bash
python scripts/inspect_data.py --data_root datasets
```

**Step 2. Smoke test (1 epoch, tiny subset -- confirms pipeline works):**
```bash
python train.py --dataset acdc --smoke_test
```

**Step 3. Full baseline training:**
```bash
sbatch scripts/train_acdc.sh       
sbatch scripts/train_synapse.sh
sbatch scripts/train_isic.sh
```

**Step 4. Evaluate a checkpoint:**
```bash
python test.py --dataset acdc --ckpt experiments/acdc_unet_baseline/best.pth
```

***

## Metrics

- **DSC** (Dice Similarity Coefficient): measures overlap between prediction and ground truth. Higher is better.
- **HD95** (95th percentile Hausdorff Distance): measures boundary error in pixels. Lower is better.

Computed per organ/class and averaged, matching the baseline EW-ViT protocol.

***

## Datasets

Three benchmark datasets are used, each with a different imaging type and segmentation task.

### ACDC - Cardiac MRI

**What it is:** 100 cardiac MRI scans. Each scan is a 3D volume (stack of 2D grayscale slices through the heart).

**File format:**
```
ACDC_training_volumes/
  patient001_frame01.h5
    "image" -> (D, H, W)  float32   # D grayscale slices
    "label" -> (D, H, W)  uint8     # class ID per pixel
```

**Split:**

| Set | Patients | Slices |
|-----|----------|--------|
| Train | 70 patients | ~1,930 slices |
| Val | 10 patients | Full 3D volumes |
| Test | 20 patients | Full 3D volumes |

**Training strategy:** Each 3D volume is split into individual 2D slices at runtime. This gives ~1,930 training samples instead of 70 and reduces GPU memory usage. Val/test use the full 3D volume for accurate 3D Dice evaluation.

**Prediction labels:**

| Class ID | Structure |
|----------|-----------|
| 0 | Background |
| 1 | Right Ventricle (RV) |
| 2 | Myocardium (Myo) |
| 3 | Left Ventricle (LV) |

**Model config:** `in_channels=1`, `num_classes=4`, output: Softmax

**Shape journey (train):**
```
.h5 file -> read one slice -> (H, W)
  -> DataLoader -> (B, 1, H, W)
  -> U-Net -> (B, 4, H, W)
  -> CrossEntropyLoss vs label (B, H, W)
```

***

### ISIC 2018 - Skin Lesion Segmentation

**What it is:** RGB dermoscopy photos of skin lesions. Task is binary: lesion vs. background. Dataset is already split into train/val/test folders.

**File format:**
```
ISIC/train/images/ISIC_0000000.jpg        # RGB skin photo
ISIC/train/masks/ISIC_0000000_segmentation.png   # white=lesion, black=skin
```

**Split:** Pre-divided into train/val/test folders. Just point the loader to the correct folder.

**Prediction labels:**

| Class ID | Description |
|----------|-------------|
| 0 | Background (healthy skin) |
| 1 | Lesion region |

**Model config:** `in_channels=3`, `num_classes=1`, output: Sigmoid (binary, not softmax)

**Transforms:**
```
Train: Resize(256) + HorizontalFlip + VerticalFlip + Rotate + ShiftScaleRotate + Normalize
Val/Test: Resize(256) + Normalize only
```
Uses ImageNet mean/std normalization `(0.485, 0.456, 0.406)` since ISIC is RGB.

**Shape journey:**
```
.jpg file -> cv2.imread -> BGR to RGB -> (H, W, 3)
  -> Albumentations -> Resize + Normalize -> (256, 256, 3)
  -> transpose(2,0,1) -> (3, 256, 256)
  -> DataLoader -> (B, 3, 256, 256)
  -> U-Net -> (B, 1, 256, 256)
  -> BCEDiceLoss vs mask (B, 256, 256)
```

***

### Synapse - Abdominal Multi-Organ CT

**What it is:** 30 abdominal CT scan cases, 3,779 total slices. Segment 8 abdominal organs simultaneously.

**File format:**
```
# Training: pre-sliced 2D slices (one .npz per slice)
train_npz/case0005_slice003.npz
  "image" -> (224, 224)  float32   # one CT slice
  "label" -> (224, 224)  uint8     # organ class ID per pixel

# Test: full 3D volumes
test_vol_h5/case0008.npy.h5
  "image" -> (D, 512, 512)  float32
  "label" -> (D, 512, 512)  uint8
```

**Split (via text file lists):**

| Set | Cases |
|-----|-------|
| Train | 18 cases (listed in lists/train.txt) |
| Test | 12 cases (listed in lists/test_vol.txt) |

Unlike ACDC, Synapse slices are pre-extracted. No runtime slicing needed.

**Prediction labels:**

| Class ID | Organ |
|----------|-------|
| 0 | Background |
| 1 | Aorta |
| 2 | Gallbladder |
| 3 | Spleen |
| 4 | Left Kidney |
| 5 | Right Kidney |
| 6 | Liver |
| 7 | Stomach |
| 8 | Pancreas |

**Model config:** `in_channels=1`, `num_classes=9`, output: Softmax

**Shape journey (train):**
```
.npz file -> np.load -> (224, 224)
  -> transform -> DataLoader -> (B, 1, 224, 224)
  -> U-Net -> (B, 9, 224, 224)
  -> CrossEntropyLoss vs label (B, 224, 224)
```

### U-Net Config Per Dataset

```python
# ACDC
model = UNet(in_channels=1, num_classes=4)

# ISIC 2018
model = UNet(in_channels=3, num_classes=1)   # sigmoid, not softmax

# Synapse
model = UNet(in_channels=1, num_classes=9)
```

***

## Model Architecture

### Baseline: U-Net

Standard U-Net with encoder (downsampling) and decoder (upsampling) connected by skip connections.

```
Input (1ch or 3ch)
  -> inc: DoubleConv -> x1 (64 x H x W)             -- saved for skip
  -> down1: MaxPool + DoubleConv -> x2 (128 x H/2)  -- saved for skip
  -> down2: MaxPool + DoubleConv -> x3 (256 x H/4)  -- saved for skip
  -> down3: MaxPool + DoubleConv -> x4 (512 x H/8)  -- saved for skip
  -> down4: MaxPool + DoubleConv -> x5 (512 x H/16) -- BOTTLENECK
  -> up1: Upsample + cat(x4) + DoubleConv
  -> up2: Upsample + cat(x3) + DoubleConv
  -> up3: Upsample + cat(x2) + DoubleConv
  -> up4: Upsample + cat(x1) + DoubleConv
  -> outc: Conv2d(64 -> num_classes)
Output: (B, num_classes, H, W)  -- one class score per pixel
```

**Skip connections** save a copy of each encoder feature map before it shrinks. On the way back up, that saved copy is concatenated with the upsampled deep features. This gives the decoder both sharp spatial detail (from skip) and abstract understanding (from bottleneck).

**Bottleneck** is the smallest spatial point (e.g. 16x16 for 256x256 input). It has the most abstract understanding of the image but no spatial detail. Skip connections restore that detail on the way up.

**Upsampling** uses bilinear interpolation: blends 4 neighboring pixels by weighted average to produce a smooth larger feature map. No learned weights, no extra parameters.

***

### Improvement: AMF-SSNet (Phase 4)

The key improvement is replacing MaxPool downsampling with an **Adaptive Multi-Wavelet Bank**.

#### Why Replace MaxPool?

MaxPool keeps only the largest value in each 2x2 region and throws away the rest. It discards edge and texture information which is critical for precise organ/lesion boundary segmentation.

#### What a Wavelet Does

A wavelet splits a feature map into 4 frequency sub-bands, all at half the spatial size:

```
Feature map (B, C, H, W)
        -> wavelet decomposition
LL (B, C, H/2, W/2)   -- low frequency: smooth/global shape (like a blurred version)
LH (B, C, H/2, W/2)   -- horizontal edges
HL (B, C, H/2, W/2)   -- vertical edges
HH (B, C, H/2, W/2)   -- diagonal edges and fine texture
```

Same output size as MaxPool but **nothing is thrown away**. All edge and texture information is preserved in the high-frequency bands.

#### The 4-Wavelet Bank

The baseline EW-ViT used only one fixed wavelet (Haar). AMF-SSNet runs 4 wavelets simultaneously:

| Wavelet | Good At |
|---------|---------|
| Haar | Sharp edges (simplest) |
| db4 | Smooth curved boundaries |
| sym4 | Symmetric, less distortion |
| coif4 | Balance of local and global |

A small gating network predicts how much to trust each wavelet for the current input:

```
Input feature map
  -> GlobalAvgPool -> Conv -> Softmax
weights = [0.4, 0.3, 0.2, 0.1]   -- learned per sample, changes each forward pass
```

The outputs of all 4 wavelets are blended using these weights. This means the model can automatically prefer sharper wavelets for ACDC heart walls and smoother wavelets for ISIC lesion boundaries.

#### WaveletDown Block

```
x (B, C, H, W)
  -> 4 wavelets in parallel -> LL and [LH, HL, HH] for each
  -> weighted blend across 4 wavelets
  -> cat LL + high bands -> (B, 4C, H/2, W/2)
  -> Conv(4C -> C) to fuse
  -> DoubleConv(C -> out_ch)
Output: (B, out_ch, H/2, W/2)   -- same size as MaxPool, but richer content
```

#### MaxPool vs WaveletDown

| Property | MaxPool | WaveletDown |
|----------|---------|-------------|
| How it shrinks | Keep max value | Wavelet frequency split |
| Information lost | ~75% thrown away | Nothing thrown away |
| Edge detail | Lost | Preserved in LH/HL/HH bands |
| Adaptive to input | Fixed | Gating weights learned per sample |
| Wavelet choice | None | 4 wavelets blended |

#### Ablation Flag

```python
# WaveletDown used (AMF-SSNet)
model = AMFSSNet(in_channels=1, num_classes=9, use_wavelet=True)

# MaxPool used (plain U-Net control)
model = AMFSSNet(in_channels=1, num_classes=9, use_wavelet=False)
```

Setting `use_wavelet=False` falls back to plain MaxPool downsampling using the exact same training code. This gives a clean internal ablation to measure the wavelet contribution.