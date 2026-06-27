# Datasets

Three benchmark datasets are used, each with a different imaging type and segmentation task.

---

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
  -> model -> (B, 4, H, W)
  -> CrossEntropyLoss vs label (B, H, W)
```

---

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
  -> model -> (B, 1, 256, 256)
  -> BCEDiceLoss vs mask (B, 256, 256)
```

---

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
  -> model -> (B, 9, 224, 224)
  -> CrossEntropyLoss vs label (B, 224, 224)
```

---

### Model Config Per Dataset

```python
# ACDC
model = AMFSSNet(in_channels=1, num_classes=4)

# ISIC 2018
model = AMFSSNet(in_channels=3, num_classes=1)   # sigmoid, not softmax

# Synapse
model = AMFSSNet(in_channels=1, num_classes=9)
```
