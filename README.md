# GenSeg-HEp2

GAN-based data augmentation pipeline for low-data HEp-2 cell segmentation using Neural Architecture Search and U-Net.

> **Academic context**
>
> This repository contains the source code developed for my Master's thesis in Biomedical Engineering. The project investigates the use of synthetic data generation to improve semantic segmentation performance when only a limited number of annotated medical images are available.

---

# Overview

Accurate segmentation of HEp-2 cell specimens is an important step in computer-aided diagnosis for autoimmune diseases. However, collecting and manually annotating biomedical images is expensive and time-consuming, making the available datasets relatively small.

GenSeg-HEp2 proposes a complete deep learning pipeline that addresses this limitation through synthetic data generation.

The proposed workflow consists of:

1. preprocessing the original HEp-2 dataset;
2. training a pix2pix conditional GAN;
3. optimizing the generator architecture through differentiable Neural Architecture Search (NAS);
4. generating synthetic image-mask pairs;
5. augmenting the segmentation dataset;
6. training a U-Net segmentation model;
7. comparing the proposed method against a baseline trained without synthetic augmentation.

---

# Objectives

The main goals of the project are:

- investigate GAN-based data augmentation for medical image segmentation;
- optimize the GAN generator using differentiable NAS;
- evaluate the impact of synthetic samples on segmentation performance;
- compare the proposed approach against a standard U-Net baseline.

---

# Repository Structure

```
architecture_pix2pix/     Differentiable NAS primitives
models_pix2pix/           pix2pix implementation
options/                  Training configuration
running_files/            Training and evaluation entry points
scripts/                  Bash scripts for experiments
unet/                     Segmentation network
util/                     Dataset preparation and utilities

data/                     Dataset (not included)
plots/                    Training curves
visuals/                  Generated samples
test_HEp2/                Evaluation outputs
```

---

# Project Workflow

The complete pipeline consists of several stages.

## 1. Dataset preparation

The original HEp-2 images are divided into fixed-size patches.

Cross-validation folds are then generated while preserving patient distribution and intensity classes.

Additional metadata files are automatically created for training and testing.

---

## 2. GAN pre-training

A conditional pix2pix model is trained to learn the mapping between HEp-2 images and segmentation masks.

The resulting generator serves as the starting point for the optimization stage.

---

## 3. Neural Architecture Search

Instead of using a fixed generator architecture, convolutional blocks are optimized through differentiable Neural Architecture Search.

Candidate operations are combined using learnable architecture parameters that are jointly optimized during training.

---

## 4. Synthetic data generation

The optimized generator produces realistic synthetic image-mask pairs.

These synthetic samples are used to increase the diversity of the available training data.

---

## 5. Segmentation training

A U-Net model is trained using the augmented dataset.

For comparison, an additional baseline U-Net is trained using only real images.

---

## 6. Evaluation

The trained models are evaluated on the test set using segmentation metrics such as Dice Score.

Training statistics, generated samples and qualitative visualizations are also produced.

---

# Technologies

- Python 3.9
- PyTorch
- CUDA
- pix2pix
- U-Net
- Differentiable Neural Architecture Search
- Betty-ML
- NumPy
- OpenCV

---

# Requirements

- Python 3.9
- PyTorch 1.13.1
- CUDA 11.6

Environment creation:

```bash
bash env.sh
conda activate GenSeg
```

---

# Dataset Preparation

The repository expects the HEp-2 dataset to follow the structure below.

```
data/
└── HEp-2_specimen/
    ├── train/
    ├── test/
    ├── train.csv
    └── test.csv
```

The dataset itself is **not included** in this repository.

Before training, execute:

```bash
python util/create_patches.py
python util/create_fold_splits.py
python util/create_test_indices.py
```

These scripts generate:

- image patches
- cross-validation splits
- test indices

required by the training pipeline.

---

# Training

The complete experimental pipeline consists of four stages.

### Train pix2pix

```bash
bash scripts/train_pix2pix_hep2.sh
```

---

### Train the proposed NAS-based approach

```bash
bash scripts/train_end2end_hep2.sh
```

---

### Train the baseline

```bash
bash scripts/baseline_hep2.sh
```

---

### Evaluate the models

```bash
bash scripts/test_hep2.sh
```

---

# Outputs

The repository automatically generates:

- trained pix2pix models
- trained U-Net models
- loss curves
- generated images
- qualitative visualizations
- quantitative evaluation metrics

---

# Known Limitations

Some utility scripts currently contain hardcoded dataset paths.

These paths may need to be adapted before running the pipeline on a different machine.

The repository also assumes the availability of CUDA-compatible hardware for training.

---

# Missing Resources

The following resources are intentionally not distributed:

- HEp-2 dataset
- trained model checkpoints
- generated experimental results

Users should generate these files by executing the training pipeline.

---

# Possible Issues

Common issues include:

- incorrect dataset directory structure;
- missing metadata files generated during preprocessing;
- unavailable CUDA environment;
- incorrect model checkpoint paths during evaluation.

Most execution errors are caused by one of the above configuration problems.

---

# Academic Context

This project was developed as part of my Master's Thesis in Biomedical Engineering.

The objective was to investigate whether GAN-generated synthetic biomedical images, combined with differentiable Neural Architecture Search, could improve semantic segmentation performance under low-data conditions.
