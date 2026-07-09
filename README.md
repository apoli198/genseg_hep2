# Generative AI Enables Medical Image Segmentation in Ultra Low-Data Regimes

## Requirements

Python 3.9 and Pytorch 1.13.1 and CUDA 11.6 are recommended. A local conda env can be create using the following:

```bash
bash env.sh
conda activate semantic
```

## Datasets

```bash
data/HEp-2_specimen
├── train
│   ├── 00001_p0_Mask.tif
│   ├── 00001_p0.tif
│   ├── ...
├── test
│   ├── 00001_p0_Mask.tif
│   ├── 00001_p0.tif
│   ├── ...
├── ...
├── test.csv
├── train.csv
project code
├── ...
```

## Train/Validation/Test
Create patches of the original images, for both train and test set
Change path for the specific set at line 11 and 12, when creating test set patches, mask are not needed, so all mask specific lines can be commented out
```bash
python util/create_patches.py
```
Create fold splits of the dataset and generate a CSV file for test images, used by the dataloader
```bash
python util/create_fold splits.py
python util/create_test_indices.py
```
## Training and Testing

We pre-train the GAN-based augmentation model on the train and val sets followed by training both augmentation and semantic segmentation models end-to-end on the. Finally, we test the trained models on the test set (if avialable).

To train the models from scratch, use the following command (Related configurations of model path should be changed mutually):

```bash
# Pre-train the augmentation model
bash scripts/train_pix2pix_hep2.sh

# Train the segmentation based on our framework
bash scripts/train_end2end_hep2.sh

# Train the baseline
bash scripts/baseline_hep2.sh

# Inference the trained segmentation model
bash scripts/test_hep2.sh
```

During test, in the test_hep2.sh file the model_dir has to be set to load the model either from the baseline or the end2end segmentation model. This modification must be done also in the test_hep2_fold.py file, at line 162