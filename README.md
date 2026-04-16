#Automatic reading method for pointer meters based on lightweight semantic segmentation and spatially constrained OCR 
## Overview
This repository contains the complete code and data for the FSS-DeeplabV3Plus model, as well as the algorithm code for meter detection and reading.
## 1. Requirements

The project requires the following dependencies:

```
numpy==2.4.4
Pillow==12.2.0
timm==1.0.26
torch==2.7.0+cu128
torchvision==0.22.0+cu128
tqdm==4.65.2
```
## 2. Data Structure

### Dataset1
This dataset is used for meter detection.

### Dataset2

This dataset is used for meter segmentation, specifically to segment the pointer and the main scale lines of the meter.

##### Fine-tuned BERT:Fine-tuned BERT configuration file. embedding.py is used to convert symptom and herb text into corresponding embeddings.

## 3. Code Structure

- **utils.py**: Contains basic utility functions for data processing, evaluation metrics, and helper functions used throughout the project.
- **dataloader.py**: Defines data loading functions and custom dataset classes.
- **model.py**: Encapsulates the complete FSS-DeeplabV3Plus model architecture.
- **train_yolo.py**: Main training and evaluation script for Dataset1.
- **train.py**: Main training and evaluation script for Dataset2.
-  **mask1.py**:Keypoint detection and reading are then performed on the segmented results.

## 4. weight
链接: https://pan.baidu.com/s/1P5WZIVjy56cS9c8GETiWog 提取码: ez9e
