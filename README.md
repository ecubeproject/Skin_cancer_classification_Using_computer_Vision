# README.md

# Skin Cancer Detection Using Computer Vision

Deep Learning for Multi Class and Binary Classification on the ISIC 2018 Dataset

## Overview

This repository contains the complete implementation of a computer vision project for automated skin lesion classification using dermoscopic images from the ISIC 2018 Challenge. Two deep learning models were developed and evaluated:

1. A custom Convolutional Neural Network (CNN) trained from scratch
2. A transfer learning model based on EfficientNet B3

The project includes the full training pipeline, evaluation tools, GradCAM explainability, dataset preprocessing scripts, and a deployed Gradio web application for interactive inference.

The codebase follows the modular structure shown below.

````markdown
```text
project_root/
│
├── src/
│   ├── models/
│   │   ├── model_scratch.py
│   │   └── model_effnet.py
│   ├── data/
│   │   ├── dataset_isic.py
│   │   ├── transforms.py
│   │   ├── dataloaders.py
│   │   └── sampler_isic.py
│   ├── training/
│   │   ├── trainer_scratch.py
│   │   ├── trainer_effnet.py
│   │   └── eval_utils.py
│   └── utils/
│       ├── gradcam.py
│       └── gradcam_effnet.py
│
├── notebooks/
│   ├── Notebook_Final.ipynb
│   └── Notebook_Final.pdf
│
├── reports/
│   ├── metrics/
│   ├── models/
│   └── visuals/
│
├── app/
│   ├── app.py
│   └── sample_inputs/
│
├── checkpoints/
│   └── README_WEIGHTS.md
│
├── data/
│   ├── README_DATA.md
│   └── class_balance_stats.json
│
├── requirements.txt
├── requirements_app.txt
├── Final_Report.pdf (optional)
└── README.md
```
````

## Dataset

This project uses the publicly available ISIC 2018 Skin Lesion Analysis dataset. It contains high-resolution dermoscopic images labeled into seven diagnostic classes: MEL, NV, BCC, AKIEC, BKL, DF, VASC.

The dataset cannot be redistributed. Instructions for downloading and preparing it are provided in `data/README_DATA.md`.

## Models Implemented

### 1. Scratch CNN

A custom CNN architecture designed and trained for 150 epochs. It serves as the baseline model for comparison.
The repository includes complete evaluation results such as confusion matrices, ROC curves, and GradCAM visualizations.

### 2. EfficientNet B3

A transfer learning model fine-tuned using ImageNet-normalized inputs.
EfficientNet B3 provides significant improvements in accuracy and generalization.
Final trained weights are stored externally, with download instructions in `checkpoints/README_WEIGHTS.md`.

## Evaluation Metrics

Both models were evaluated using the following metrics and artifacts:

* Accuracy
* Precision, Recall, F1-score (macro, micro, weighted)
* Confusion matrix for 7-class classification
* Binary malignant vs benign performance
* ROC AUC
* GradCAM heatmaps
* Training and validation loss curves
* Epoch-wise accuracy curves

All evaluation outputs are stored under `reports/metrics` and `reports/visuals`.

## Gradio Web Application

A fully functional inference interface is included. Users can upload dermoscopic images and obtain:

* Predicted class
* Class probability distribution
* GradCAM heatmaps for interpretability

Deployed demo (placeholder):
[https://your-app-url.example.com](https://your-app-url.example.com)

To run the app locally:

```
pip install -r requirements_app.txt
python app/app.py
```

## Training the Models

### Train the Scratch CNN

```
pip install -r requirements.txt
python src/training/trainer_scratch.py
```

### Train the EfficientNet B3 Model

```
python src/training/trainer_effnet.py
```

Model checkpoints are saved automatically to the `checkpoints/` directory.
External weight download instructions are provided in `README_WEIGHTS.md`.

## Reproducibility

This repository ensures reproducibility through:

* Fixed and documented directory structure
* Modular and readable code
* Deterministic training and validation splits
* Saved training histories
* Comprehensive evaluation scripts

## Final Report

A complete APA-style technical report summarizing the methodology, experiments, and findings is provided as:

`Final_Report.pdf`

## Citation

If using this repository, please cite the ISIC 2018 dataset and the EfficientNet paper.
Citation details are included in `Final_Report.pdf`.

---

