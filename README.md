Skin Cancer Detection Using Computer Vision. Deep Learning for Multi Class and Binary Classification on the ISIC 2018 Dataset

Overview: This repository contains the full implementation of a computer vision project focused on automated skin lesion classification using dermoscopic images from the ISIC 2018 Challenge. Two models were developed and evaluated:
- (1) A custom Convolutional Neural Network (CNN) trained from scratch
- (2) A transfer learning model based on EfficientNet B3

The project includes model training pipelines, evaluation tools, GradCAM explainability, dataset preprocessing, and a deployed Gradio web application for interactive inference.
The modular code is ass per the structure below:

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

3. Dataset: This project uses the publicly available ISIC 2018 Skin Lesion Analysis dataset. It contains high-resolution dermoscopic images labeled into seven diagnostic classes: MEL, NV, BCC, AKIEC, BKL, DF, VASC. The dataset cannot be redistributed. Instructions for downloading it are provided in data/README_DATA.md.

4. Models Implemented
  - 4.1 Scratch CNN: A custom architecture designed and trained for 150 epochs. Serves as the baseline for comparison.Includes complete evaluation metrics, ROC curve, confusion matrices, and GradCAM visualizations.
  - 4.2 EfficientNet B3: A transfer learning model fine-tuned using ImageNet-normalized inputs.Provides significantly improved accuracy and generalization.Final model checkpoint is stored externally with download instructions in checkpoints/README_WEIGHTS.md.

5. Evaluation Metrics: The following metrics and artifacts are included for both models: Accuracy, Precision, Recall, F1 (macro, micro, weighted),Confusion matrix for 7-class classification, Binary malignant vs benign evaluation, ROC AUC, GradCAM heatmaps, Loss and accuracy curves across epochs. All figures are available under reports/metrics and reports/visuals.

6. Gradio Web Application

A fully functional inference interface allows users to upload skin lesion images and receive:

Predicted diagnostic class

Class probabilities

GradCAM heatmaps for interpretability
Deployed demo (placeholder link): https://your-app-url.example.com
To run locally:
pip install -r requirements_app.txt
python app/app.py 
7. Training the Models
pip install -r requirements.txt
python src/training/trainer_scratch.py
7.2 Train EfficientNet B3
python src/training/trainer_effnet.py
Weights will automatically save to checkpoints/.
External download instructions are provided in README_WEIGHTS.md.
8. Reproducibility

This repository ensures reproducibility through:

Fixed directory structure

Modular and documented code

Deterministic data splits

Saved training histories

Comprehensive evaluation scripts
9. Final Report

A full APA-style technical report summarizing methodology, experiments, and findings is included as: Final_Report.pdf
10. Citation: If using this repository, please also cite the ISIC dataset and the EfficientNet paper (citations included in Final_Report.pdf).
