# MUJ-DS-2430010320 — Data Science Training Portfolio

| Field | Details |
| --- | --- |
| **Name** | Shaurya Gupta |
| **Registration Number** | 2430010320 |
| **Branch** | Data Science |
| **Batch** | D |
| **Project Title** | Wildfire Detection Using CNN |
| **GitHub Username** | [@shauryagupta575](https://github.com/shauryagupta575) |
| **Training Program** | [Program name], Manipal University Jaipur, [Start month] – [End month] 2026 |
| **Instructor** | [@instructor-username](https://github.com/instructor-username) |
| **Capstone** | In this repository: [`capstone/`](capstone/) |

## Project summary

Binary image classification (fire vs no-fire) on the Kaggle FIRE dataset (999 images). A fine-tuned **ViT-B/16** is compared with ResNet50, MobileNetV2 and a series of custom CNNs that demonstrate underfitting, overfitting and regularisation, plus an SE-attention + focal-loss innovation tested with an ablation study.

| Model | Test accuracy | Macro-F1 | Missed fires (of 113) |
| --- | --- | --- | --- |
| **ViT-B/16 (fine-tuned)** | **99.33%** | **0.991** | **1** |
| ResNet50 (frozen) | 99.33% | 0.991 | 1 |
| MobileNetV2 (frozen) | 97.33% | 0.965 | 4 |
| CNN + SE attention | 92.67% | 0.897 | 3 |
| Regularised CNN (Model C) | 87.33% | 0.846 | 16 |

## Repository structure

| Folder | Contents |
| --- | --- |
| [`assignments/`](assignments/) | CO4 comparative analysis, CO5 innovation report, literature review spreadsheet |
| [`notebooks/`](notebooks/) | Full Jupyter notebook with recorded outputs |
| [`code/`](code/) | Python script exported from the notebook, plus bug fixes |
| [`resources/`](resources/) | Dataset information and reference papers |
| [`presentations/`](presentations/) | Project presentation (PowerPoint) |
| [`capstone/`](capstone/) | Capstone deliverables: results, screenshots, installation guide, contribution workflow |

## Quick start

Full setup instructions: [`capstone/INSTALLATION.md`](capstone/INSTALLATION.md). Short version: open [`notebooks/wildfire_detection_vit-2.ipynb`](notebooks/wildfire_detection_vit-2.ipynb) in Google Colab with a T4 GPU and run all cells; the dataset downloads automatically. To run locally:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/wildfire_detection_vit-2.ipynb
```

## Weekly progress log

| Week | Date | Update |
| --- | --- | --- |
| 1 | [date] | Repository set up; dataset exploration and class-imbalance analysis |
| 2 | [date] | Underfitting / overfitting experiments (Models A, B) |
| 3 | [date] | Regularisation: dropout, BatchNorm, L1 vs L2, early stopping |
| 4 | [date] | 5-fold cross-validation |
| 5 | [date] | ViT-B/16 transfer learning and evaluation |
| 6 | [date] | CO4 comparison with ResNet50, MobileNetV2 and published work |
| 7 | [date] | CO5 SE + focal-loss ablation study |
| 8 | [date] | Presentation and final documentation |
