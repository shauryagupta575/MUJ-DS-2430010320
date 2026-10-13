# CO4 – Comparative Analysis: ViT-B/16 Wildfire Detection vs State of the Art

*Report date: 13 October 2026 · Source notebook: `wildfire_detection_vit-2.ipynb`*

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Proposed model and evaluation protocol](#2-proposed-model-and-evaluation-protocol)
3. [Selection of state-of-the-art methods](#3-selection-of-state-of-the-art-methods)
4. [Same-dataset comparison with published work](#4-same-dataset-comparison-with-published-work)
5. [Same-split comparison with re-implemented baselines](#5-same-split-comparison-with-re-implemented-baselines)
6. [Transformer-based state of the art on other datasets](#6-transformer-based-state-of-the-art-on-other-datasets)
7. [Why the proposed model performs better, similarly, or worse](#7-why-the-proposed-model-performs-better-similarly-or-worse)
8. [Threats to validity](#8-threats-to-validity)
9. [Conclusion](#9-conclusion)
10. [References](#10-references)

## 1. Purpose and scope

The proposed ViT-B/16 transfer-learning model reaches **99.33% accuracy and 0.9911 macro-F1** on the held-out test split of the Kaggle fire dataset. That edges out the only published method evaluated on the same dataset (99.2%) and ties a frozen ResNet50 baseline trained on the identical split; neither gap is statistically meaningful on 150 test images.

This report addresses each CO4 requirement as follows.

| CO4 requirement | Where it is met |
| --- | --- |
| Select relevant state-of-the-art methods | Section 3 |
| Use the same dataset wherever possible | Sections 4 (published, same dataset) and 5 (re-implemented, same split) |
| Report appropriate performance metrics | Section 2 (definitions); Sections 4–6 (results) |
| Clearly compare proposed vs existing | Sections 4, 5 and 6 |
| Explain better / similar / worse | Section 7 |
| Provide references | Section 10 |

**Which model is "proposed".** The notebook's CO4 cell labels the custom regularised CNN (Model C) as "Proposed" and the ViT as "SOTA 1". This report treats the fine-tuned ViT-B/16 (Model D) as the proposed system, because it is the notebook's final, best-performing and saved model. Model C and the SE + Focal-loss variant are still reported in Section 5.

## 2. Proposed model and evaluation protocol

All proposed-model numbers come from one stratified 70/15/15 split of the 999-image Kaggle fire dataset, with the test split held out from training and early stopping.

**Dataset.** Kaggle `phylake1337/fire-dataset` (also distributed as the NASA Space Apps Challenge fire dataset): 999 images, 755 fire and 244 no-fire (3.09 : 1 imbalance; majority-class baseline 75.58%). All files decoded successfully.

| Split | Images | Fire | No-fire |
| --- | --- | --- | --- |
| Train | 699 | 529 | 170 |
| Validation | 150 | 113 | 37 |
| Test | 150 | 113 | 37 |

**Model (Model D).** ViT-B/16 pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k (`vit_base_patch16_224_imagenet`, keras-hub 0.32). Each 224×224 image becomes 196 patches of 16×16 plus a [CLS] token, passed through 12 encoder blocks. The [CLS] output feeds a new head: Dense(128, GELU) → Dropout(0.4) → Dense(1, sigmoid). Total parameters: 85.9 M.

**Training.**

1. Phase 1, feature extraction: backbone frozen (98,561 trainable parameters), Adam LR 1e-3, up to 20 epochs, early stopping on validation loss (patience 6).
2. Phase 2, fine-tuning: last 4 of 12 encoder blocks and the final LayerNorm unfrozen (28.45 M trainable), Adam LR 1e-5, early stopping (patience 8) and ReduceLROnPlateau.
3. Both phases use class weights {no-fire: 2.056, fire: 0.661} and training-only augmentation (horizontal flip, ±8% rotation, ±12% zoom, contrast and brightness jitter), followed by the ViT's own [-1, 1] preprocessing.

**Metrics.** Fire is the positive class and the decision threshold is 0.5. Because the classes are imbalanced, accuracy alone is misleading, so the report also gives precision, recall, F1 for the fire class, macro-F1 (the unweighted mean of both classes' F1), ROC-AUC, and the number of missed fires (false negatives), which is the most costly error in wildfire detection.

```math
F_1 = \frac{2PR}{P + R}, \qquad \text{macro-}F_1 = \tfrac{1}{2}\left(F_1^{\text{fire}} + F_1^{\text{no-fire}}\right)
```

## 3. Selection of state-of-the-art methods

Nine methods were selected in three tiers, ordered by how directly they can be compared with the proposed model. Tier A shares the exact train/test split, Tier B shares the dataset but not the split, and Tier C uses other datasets and is included for architectural context only.

| Method | Tier | Architecture family | Why it was selected |
| --- | --- | --- | --- |
| ResNet50, frozen ImageNet features [3] | A — same split | Deep residual CNN | The most widely used transfer-learning baseline in fire detection; re-trained by us on the identical split |
| MobileNetV2, frozen ImageNet features [4] | A — same split | Lightweight CNN | Edge-deployable baseline for camera towers and drones; re-trained on the identical split |
| Custom CNNs: Models B, C, L1/L2, SE + Focal | A — same split | From-scratch CNNs | Show what is achievable without pre-training |
| Lilhore et al. 2026, ViT + 3D-CNN + Transformer [6] | B — same dataset | Hybrid ViT, spatio-temporal | Only peer-reviewed method confirmed on the same 999-image dataset; also ViT-based |
| Fan et al. 2026, frozen DINOv3 + MLP [7] | B — likely same dataset | Self-supervised ViT | Same frozen-ViT + small-head recipe as the proposed Phase 1; reports a dataset named "Fire" |
| Vuppari et al. 2025, fine-tuned ViT-B/16 [8] | C — other dataset | ViT | Identical backbone, learning rate and batch size |
| Şener et al. 2026, DeiT3 + MaxViT + Swin fusion [9] | C — other dataset | Multi-ViT fusion + SVM | Current transformer-fusion state of the art for fire / no-fire |
| Azhar et al. 2026, SmartFire Vision [10] | C — other dataset | Pruned ViT-B/16 + DETR | Most recent ViT-based fire system (Aug 2026), same ViT-B/16 configuration |
| Hamdan et al. 2026, Swin Transformer on GWFP [11] | C — other dataset | Hierarchical ViT | Only study measuring cross-dataset generalisation of a transformer |

The CNN-ViT hybrid CN2VF-Net (Ahmad et al., 2025 [5]) was reviewed but excluded from the tables because it solves a detection task (bounding boxes, mAP) rather than image classification.

## 4. Same-dataset comparison with published work

On the 999-image Kaggle fire dataset, the proposed model performs **similarly to** the strongest published method (99.33% vs 99.2% accuracy) while using a simpler, image-only pipeline and a stricter test protocol.

| Method | Input | Test set | Accuracy | Precision (fire) | Recall (fire) | F1 (fire) |
| --- | --- | --- | --- | --- | --- | --- |
| **Proposed: ViT-B/16, frozen → last 4 blocks fine-tuned** | Single image | 150 original images, never augmented | **99.33%** | **1.000** | **0.991** | **0.996** |
| Lilhore et al. 2026: ViT + 3D-CNN + Transformer encoder [6] | Image + 10-frame clip | 3,826 images drawn from an augmented pool | 99.2% | 0.992 | 0.991 | 0.991 |
| Lilhore et al. 2026: their ResNet50 / VGG16 baselines [6] | Single image | As above | 85–94% (range) | — | — | — |
| Fan et al. 2026: frozen DINOv3 + MLP [7] | Single image | Not stated in abstract | up to 99% (best of 3 datasets) | — | — | — |

Two differences in protocol matter when reading this table.

- **Lilhore et al. augmented before splitting.** They expanded the 999 images to 12,755 per class and then split 70/15/15 (17,857 / 3,827 / 3,826; these sum to 2 × 12,755). Augmented copies of one source image can therefore land in both training and test sets, which tends to inflate test accuracy. The proposed model's test split contains only untouched originals.
- **Fan et al. report only a best-case figure.** The abstract gives "up to 99%" across three datasets without saying which one; their "Fire" dataset is probably, but not confirmably, the same Kaggle set. Their result is listed for completeness, not as a head-to-head score.

The proposed model also beats the ResNet50 / VGG16 results Lilhore et al. report on this dataset by 5–14 points. Section 5 shows that this gap reflects weak baselines in that paper rather than an advantage of ViT: a properly configured frozen ResNet50 matches the proposed model exactly.

## 5. Same-split comparison with re-implemented baselines

On the identical 150-image test split, the proposed ViT-B/16 **ties** a frozen ResNet50, is **not significantly better** than MobileNetV2 or the L1/L2-regularised CNNs, and is **significantly better** than every other from-scratch CNN.

| Model | Accuracy | Precision (fire) | Recall (fire) | F1 (fire) | Macro-F1 | Missed fires (FN) | False alarms (FP) | McNemar p vs proposed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Proposed: ViT-B/16 (Model D)** | **99.33%** | **1.000** | **0.991** | **0.996** | **0.991** | **1** | **0** | — |
| ResNet50, frozen [3] | 99.33% | 1.000 | 0.991 | 0.996 | 0.991 | 1 | 0 | 1.00 |
| MobileNetV2, frozen [4] | 97.33% | 1.000 | 0.965 | 0.982 | 0.965 | 4 | 0 | ≤ 0.38 |
| L1 / L2-regularised CNN (identical results) | 95.33% | 0.982 | 0.956 | 0.969 | 0.939 | 5 | 2 | ≤ 0.07 |
| Model B: CNN, no regularisation | 93.33% | 0.990 | 0.920 | 0.954 | 0.916 | 9 | 1 | ≤ 0.012 |
| CNN + SE attention (ablation) | 92.67% | 0.932 | 0.973 | 0.952 | 0.897 | 3 | 8 | ≤ 0.007 |
| CNN + SE + Focal loss (notebook's CO5 proposal) | 88.67% | 0.869 | 1.000 | 0.930 | 0.816 | 0 | 17 | < 0.001 |
| Model C: regularised CNN (notebook's CO4 "proposed") | 87.33% | 0.970 | 0.858 | 0.911 | 0.846 | 16 | 3 | < 0.001 |

Notes on the table:

- **McNemar p-values** test whether two models make different errors on the same images. Image-level predictions were not saved, so each p is the largest value consistent with both models' error counts (exact binomial test); "≤ 0.07" means the true p is 0.07 or lower. Values below 0.05 indicate a real difference.
- **Model C architecture under 5-fold cross-validation** reaches 94.81% ± 2.44 accuracy (F1 0.966), so its single-split 87.33% is an unlucky run. Even the cross-validated figure is 4.5 points below the proposed model.
- **Model size** differs by 330×: ViT-B/16 has 85.9 M parameters, ResNet50 about 23.9 M, MobileNetV2 2.4 M, and Model C 0.26 M.
- The SE + Focal model is the only one with zero missed fires, but it raises 17 false alarms (46% of no-fire images). Section 7 explains why.

## 6. Transformer-based state of the art on other datasets

These results use different datasets, so they show what transformer methods achieve elsewhere, not whether the proposed model is better. The key lesson is that scores fall sharply on harder or unseen data.

| Method | Dataset (size) | Task | Accuracy | F1 | What it tells us |
| --- | --- | --- | --- | --- | --- |
| Şener et al. 2026: DeiT3 + MaxViT + Swin → mRMR → SVM [9] | FlameVision (8,600) | Fire / no-fire | 100% (fused); Swin alone 95.0% | — | A single frozen transformer is not automatically near-perfect; feature fusion added 5–14 points |
| Vuppari et al. 2025: fine-tuned ViT-B/16 [8] | The Wildfire Dataset (2,700) | Fire / no-fire | 96.10% | 0.96 (fire) | Same backbone and hyper-parameters as the proposed model scores 3 points lower on a larger, more varied dataset |
| Azhar et al. 2026: pruned ViT-B/16 + DETR [10] | FURG Fire video (28,022 frames) | Fire / smoke | 85.40% | 0.854 | With a video-level split and a harder fire-vs-smoke task, a ViT-based system lands near 85% |
| Hamdan et al. 2026: Swin Transformer, in-domain [11] | GWFP | Fire / no-fire | 94.6% | 0.945 | Strong in-domain performance … |
| Hamdan et al. 2026: same Swin, tested on FlameVision [11] | FlameVision (unseen) | Fire / no-fire | 82.1% | 0.612 | … but F1 collapses by 33 points on an unseen dataset, the largest drop of all models tested |

Against this backdrop, the proposed model's 99.33% says more about the 999-image dataset being relatively easy than about ViT-B/16 being stronger than these methods.

## 7. Why the proposed model performs better, similarly, or worse

| Compared with | Outcome | Main reason |
| --- | --- | --- |
| From-scratch CNNs (Models B, C, SE, SE + Focal) | Better (p < 0.05) | Pre-trained features vs learning from 699 images |
| ResNet50 (frozen) | Similar (identical scores) | Both pre-trained feature sets already separate the classes |
| MobileNetV2, L1/L2 CNNs | Similar (not significant) | Too few test images to separate them |
| Lilhore et al. 2026 | Similar (99.33% vs 99.2%) | Dataset ceiling; cleaner protocol on our side |
| Efficiency (all CNNs) | Worse | 35× more parameters than MobileNetV2 |
| Unseen data | Unknown, likely worse | Not tested; transformers dropped most in [11] |

### Better than from-scratch CNNs: pre-training

ViT-B/16 arrives with features learned from about 14 million ImageNet-21k images; the custom CNNs must learn colour, texture and shape cues from 699 images. Model C also suffered BatchNorm instability: its validation accuracy sat at exactly 75.33% (the "predict everything as fire" rate) for its first eight epochs. The SE + Focal model's 17 false alarms trace to its focal-loss setting α = 0.75, which up-weights the fire class even though fire is already the majority.

### Similar to ResNet50 and Lilhore et al.: the dataset ceiling

The proposed model's own training log shows the dataset is near-saturated. In Phase 1 the best validation loss came at **epoch 1**, and validation accuracy stayed at 99.33% (one error) for every epoch of both phases. Fine-tuning 28 M parameters never improved on Phase 1's best validation loss (0.0677 vs 0.0650).

In plain terms, frozen ImageNet features, whether from a transformer or a CNN, already place fire and no-fire images far apart, so a small head separates them. ViT's main architectural advantage, self-attention that links a flame in one corner to smoke in another, has no room to show on this dataset. One test image equals 0.67 points of accuracy, so the 95% confidence interval on 149 / 150 is roughly 96.3%–99.9%, wide enough to contain Lilhore et al.'s 99.2% and the top three models in Section 5.

### Worse on efficiency

ViT-B/16 is 85.9 M parameters against MobileNetV2's 2.4 M, and Phase 2 training ran at roughly 0.9 s per step on a T4 GPU, about twice the CNNs' rate. MobileNetV2 gives up 2 points of accuracy (3 extra missed fires out of 113) for a 35× smaller model. For a camera tower or drone, that trade may favour MobileNetV2; inference latency was not measured in this work.

### Unknown, probably worse, on unseen data

Hamdan et al. [11] found that a Swin Transformer's F1 fell from 0.945 to 0.612 when tested on a dataset it had not seen, a larger drop than ResNet50's. The proposed model has only been tested in-domain, so its real-world robustness is unproven.

## 8. Threats to validity

The comparison's main weakness is statistical power: a 150-image test set cannot separate models that differ by fewer than about five errors.

- **Small test set, single run.** Each notebook model was trained once with one seed; the 5-fold CV of Model C shows run-to-run spread of ±2.44 points, larger than most gaps in Section 5.
- **Test set reused.** The same test split was used to compare many models and to sweep the decision threshold, which can make the best model look slightly better than it would on fresh data.
- **No external test.** All proposed-model results are in-domain (Section 7).
- **Possible near-duplicates.** The dataset is web-scraped and was not checked for near-duplicate images across splits.
- **Explainability not shown.** The notebook's Grad-CAM step failed with an AttributeError, so there is no visual evidence that the model attends to flames rather than background cues.
- **Published figures taken as reported.** Fan et al. [7] is available only as an abstract; Hamdan et al. [11] is an un-reviewed preprint; Vuppari et al. [8] contains an internal inconsistency (its confusion matrix gives 95.1% accuracy, not the stated 96.10%).
- **Baseline training-accuracy bug.** In the notebook, the ResNet50 and MobileNetV2 *training* accuracies were computed on unpreprocessed images and are invalid. Their *test* results, used in this report, are correctly preprocessed and unaffected.

## 9. Conclusion

The proposed ViT-B/16 matches the best published result on the same dataset (99.33% vs 99.2%) and clearly outperforms every from-scratch CNN, but it does not beat a frozen ResNet50 trained on the same split. Its advantage over custom CNNs comes from large-scale pre-training, not from the transformer architecture as such. The dataset is too small and too easy to separate strong pre-trained models.

Three steps would make the comparison decisive: evaluate on an external dataset such as FlameVision or DeepFire; repeat training over at least three seeds and report mean ± standard deviation; and measure inference latency to weigh accuracy against the 35× size gap to MobileNetV2.

## 10. References

1. A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," *ICLR*, 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
2. phylake1337, "FIRE Dataset" (NASA Space Apps Challenge fire images), Kaggle. [kaggle.com/datasets/phylake1337/fire-dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset)
3. K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," *CVPR*, pp. 770–778, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
4. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," *CVPR*, pp. 4510–4520, 2018. [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)
5. N. Ahmad, M. Akbar, E. H. Alkhammash, M. M. Jamjoom, "CN2VF-Net: A Hybrid Convolutional Neural Network and Vision Transformer Framework for Multi-Scale Fire Detection in Complex Environments," *Fire*, vol. 8, no. 6, art. 211, May 2025. [doi:10.3390/fire8060211](https://doi.org/10.3390/fire8060211)
6. U. K. Lilhore et al., "Real time fire and smoke detection using vision transformers and spatiotemporal learning," *Scientific Reports*, vol. 16, art. 8928, Mar. 2026. [doi:10.1038/s41598-026-36687-9](https://doi.org/10.1038/s41598-026-36687-9)
7. X. Fan, T. Zhu, X. Zi, X. Tao, M. Pratama, C. Moreira, H. Lu, M. Prasad, "DINOv3-Driven Wildfire Classification Using Robust Visual Features Learning," *FLINS-ISKE 2026*, LNCS vol. 16754, pp. 427–439, Springer, Jul. 2026. [doi:10.1007/978-981-92-2487-6_26](https://doi.org/10.1007/978-981-92-2487-6_26)
8. G. R. Vuppari, N. Gupta, A. El-Sayed, X. Xiong, "Wildfire Detection Using Vision Transformer with the Wildfire Dataset," *2025 ASEE Northeast Section Conference*, Mar. 2025. [arXiv:2505.17395](https://arxiv.org/abs/2505.17395)
9. A. Şener, B. Ergen, K. Demir, S. Ekinci, V. Tümen, "A Hybrid Deep Learning Approach for Early Detection of Wildfires Using Transformer Models and Feature Fusion Techniques," *Electronics*, vol. 15, no. 19, art. 4470, Sep. 2026. [doi:10.3390/electronics15194470](https://doi.org/10.3390/electronics15194470)
10. M. Azhar, M. Arman, A. Iqbal, A. Amjad, D. A. Dewi, "SmartFire Vision: An Attention-Pruned Hybrid Vision Transformer and Detection Transformer Framework for Accurate, Efficient, and Real-Time Fire and Smoke Detection in Smart City Video Surveillance," *Information*, vol. 17, no. 9, art. 845, Aug. 2026. [doi:10.3390/info17090845](https://doi.org/10.3390/info17090845)
11. E. Hamdan, Y. Luo, B. U. Töreyin, E. Koyuncu, A. J. Watts, U. Güdükbay, A. E. Cetin, "A Large Scale Open-Source Image and Video Dataset for Robust Wildfire Detection and Classification," arXiv preprint, Jun. 2026. [arXiv:2606.10174](https://arxiv.org/abs/2606.10174)
