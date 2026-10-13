# CO5 – Innovation: SE Channel Attention and Focal Loss for Wildfire Image Classification

*Report date: 13 October 2026 · Source notebook: `wildfire_detection_vit-2.ipynb` (Level 3: Innovation & Ablation Study)*

## Contents

1. [Summary](#1-summary)
2. [Innovation statement](#2-innovation-statement)
3. [Technical justification](#3-technical-justification)
4. [Experimental design](#4-experimental-design)
5. [Ablation results](#5-ablation-results)
6. [Contribution of each component](#6-contribution-of-each-component)
7. [Diagnosis: why focal loss did not help](#7-diagnosis-why-focal-loss-did-not-help)
8. [Corrected experiment protocol](#8-corrected-experiment-protocol)
9. [Threats to validity](#9-threats-to-validity)
10. [Conclusion](#10-conclusion)
11. [References](#11-references)

## 1. Summary

The proposed innovation adds **Squeeze-and-Excitation (SE) channel attention** to a lightweight CNN and trains it with **focal loss** instead of class-weighted cross-entropy. A 2 × 2 ablation on the 150-image test split shows that **SE attention is the component that contributes**: it lifts macro-F1 from 0.458 to 0.897 and cuts false alarms from 36 to 8, for a 5.1% parameter overhead. **Focal loss, as configured, does not help**: it lowers ROC-AUC by about 5 points in both pairings, and the full SE + Focal model (macro-F1 0.816) scores below SE alone (0.897).

| Claim | Evidence | Verdict |
| --- | --- | --- |
| SE attention improves classification | Macro-F1 +0.439 over baseline; +0.072 over Focal-only | Supported, with a caveat (Section 6) |
| Focal loss improves handling of class imbalance | AUC −0.053 and −0.048; no-fire recall falls when added to SE | Not supported in current form (Section 7) |
| SE + Focal is the best configuration | Macro-F1 0.816 vs 0.897 for SE alone | Not supported |
| SE + Focal never misses a fire | Recall 1.000 (0 of 113 fires missed) | Supported, but at 17 false alarms |

The report keeps the original results unchanged, explains each failure, and gives a corrected protocol (Section 8) whose rerun would settle the remaining questions.

## 2. Innovation statement

**Problem.** Fire and no-fire images in the Kaggle dataset differ mainly in *colour and texture* (orange flames, grey smoke) rather than object shape, and the classes are imbalanced 3.09 : 1 toward fire. A plain CNN trained from 699 images must both find the right colour channels and avoid collapsing to the majority class.

**Proposed contribution.** Two changes to a baseline 3-block CNN, designed to address those two issues directly:

1. **SE channel attention after every convolutional block** — the network learns, per image, which feature channels to amplify (e.g. flame-colour detectors) and which to suppress (e.g. sky or foliage detectors).
2. **Focal loss (γ = 2, α = 0.75)** in place of class-weighted binary cross-entropy — easy, confidently classified images are down-weighted so training focuses on hard, ambiguous scenes such as sunsets, fog and smoke-like clouds.

**Scope.** The contribution is a design choice for small, imbalanced fire datasets, evaluated against its own ablations. It is not proposed as a replacement for large pre-trained backbones such as ViT-B/16 (CO4).

## 3. Technical justification

### SE channel attention

An SE block [1] summarises each feature channel by global average pooling (*squeeze*), passes the summary through a small bottleneck MLP, and rescales every channel by a learned weight between 0 and 1 (*excite*):

```math
s = \sigma\big(W_2\,\delta(W_1\,z)\big), \qquad z_c = \frac{1}{HW}\sum_{i,j} u_c(i,j), \qquad \tilde{u}_c = s_c \, u_c
```

where δ is ReLU, σ is sigmoid, and the bottleneck uses reduction ratio r = 8.

Why it fits this problem:

- **Fire evidence is channel-specific.** Early convolutional channels respond to colour and texture; letting the network up-weight flame-colour channels per image is exactly the cue the classes differ on.
- **It is cheap.** The three SE blocks add 5,628 parameters to a 110,785-parameter network (5.1%).
- **Prior wildfire work supports it.** Ul Haq et al. [3] added CBAM (channel + spatial attention) to MobileNetV2 and lost 3.1 accuracy points when the attention module was removed in their ablation.

### Focal loss

Focal loss [2] multiplies cross-entropy by a factor that shrinks as the model becomes confident:

```math
\mathrm{FL}(p_t) = -\,\alpha_t\,(1 - p_t)^{\gamma}\,\log(p_t)
```

With γ = 2, an image classified with p_t = 0.9 contributes 100× less loss than plain cross-entropy would give it. The intent here is to stop the many easy fire images from dominating training, so the hard no-fire look-alikes get more gradient. Focal loss has been used for the same purpose in recent fire detectors [4, 5].

## 4. Experimental design

**Data.** Same stratified split as the rest of the notebook: 699 train / 150 validation / 150 test images (test: 113 fire, 37 no-fire).

**Architecture (all four arms).** Input 224 × 224 → augmentation → rescale to [0, 1] → three blocks of Conv2D (32, 64, 128 filters, 3 × 3) + BatchNorm [+ SE] + MaxPool → Dropout 0.25 → GlobalAveragePooling → Dense(128, ReLU) → Dropout 0.5 → Dense(1, sigmoid).

**Ablation grid (2 × 2 factorial).**

| Arm | SE blocks | Loss | Class weights |
| --- | --- | --- | --- |
| Baseline | No | Binary cross-entropy | Yes (no-fire 2.056, fire 0.661) |
| SE only | Yes | Binary cross-entropy | Yes |
| Focal only | No | Focal (γ = 2, α = 0.75) | No |
| Proposed: SE + Focal | Yes | Focal (γ = 2, α = 0.75) | No |

**Training.** Adam, learning rate 1e-3, up to 15 epochs, early stopping on validation loss (patience 5, best weights restored), seed 42 for every arm.

**Metrics.** Fire is the positive class; threshold 0.5. Macro-F1 is the headline metric because of class imbalance. ROC-AUC is reported because it is threshold-independent and so separates *better features* from *a better-placed decision boundary*.

## 5. Ablation results

| Arm | Accuracy | Precision (fire) | Recall (fire) | Macro-F1 | ROC-AUC | Missed fires (FN) | False alarms (FP) | No-fire recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 76.00% | 0.758 | 1.000 | 0.458 | 0.975 | 0 | 36 | 2.7% |
| **SE only** | **92.67%** | **0.932** | 0.973 | **0.897** | 0.974 | 3 | **8** | **78.4%** |
| Focal only | 84.67% | 0.841 | 0.982 | 0.744 | 0.922 | 2 | 21 | 43.2% |
| Proposed: SE + Focal | 88.67% | 0.869 | **1.000** | 0.816 | 0.926 | **0** | 17 | 54.1% |

Confusion-matrix counts (FN, FP, no-fire recall) are derived from each arm's reported accuracy, precision and recall on the 113 / 37 test split.

For reference on the same test split: the 4-block regularised CNN (Model C) scored macro-F1 0.846, and the fine-tuned ViT-B/16 (Model D) scored 0.991 with AUC 1.000.

## 6. Contribution of each component

Because the ablation is a full 2 × 2 grid, the effect of each component can be separated.

| Effect | On macro-F1 | On ROC-AUC |
| --- | --- | --- |
| SE main effect (average gain from adding SE) | **+0.255** | +0.002 |
| Focal main effect (average gain from adding focal loss) | +0.103 | **−0.050** |
| SE × Focal interaction | −0.367 | +0.005 |

**SE attention fixes the decision boundary more than the features.** SE's macro-F1 gain is large, but its AUC gain is essentially zero (0.975 → 0.974). The baseline already *ranked* images almost perfectly (AUC 0.975); it failed because its outputs sat on the wrong side of 0.5, labelling 36 of 37 no-fire images as fire. SE's main measurable contribution is a model whose 0.5 threshold separates the classes: 78.4% no-fire recall vs 2.7%. That is a real practical gain, but it is not evidence that SE learned better fire features. A baseline with a validation-tuned threshold might close much of the gap; this was not tested.

**Focal loss hurts ranking quality.** Adding focal loss lowered AUC in both pairings (0.975 → 0.922 without SE; 0.974 → 0.926 with SE). Its positive macro-F1 main effect comes entirely from rescuing the collapsed baseline, which is why the interaction term is strongly negative: once SE has stabilised the model, focal loss makes it worse (macro-F1 0.897 → 0.816, false alarms 8 → 17).

**The only arm with zero missed fires is SE + Focal.** For wildfire alerting, recall matters most, and SE + Focal catches all 113 fires. It does so by flagging 46% of no-fire images as fire, so in deployment it would trade missed fires for frequent false dispatches.

## 7. Diagnosis: why focal loss did not help

Three implementation choices in the notebook explain the result.

1. **α weights the wrong class.** The code applies α = 0.75 to fire (label 1) and 0.25 to no-fire. Fire is already the majority (75%), so the loss pushes the model further toward predicting fire. Combined with the 3 : 1 class ratio, no-fire errors carry roughly one-ninth the total weight of fire errors. The focal arms also run *without* class weights, so nothing counteracts this. The result matches the data: adding focal to SE more than doubled false alarms (8 → 17).
2. **Sum instead of mean reduction.** The loss is summed over the batch. Under Adam this barely changes the update size, but it makes training and validation loss values scale with batch size, so they cannot be compared with the cross-entropy arms.
3. **Short schedule for a harder loss.** Focal loss shrinks gradients from easy examples, which slows learning; with a 15-epoch cap and patience 5, the focal arms may have stopped before converging.

A correctly balanced focal loss is available in Keras directly:

```python
loss = keras.losses.BinaryFocalCrossentropy(
    apply_class_balancing=True,
    alpha=0.25,   # weight on fire (label 1); no-fire gets 1 - alpha = 0.75
    gamma=2.0,
)                 # mean reduction by default
```

## 8. Corrected experiment protocol

The original results stay as reported above. The following rerun would test the innovation fairly; its result cells are left blank to be filled from the notebook.

| Change | Reason |
| --- | --- |
| Focal α = 0.25 on fire (0.75 on no-fire), mean reduction | Up-weight the minority class, as focal loss intends |
| Run each arm with 5 seeds (or 5-fold CV) and report mean ± SD | Single-seed differences are within run-to-run noise (±2.44 pp in Model C's CV) |
| Up to 40 epochs, patience 8 | Give focal arms time to converge |
| Choose each arm's threshold on the validation set | Separates feature quality from boundary placement |
| Report AUC and average precision alongside macro-F1 | Threshold-independent evidence of better features |
| Add a "SE + tuned-threshold" arm | Tests whether SE adds anything beyond a better threshold |

Template for the rerun (mean ± SD over 5 seeds):

| Arm | Macro-F1 | ROC-AUC | Missed fires | False alarms |
| --- | --- | --- | --- | --- |
| Baseline | | | | |
| SE only | | | | |
| Focal (corrected) only | | | | |
| SE + Focal (corrected) | | | | |
| Baseline + tuned threshold | | | | |

## 9. Threats to validity

- **Collapsed baseline.** The baseline predicted fire for 149 of 150 images. Gains measured against it overstate any component's value; the SE-vs-SE+Focal comparison is the more reliable one.
- **Single seed, small test set.** Each arm ran once; one test image equals 0.67 accuracy points.
- **Test set reused.** The ablation was scored on the same test split used throughout the notebook for model comparison.
- **Architecture differs from Model C.** The ablation network has three conv blocks; Model C has four, so the ablation baseline is not the same model reported in CO4.
- **No external dataset.** All results are in-domain.

## 10. Conclusion

The innovation is clearly defined, technically motivated and evaluated with a full 2 × 2 ablation. The quantitative results support one of its two components. **SE channel attention** is the meaningful contribution: at 5.1% extra parameters it turns a collapsed classifier (macro-F1 0.458) into a usable one (0.897, 8 false alarms). **Focal loss**, as implemented, does not contribute: it lowers ROC-AUC by about 5 points and makes the SE model worse, because its α weighting favours the majority class.

The recommended final configuration from these results is **SE attention with class-weighted cross-entropy**. Rerunning the ablation with the corrected focal loss and multiple seeds (Section 8) is the step needed to establish whether a properly balanced focal loss adds value on top of SE.

## 11. References

1. J. Hu, L. Shen, G. Sun, "Squeeze-and-Excitation Networks," *CVPR*, pp. 7132–7141, 2018. [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
2. T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár, "Focal Loss for Dense Object Detection," *ICCV*, pp. 2980–2988, 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
3. I. Ul Haq, G. Husnain, A. Iqbal, A. S. Alzahrani, "Attention-enhanced MobileNetV2 models for robust forest fire detection and classification," *Scientific Reports*, vol. 16, art. 4805, Jan. 2026. [doi:10.1038/s41598-026-35207-z](https://doi.org/10.1038/s41598-026-35207-z)
4. N. Ahmad, M. Akbar, E. H. Alkhammash, M. M. Jamjoom, "CN2VF-Net: A Hybrid Convolutional Neural Network and Vision Transformer Framework for Multi-Scale Fire Detection in Complex Environments," *Fire*, vol. 8, no. 6, art. 211, May 2025. [doi:10.3390/fire8060211](https://doi.org/10.3390/fire8060211)
5. E. Hamdan, Y. Luo, B. U. Töreyin, E. Koyuncu, A. J. Watts, U. Güdükbay, A. E. Cetin, "A Large Scale Open-Source Image and Video Dataset for Robust Wildfire Detection and Classification," arXiv preprint, Jun. 2026. [arXiv:2606.10174](https://arxiv.org/abs/2606.10174)
6. phylake1337, "FIRE Dataset" (NASA Space Apps Challenge fire images), Kaggle. [kaggle.com/datasets/phylake1337/fire-dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset)
