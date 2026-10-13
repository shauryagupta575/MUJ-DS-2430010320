# Results

| File | Contents |
| --- | --- |
| `test_set_results.csv` | Accuracy, precision, recall, F1, macro-F1, FN, FP and parameters for every model on the 150-image test split |
| `bias_variance.csv` | Train / test accuracy, bias and variance for Models A–D, L1, L2 |
| `cross_validation_5fold.csv` | Per-fold metrics of the regularised CNN under stratified 5-fold CV |
| `ablation_study.csv` | CO5 2 × 2 ablation: SE block × focal loss |
| `screenshots/` | All figures from the notebook (class balance, curves, confusion matrices, ROC/PR, etc.) |

Values are copied from the recorded notebook outputs. Note: `18_gradcam_failed.png` shows the known Grad-CAM failure; see `code/fixes.py`.
