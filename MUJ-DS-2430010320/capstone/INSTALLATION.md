# Installation Guide

## Option 1: Google Colab (recommended)

1. Go to <https://colab.research.google.com> → **File → Open notebook → GitHub** and paste this repository's URL, then pick `notebooks/wildfire_detection_vit-2.ipynb` (repository root).
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**. The first cell installs `keras-hub`; the dataset downloads automatically through `kagglehub` (no API key needed in Colab).

## Option 2: Local machine

Requirements: Python 3.11+, an NVIDIA GPU with CUDA strongly recommended (CPU works but is very slow for ViT-B/16), about 2 GB free disk space for the dataset and model weights.

```bash
git clone https://github.com/shauryagupta575/MUJ-DS-2430010320.git
cd MUJ-DS-2430010320
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the notebook:

```bash
pip install jupyter
jupyter notebook notebooks/wildfire_detection_vit-2.ipynb
```

Or run the exported script:

```bash
python code/wildfire_pipeline.py
```

### Kaggle access outside Colab

`kagglehub` needs credentials locally. Create an API token at <https://www.kaggle.com/settings> (**Create New Token**), then place `kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`). See [`capstone/data/README.md`](data/README.md) for the manual download alternative.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `keras-nlp ... requires keras-hub==...` warning | Harmless in Colab; ignore |
| Out of GPU memory during ViT fine-tuning | Lower `BATCH_SIZE` from 32 to 16 |
| Grad-CAM cell prints "Grad-CAM failed: AttributeError" | Known issue; use `make_gradcam_heatmap` from `code/fixes.py` |
| Very slow training | Check that a GPU is visible: `tf.config.list_physical_devices("GPU")` |
