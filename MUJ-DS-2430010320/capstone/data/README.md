# Data

The dataset is **not committed** to this repository (see `.gitignore`).

- **Source:** [FIRE Dataset on Kaggle](https://www.kaggle.com/datasets/phylake1337/fire-dataset)
- **Contents:** 999 images — `fire_images/` (755) and `non_fire_images/` (244)
- **Automatic download:** the notebook calls `kagglehub.dataset_download("phylake1337/fire-dataset")`.
- **Manual download:** download the zip from Kaggle, extract it to a folder named `data/fire_dataset/` in the repository root (ignored by Git) with the two class folders inside, and set `DATA_ROOT` in the notebook to that path.
