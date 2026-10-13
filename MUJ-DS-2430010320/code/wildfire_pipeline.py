"""Wildfire (fire / no-fire) detection pipeline.

Auto-exported from notebooks/wildfire_detection_vit-2.ipynb (code cells only, in order).
Markdown explanations and recorded outputs live in the notebook.
Shell commands (!pip ...) are commented out; install requirements.txt first.
Designed for Google Colab with a GPU.
"""


# ======================================================================
# Wildfire (Fire / Smoke) Detection from Images Using CNN
# ======================================================================


# ======================================================================
# 1. Environment Setup
# ======================================================================

# Pre-trained Vision Transformer weights for Model D come from keras-hub.
# Install BEFORE importing TensorFlow/Keras so a version change is picked up cleanly.
# !pip install -q -U keras-hub

import os, random, pathlib, itertools, shutil, collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reproducibility -- important so your reported numbers can be regenerated.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

print("TensorFlow version :", tf.__version__)
print("GPU available      :", len(tf.config.list_physical_devices("GPU")) > 0)
print("GPU devices        :", tf.config.list_physical_devices("GPU"))


# ======================================================================
# 2. Download the Dataset
# ======================================================================

# --- Option 1: kagglehub (recommended) ---
# !pip install -q kagglehub

import kagglehub
path = kagglehub.dataset_download("phylake1337/fire-dataset")
print("Downloaded to:", path)

# Locate the folder that actually contains the two class directories.
DATA_ROOT = path
for root, dirs, files in os.walk(path):
    if "fire_images" in dirs and "non_fire_images" in dirs:
        DATA_ROOT = root
        break

print("DATA_ROOT :", DATA_ROOT)
print("Contents  :", sorted(os.listdir(DATA_ROOT)))


# ======================================================================
# 3. Dataset Inventory and Integrity Check
# ======================================================================

FIRE_DIR    = os.path.join(DATA_ROOT, "fire_images")
NOFIRE_DIR  = os.path.join(DATA_ROOT, "non_fire_images")

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

def list_images(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(VALID_EXT)])

fire_paths_all   = list_images(FIRE_DIR)
nofire_paths_all = list_images(NOFIRE_DIR)

print(f"fire_images      : {len(fire_paths_all)}")
print(f"non_fire_images  : {len(nofire_paths_all)}")
print(f"TOTAL            : {len(fire_paths_all) + len(nofire_paths_all)}")

# This dataset is known to contain at least one file that is not a decodable
# image (a stray non-image saved with an image extension). A single bad file
# will crash training mid-epoch, so we filter them out ONCE, up front.

def is_decodable(p):
    try:
        raw = tf.io.read_file(p)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        return img.shape.rank == 3 and img.shape[0] > 0 and img.shape[1] > 0
    except Exception:
        return False

bad = []
fire_paths, nofire_paths = [], []
for p in fire_paths_all:
    (fire_paths if is_decodable(p) else bad).append(p)
for p in nofire_paths_all:
    (nofire_paths if is_decodable(p) else bad).append(p)

print("Unreadable / non-image files removed:", len(bad))
for b in bad:
    print("   dropped ->", os.path.basename(b))

print(f"\nUsable fire     : {len(fire_paths)}")
print(f"Usable no_fire  : {len(nofire_paths)}")

# Class-name convention: index 0 = no_fire, index 1 = fire  (fire is the POSITIVE class,
# because "detecting a fire" is the event we care about).
CLASS_NAMES = ["no_fire", "fire"]

all_paths  = np.array(nofire_paths + fire_paths)
all_labels = np.array([0] * len(nofire_paths) + [1] * len(fire_paths))

n_pos = int(all_labels.sum())
n_neg = int((all_labels == 0).sum())
imbalance_ratio = n_pos / n_neg

print(f"no_fire (0) : {n_neg}")
print(f"fire    (1) : {n_pos}")
print(f"Imbalance ratio (fire : no_fire) = {imbalance_ratio:.2f} : 1")

majority_baseline = max(n_pos, n_neg) / len(all_labels)
print(f"\nMajority-class baseline accuracy to beat: {majority_baseline*100:.2f}%")
print("A model that blindly predicts 'fire' for every image already scores this.")


# ======================================================================
# Observation 1 — the dataset is imbalanced, and that decides our metrics
# ======================================================================

# Visual: class balance
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))

ax[0].bar(CLASS_NAMES, [n_neg, n_pos], color=["#55A868", "#C44E52"])
for i, v in enumerate([n_neg, n_pos]):
    ax[0].text(i, v + 8, str(v), ha="center", fontweight="bold")
ax[0].set_ylabel("number of images")
ax[0].set_title("Class distribution (absolute)")
ax[0].grid(axis="y", alpha=.3)

ax[1].pie([n_neg, n_pos], labels=CLASS_NAMES, autopct="%1.1f%%",
          colors=["#55A868", "#C44E52"], startangle=90,
          wedgeprops={"edgecolor": "white", "linewidth": 2})
ax[1].set_title("Class distribution (proportion)")

plt.tight_layout(); plt.show()

# Sample images -- always look at your data before you model it.
fig, axes = plt.subplots(2, 5, figsize=(16, 6.8))

for j, p in enumerate(random.sample(fire_paths, 5)):
    img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
    axes[0, j].imshow(img.numpy()); axes[0, j].axis("off")
    axes[0, j].set_title("fire", color="#C44E52", fontweight="bold")

for j, p in enumerate(random.sample(nofire_paths, 5)):
    img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
    axes[1, j].imshow(img.numpy()); axes[1, j].axis("off")
    axes[1, j].set_title("no_fire", color="#55A868", fontweight="bold")

plt.suptitle("Top row: fire   |   Bottom row: no_fire", y=1.02, fontsize=13)
plt.tight_layout(); plt.show()

# Image dimensions vary a lot -- confirm that resizing is necessary.
sizes = []
for p in random.sample(list(all_paths), min(200, len(all_paths))):
    img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
    sizes.append((img.shape[1], img.shape[0]))       # (width, height)

sz = pd.DataFrame(sizes, columns=["width", "height"])
print(sz.describe().round(1))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(sz["width"], sz["height"], alpha=.45, s=18, color="#4C72B0")
ax[0].set_xlabel("width (px)"); ax[0].set_ylabel("height (px)")
ax[0].set_title("Raw image dimensions are not uniform"); ax[0].grid(alpha=.3)

ax[1].hist(sz["width"] / sz["height"], bins=25, color="#DD8452", edgecolor="white")
ax[1].set_xlabel("aspect ratio (w/h)"); ax[1].set_ylabel("count")
ax[1].set_title("Aspect-ratio spread"); ax[1].grid(axis="y", alpha=.3)

plt.tight_layout(); plt.show()
print("\n-> Every image must be resized to a fixed shape before batching.")


# ======================================================================
# 4. Stratified Train / Validation / Test Split
# ======================================================================

from sklearn.model_selection import train_test_split

# First carve off the test set, then split the remainder into train/val.
paths_trainval, paths_test, y_trainval, y_test = train_test_split(
    all_paths, all_labels, test_size=0.15, stratify=all_labels, random_state=SEED)

paths_train, paths_val, y_train, y_val = train_test_split(
    paths_trainval, y_trainval, test_size=0.1765,      # 0.1765 * 0.85 ~= 0.15
    stratify=y_trainval, random_state=SEED)

def describe(name, y):
    n = len(y); pos = int(y.sum())
    print(f"{name:<12} n={n:<5} fire={pos:<5} no_fire={n-pos:<5} "
          f"fire%={pos/n*100:5.1f}%")

print("Stratified split (fire% should be near-identical across all three):\n")
describe("TRAIN", y_train)
describe("VAL",   y_val)
describe("TEST",  y_test)
describe("TOTAL", all_labels)

IMG_SIZE   = (224, 224)          # 224 so Model D can reuse ViT-B/16 (16x16 patches -> 14x14 grid) weights directly
BATCH_SIZE = 32
AUTOTUNE   = tf.data.AUTOTUNE

def load_and_resize(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img.set_shape(IMG_SIZE + (3,))
    return img, tf.cast(label, tf.float32)

def make_ds(paths, labels, shuffle=False, batch_size=BATCH_SIZE):
    # Build a tf.data pipeline from a list of file paths.
    # Note: images are NOT rescaled here. Rescaling lives *inside* each model so
    # that the exact same pipeline can feed both the custom CNNs (which expect
    # 0-1 inputs) and the Vision Transformer (which expects its own [-1, 1] scaling).
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_and_resize, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)

# shuffle=True only for training. val/test stay in a FIXED order so that
# predictions line up with y_val / y_test index-for-index.
train_ds = make_ds(paths_train, y_train, shuffle=True)
val_ds   = make_ds(paths_val,   y_val)
test_ds  = make_ds(paths_test,  y_test)

print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Val batches  :", tf.data.experimental.cardinality(val_ds).numpy())
print("Test batches :", tf.data.experimental.cardinality(test_ds).numpy())

xb, yb = next(iter(train_ds))
print("\nBatch image tensor:", xb.shape, xb.dtype, " range:", float(tf.reduce_min(xb)), "-", float(tf.reduce_max(xb)))
print("Batch label tensor:", yb.shape, yb.dtype)


# ======================================================================
# Class weights
# ======================================================================

from sklearn.utils.class_weight import compute_class_weight

cw = compute_class_weight(class_weight="balanced",
                          classes=np.array([0, 1]), y=y_train)
class_weight = {0: float(cw[0]), 1: float(cw[1])}

print("class_weight =", {k: round(v, 4) for k, v in class_weight.items()})
print(f"\nThe minority class (no_fire) is up-weighted {cw[0]/cw[1]:.2f}x relative to fire,")
print("so a mistake on a no_fire image costs the loss function proportionally more.")
print("Without this, the cheapest strategy for the optimiser is to lean on 'fire'.")


# ======================================================================
# Augmentation layer (used only by Models C and D)
# ======================================================================

# Deliberately conservative: fires are not upside-down, so no vertical flip.
# Brightness/contrast jitter matters here because the real-world variation is
# camera exposure and time of day.
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.12),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.12, value_range=(0, 255)),
], name="data_augmentation")

# Preview the effect on a single fire image.
sample_img, _ = next(iter(make_ds(paths_train[:1], y_train[:1], batch_size=1)))
fig, axes = plt.subplots(1, 6, figsize=(17, 3.1))
axes[0].imshow(sample_img[0].numpy().astype("uint8")); axes[0].set_title("original"); axes[0].axis("off")
for i in range(1, 6):
    aug = data_augmentation(sample_img, training=True)
    axes[i].imshow(tf.clip_by_value(aug[0], 0, 255).numpy().astype("uint8"))
    axes[i].set_title(f"aug {i}"); axes[i].axis("off")
plt.suptitle("Augmentation expands the effective training set without new labels", y=1.06)
plt.tight_layout(); plt.show()


# ======================================================================
# 5. Shared Training / Evaluation Harness
# ======================================================================

RESULTS   = {}   # collects metrics for the final comparison table
HISTORIES = {}   # collects training curves

def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def evaluate_and_record(model, name, history, notes="", eval_train=None, eval_test=None):
    # Evaluate on train + test and store bias/variance figures.
    tr = model.evaluate(eval_train if eval_train is not None else train_ds,
                        verbose=0, return_dict=True)
    te = model.evaluate(eval_test if eval_test is not None else test_ds,
                        verbose=0, return_dict=True)

    tr_acc, te_acc = tr["accuracy"], te["accuracy"]

    RESULTS[name] = {
        "Train Acc (%)":  round(tr_acc * 100, 2),
        "Test Acc (%)":   round(te_acc * 100, 2),
        "Bias (%)":       round((1 - tr_acc) * 100, 2),
        "Variance (%)":   round((tr_acc - te_acc) * 100, 2),
        "Train Loss":     round(tr["loss"], 4),
        "Test Loss":      round(te["loss"], 4),
        "Test Precision": round(te["precision"], 3),
        "Test Recall":    round(te["recall"], 3),
        "Test AUC":       round(te["auc"], 3),
        "Params":         model.count_params(),
        "Notes":          notes,
    }
    HISTORIES[name] = history.history if history is not None else None

    print(f"\n--- {name} ---")
    print(f"Training accuracy : {tr_acc*100:6.2f}%   (loss {tr['loss']:.4f})")
    print(f"Testing accuracy  : {te_acc*100:6.2f}%   (loss {te['loss']:.4f})")
    print(f"Bias  (100 - train acc)     : {(1-tr_acc)*100:6.2f}%")
    print(f"Variance (train - test acc) : {(tr_acc-te_acc)*100:6.2f}%")
    return RESULTS[name]


def plot_history(history, title, extra_history=None):
    # Plot accuracy and loss curves. extra_history appends a fine-tune phase.
    h = dict(history.history)
    if extra_history is not None:
        for k, v in extra_history.history.items():
            if k in h:
                h[k] = h[k] + v
    epochs = range(1, len(h["accuracy"]) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    ax[0].plot(epochs, h["accuracy"],     "o-", ms=4, label="train", color="#4C72B0")
    ax[0].plot(epochs, h["val_accuracy"], "o-", ms=4, label="validation", color="#C44E52")
    ax[0].set_title(f"{title} - Accuracy"); ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("accuracy"); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(epochs, h["loss"],     "o-", ms=4, label="train", color="#4C72B0")
    ax[1].plot(epochs, h["val_loss"], "o-", ms=4, label="validation", color="#C44E52")
    ax[1].set_title(f"{title} - Loss (binary crossentropy)"); ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("loss"); ax[1].legend(); ax[1].grid(alpha=.3)

    # mark the epoch of minimum validation loss -- the "global minimum" that
    # EarlyStopping(restore_best_weights=True) is designed to recover.
    best = int(np.argmin(h["val_loss"])) + 1
    ax[1].axvline(best, ls=":", c="green", lw=1.6)
    ax[1].annotate(f"min val_loss\nepoch {best}", xy=(best, min(h["val_loss"])),
                   xytext=(6, 14), textcoords="offset points", fontsize=8.5, color="green")

    if extra_history is not None:
        split_at = len(history.history["accuracy"]) + 0.5
        for a in ax:
            a.axvline(split_at, ls="--", c="grey")
            a.text(split_at + 0.1, a.get_ylim()[0], " fine-tune", fontsize=9, color="grey")

    plt.tight_layout(); plt.show()


# ======================================================================
# 6. Model A — Deliberate **Underfitting** (High Bias)
# ======================================================================

def build_model_A():
    model = keras.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        layers.Rescaling(1./255),
        layers.Conv2D(8, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation="sigmoid"),
    ], name="Model_A_Underfit")
    return model

model_A = compile_model(build_model_A(), lr=1e-4)   # low LR compounds the underfitting
model_A.summary()

hist_A = model_A.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weight,
    verbose=1,
)

plot_history(hist_A, "Model A (Underfitting)")
evaluate_and_record(model_A, "A: Underfit CNN", hist_A,
                    notes="1 conv layer (8 filters), GAP head, lr=1e-4, 15 epochs")


# ======================================================================
# Reading the Model A result
# ======================================================================


# ======================================================================
# 7. Model B — Deliberate **Overfitting** (High Variance)
# ======================================================================

def build_model_B():
    model = keras.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        layers.Rescaling(1./255),

        layers.Conv2D(32, 3, activation="relu"),  layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),  layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(512, activation="relu"),   # deliberately large
        layers.Dense(1, activation="sigmoid"),
    ], name="Model_B_Overfit")
    return model

model_B = compile_model(build_model_B(), lr=1e-3)
model_B.summary()

hist_B = model_B.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    class_weight=class_weight,
    verbose=1,               # NOTE: no callbacks -- overfitting is the goal here
)

plot_history(hist_B, "Model B (Overfitting)")
evaluate_and_record(model_B, "B: Overfit CNN", hist_B,
                    notes="4 conv blocks + Dense(512), no dropout/aug/ES, 25 epochs")

# Pinpoint the exact epoch where overfitting begins.
vl = hist_B.history["val_loss"]
tl = hist_B.history["loss"]
best_epoch = int(np.argmin(vl)) + 1

print(f"Minimum validation loss : {min(vl):.4f}  at epoch {best_epoch}")
print(f"Final validation loss   : {vl[-1]:.4f}  at epoch {len(vl)}")
print(f"Degradation after the minimum : {vl[-1] - min(vl):+.4f}")
print(f"\nTraining loss meanwhile fell from {tl[0]:.4f} to {tl[-1]:.4f}.")
print(f"\n>>> Everything after epoch {best_epoch} is pure memorisation: the model keeps")
print(">>> improving on data it has seen while getting WORSE on data it has not.")
print(f">>> Early stopping with restore_best_weights=True would have returned the")
print(f">>> epoch-{best_epoch} weights -- i.e. the global minimum of the validation loss.")

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(tl) + 1), tl, "o-", ms=4, label="train loss", color="#4C72B0")
plt.plot(range(1, len(vl) + 1), vl, "o-", ms=4, label="validation loss", color="#C44E52")
plt.axvline(best_epoch, ls="--", c="green", label=f"min val_loss (epoch {best_epoch})")
plt.fill_between(range(best_epoch, len(vl) + 1), min(vl), max(vl),
                 color="#C44E52", alpha=.08, label="overfitting zone")
plt.xlabel("epoch"); plt.ylabel("binary crossentropy")
plt.title("Model B - the overfitting gap widens epoch after epoch")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()


# ======================================================================
# 8. Model C — **Regularised** CNN (the fix for Model B)
# ======================================================================

def build_model_C():
    model = keras.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        data_augmentation,                       # <-- change 1
        layers.Rescaling(1./255),

        layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1"),
        layers.BatchNormalization(), layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2"),
        layers.BatchNormalization(), layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3"),
        layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Dropout(0.25),                    # <-- change 2

        layers.Conv2D(128, 3, padding="same", activation="relu", name="conv4"),
        layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.GlobalAveragePooling2D(),         # far fewer params than Flatten
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ], name="Model_C_Regularised")
    return model

model_C = compile_model(build_model_C(), lr=1e-3)
model_C.summary()

print(f"\nModel B parameters : {model_B.count_params():,}")
print(f"Model C parameters : {model_C.count_params():,}")
print(f"Reduction          : {(1 - model_C.count_params()/model_B.count_params())*100:.1f}%")

callbacks_C = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,       # <-- change 3
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                      patience=4, min_lr=1e-6, verbose=1),
]

hist_C = model_C.fit(
    train_ds,
    validation_data=val_ds,
    epochs=60,               # generous on purpose -- EarlyStopping decides the real number
    class_weight=class_weight,
    callbacks=callbacks_C,
    verbose=1,
)

print(f"\nRequested epochs : 60")
print(f"Actually ran     : {len(hist_C.history['loss'])}")
print(f"Best epoch (min val_loss, weights restored to it): {int(np.argmin(hist_C.history['val_loss']))+1}")

plot_history(hist_C, "Model C (Regularised)")
evaluate_and_record(model_C, "C: Regularised CNN", hist_C,
                    notes="BN + dropout(.25/.5) + augmentation + early stopping")


# ======================================================================
# Reading the Model C result
# ======================================================================


# ======================================================================
# 9. L1 vs L2 Regularisation — Which One To Take
# ======================================================================

from tensorflow.keras.regularizers import l1, l2

def build_model_reg(reg, name):
    # Model C's conv/dense skeleton, with a weight penalty INSTEAD of dropout,
    # so the effect measured is the penalty's alone.
    model = keras.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        data_augmentation,
        layers.Rescaling(1./255),

        layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=reg),
        layers.BatchNormalization(), layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=reg),
        layers.BatchNormalization(), layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=reg),
        layers.BatchNormalization(), layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=reg),
        layers.BatchNormalization(), layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu", kernel_regularizer=reg),
        layers.Dense(1, activation="sigmoid"),
    ], name=name)
    return model

LAMBDA = 1e-4

def train_reg(reg, name):
    m = compile_model(build_model_reg(reg, name), lr=1e-3)
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                        restore_best_weights=True, verbose=0)]
    h = m.fit(train_ds, validation_data=val_ds, epochs=40,
              class_weight=class_weight, callbacks=cb, verbose=1)
    print(f"{name}: ran {len(h.history['loss'])} epochs, "
          f"best epoch {int(np.argmin(h.history['val_loss']))+1}")
    return m, h

model_L1, hist_L1 = train_reg(l1(LAMBDA), "Model_L1")

plot_history(hist_L1, "L1 regularisation (lambda=1e-4)")
evaluate_and_record(model_L1, "L1: lambda=1e-4", hist_L1,
                    notes=f"kernel_regularizer=l1({LAMBDA}), no dropout")

model_L2, hist_L2 = train_reg(l2(LAMBDA), "Model_L2")

plot_history(hist_L2, "L2 regularisation (lambda=1e-4)")
evaluate_and_record(model_L2, "L2: lambda=1e-4", hist_L2,
                    notes=f"kernel_regularizer=l2({LAMBDA}), no dropout")

# Quantify the sparsity claim: what fraction of learned weights is effectively zero?
def sparsity(model, thresh=1e-3):
    tot = zero = 0
    for lyr in model.layers:
        for w in lyr.trainable_weights:
            if "kernel" in w.name:
                a = w.numpy()
                tot  += a.size
                zero += int(np.sum(np.abs(a) < thresh))
    return zero / tot * 100, tot

sp_l1, n_l1 = sparsity(model_L1)
sp_l2, n_l2 = sparsity(model_L2)

print(f"Weights with |w| < 1e-3  ('effectively zero'):")
print(f"  L1 model : {sp_l1:6.2f}%  of {n_l1:,} kernel weights")
print(f"  L2 model : {sp_l2:6.2f}%  of {n_l2:,} kernel weights")

# Weight-magnitude distributions make the mechanism visible.
def kernels(model):
    return np.concatenate([w.numpy().ravel() for lyr in model.layers
                           for w in lyr.trainable_weights if "kernel" in w.name])

w1, w2 = kernels(model_L1), kernels(model_L2)

fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
bins = np.linspace(-0.25, 0.25, 120)
ax[0].hist(w1, bins=bins, color="#DD8452", alpha=.85, label=f"L1 ({sp_l1:.1f}% ~zero)")
ax[0].hist(w2, bins=bins, color="#4C72B0", alpha=.55, label=f"L2 ({sp_l2:.1f}% ~zero)")
ax[0].set_yscale("log"); ax[0].set_xlabel("weight value"); ax[0].set_ylabel("count (log)")
ax[0].set_title("Weight distributions: L1 spikes at zero, L2 is a smooth bell")
ax[0].legend(); ax[0].grid(alpha=.3)

ax[1].plot(range(1, len(hist_L1.history["val_loss"])+1), hist_L1.history["val_loss"],
           "o-", ms=4, label="L1 val_loss", color="#DD8452")
ax[1].plot(range(1, len(hist_L2.history["val_loss"])+1), hist_L2.history["val_loss"],
           "o-", ms=4, label="L2 val_loss", color="#4C72B0")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation loss")
ax[1].set_title("Validation loss: L1 vs L2"); ax[1].legend(); ax[1].grid(alpha=.3)

plt.tight_layout(); plt.show()

# Side-by-side verdict table
l1l2 = pd.DataFrame({
    "L1": RESULTS["L1: lambda=1e-4"],
    "L2": RESULTS["L2: lambda=1e-4"],
}).T[["Train Acc (%)", "Test Acc (%)", "Bias (%)", "Variance (%)",
      "Test Loss", "Test Precision", "Test Recall", "Test AUC"]]
print(l1l2)

winner = "L2" if RESULTS["L2: lambda=1e-4"]["Test Acc (%)"] >= RESULTS["L1: lambda=1e-4"]["Test Acc (%)"] else "L1"
print(f"\nBetter test accuracy on this run: {winner}")
print(f"Variance  L1 = {RESULTS['L1: lambda=1e-4']['Variance (%)']:.2f}%   "
      f"L2 = {RESULTS['L2: lambda=1e-4']['Variance (%)']:.2f}%")
print(f"Test loss L1 = {RESULTS['L1: lambda=1e-4']['Test Loss']:.4f}   "
      f"L2 = {RESULTS['L2: lambda=1e-4']['Test Loss']:.4f}")


# ======================================================================
# Which one to take — and why
# ======================================================================


# ======================================================================
# 10. Cross Validation (k = 5)
# ======================================================================

from sklearn.model_selection import StratifiedKFold

# Pool train + val; the test split stays untouched.
cv_paths  = np.concatenate([paths_train, paths_val])
cv_labels = np.concatenate([y_train, y_val])
print(f"CV pool: {len(cv_paths)} images "
      f"(fire={int(cv_labels.sum())}, no_fire={int((cv_labels==0).sum())})")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

fold_rows = []
fold_histories = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(cv_paths, cv_labels), start=1):
    print(f"\n{'='*58}\nFOLD {fold}/5\n{'='*58}")

    ds_tr = make_ds(cv_paths[tr_idx], cv_labels[tr_idx], shuffle=True)
    ds_va = make_ds(cv_paths[va_idx], cv_labels[va_idx])

    cwf = compute_class_weight("balanced", classes=np.array([0, 1]), y=cv_labels[tr_idx])
    cwf = {0: float(cwf[0]), 1: float(cwf[1])}

    keras.backend.clear_session()
    tf.random.set_seed(SEED + fold)
    m = compile_model(build_model_C(), lr=1e-3)

    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                       restore_best_weights=True, verbose=0)
    h = m.fit(ds_tr, validation_data=ds_va, epochs=30,
              class_weight=cwf, callbacks=[es], verbose=0)

    res = m.evaluate(ds_va, verbose=0, return_dict=True)
    prec, rec = res["precision"], res["recall"]
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    fold_rows.append({
        "Fold": fold,
        "Epochs run": len(h.history["loss"]),
        "Val Acc (%)": round(res["accuracy"] * 100, 2),
        "Val Loss": round(res["loss"], 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "AUC": round(res["auc"], 4),
    })
    fold_histories.append(h.history)
    print(f"Fold {fold}: acc={res['accuracy']*100:.2f}%  loss={res['loss']:.4f}  "
          f"F1={f1:.4f}  AUC={res['auc']:.4f}  ({len(h.history['loss'])} epochs)")

cv_df = pd.DataFrame(fold_rows)
print(cv_df)

acc  = cv_df["Val Acc (%)"].values
loss = cv_df["Val Loss"].values
f1s  = cv_df["F1"].values

print("=" * 58)
print("5-FOLD CROSS-VALIDATION SUMMARY")
print("=" * 58)
print(f"Accuracy : {acc.mean():6.2f}%  +/- {acc.std():.2f}   "
      f"(min {acc.min():.2f}%, max {acc.max():.2f}%)")
print(f"Loss     : {loss.mean():6.4f}  +/- {loss.std():.4f}")
print(f"F1       : {f1s.mean():6.4f}  +/- {f1s.std():.4f}")
print(f"\nSpread (max - min accuracy): {acc.max() - acc.min():.2f} percentage points")

CV_MEAN, CV_STD = acc.mean(), acc.std()
if CV_STD < 2.0:
    print("\n-> Std < 2pp: the single-split results in this notebook are RELIABLE.")
else:
    print("\n-> Std >= 2pp: single-split numbers are noisy; quote the CV mean +/- std instead.")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))

# (a) per-fold accuracy vs the mean
ax[0].bar(cv_df["Fold"], cv_df["Val Acc (%)"], color="#4C72B0", width=.62)
ax[0].axhline(CV_MEAN, ls="--", c="#C44E52", label=f"mean = {CV_MEAN:.2f}%")
ax[0].fill_between([0.4, 5.6], CV_MEAN - CV_STD, CV_MEAN + CV_STD,
                   color="#C44E52", alpha=.15, label=f"+/- 1 std ({CV_STD:.2f})")
for i, v in enumerate(cv_df["Val Acc (%)"], start=1):
    ax[0].text(i, v + .4, f"{v:.1f}", ha="center", fontsize=8.5)
ax[0].set_xlim(.4, 5.6); ax[0].set_ylim(min(acc) - 6, 101)
ax[0].set_xlabel("fold"); ax[0].set_ylabel("validation accuracy (%)")
ax[0].set_title("(a) Per-fold accuracy"); ax[0].legend(fontsize=8.5); ax[0].grid(axis="y", alpha=.3)

# (b) per-fold loss
ax[1].bar(cv_df["Fold"], cv_df["Val Loss"], color="#DD8452", width=.62)
ax[1].axhline(loss.mean(), ls="--", c="black", label=f"mean = {loss.mean():.4f}")
ax[1].set_xlabel("fold"); ax[1].set_ylabel("validation loss")
ax[1].set_title("(b) Per-fold loss"); ax[1].legend(fontsize=8.5); ax[1].grid(axis="y", alpha=.3)

# (c) all five loss curves -- shows run-to-run stability of training itself
for i, h in enumerate(fold_histories, start=1):
    ax[2].plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], "o-", ms=3, label=f"fold {i}")
ax[2].set_xlabel("epoch"); ax[2].set_ylabel("validation loss")
ax[2].set_title("(c) Loss vs epoch, all 5 folds"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

plt.tight_layout(); plt.show()


# ======================================================================
# 11. Model D — Transfer Learning with a Vision Transformer (ViT-B/16)
# ======================================================================

# keras-hub provides pre-trained Vision Transformers for Keras 3 / TensorFlow.
# (It is installed in Section 1; the import is repeated here for clarity.)
import keras_hub

VIT_PRESET = "vit_base_patch16_224_imagenet"   # ViT-B/16, ImageNet-21k -> 1k
print("keras-hub version:", keras_hub.__version__)
print("ViT preset       :", VIT_PRESET)

vit_converter = keras_hub.layers.ViTImageConverter.from_preset(VIT_PRESET)

def make_vit_ds(paths, labels, shuffle=False, augment=False, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_and_resize, num_parallel_calls=AUTOTUNE).batch(batch_size)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (vit_converter(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

train_ds_vit = make_vit_ds(paths_train, y_train, shuffle=True, augment=True)
val_ds_vit   = make_vit_ds(paths_val,   y_val)
test_ds_vit  = make_vit_ds(paths_test,  y_test)

# For an honest train-accuracy figure, evaluate on the UN-augmented training data.
train_ds_vit_eval = make_vit_ds(paths_train, y_train)

print("ViT pipelines ready.")
xb, _ = next(iter(train_ds_vit))
print("Preprocessed batch range:", float(tf.reduce_min(xb)), "to", float(tf.reduce_max(xb)))

# ---------------- PHASE 1: frozen feature extraction ----------------
vit_backbone = keras_hub.models.ViTBackbone.from_preset(VIT_PRESET)
vit_backbone.trainable = False

inputs  = layers.Input(shape=IMG_SIZE + (3,), name="image")
tokens  = vit_backbone(inputs)                       # (batch, 197, 768) = [CLS] + 196 patches
cls_tok = tokens[:, 0, :]                            # [CLS] token -> (batch, 768)
x       = layers.Dense(128, activation="gelu")(cls_tok)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model_D = keras.Model(inputs, outputs, name="Model_D_ViT_B16")

model_D = compile_model(model_D, lr=1e-3)
model_D.summary()

trainable = int(np.sum([np.prod(w.shape) for w in model_D.trainable_weights]))
print(f"\nPhase 1 trainable parameters: {trainable:,} "
      f"of {model_D.count_params():,} total "
      f"({trainable/model_D.count_params()*100:.2f}%)")

es_D1 = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                      restore_best_weights=True, verbose=1)

hist_D1 = model_D.fit(train_ds_vit, validation_data=val_ds_vit,
                      epochs=20, class_weight=class_weight,
                      callbacks=[es_D1], verbose=1)

# ---------------- PHASE 2: unfreeze the last 4 encoder blocks and fine-tune ----------------
N_UNFREEZE = 4   # of 12 Transformer encoder blocks

vit_backbone.trainable = True
for lyr in vit_backbone.layers:              # patch embedding etc. stay frozen
    lyr.trainable = False
encoder = vit_backbone.get_layer("vit_encoder")
encoder.trainable = True
for blk in encoder.encoder_layers[:-N_UNFREEZE]:
    blk.trainable = False                    # early blocks stay frozen
# encoder.layer_norm (final LayerNorm) stays trainable

# Re-compile is MANDATORY after changing .trainable, and the learning rate must
# drop by ~100x so the pre-trained attention weights are nudged, not wrecked.
model_D = compile_model(model_D, lr=1e-5)

trainable = int(np.sum([np.prod(w.shape) for w in model_D.trainable_weights]))
print(f"Phase 2 trainable parameters: {trainable:,} "
      f"({trainable/model_D.count_params()*100:.2f}% of total)")
print(f"Unfrozen encoder blocks: last {N_UNFREEZE} of {len(encoder.encoder_layers)}")

callbacks_D2 = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                      patience=4, min_lr=1e-8, verbose=1),
]

hist_D2 = model_D.fit(train_ds_vit, validation_data=val_ds_vit,
                      epochs=25, class_weight=class_weight,
                      callbacks=callbacks_D2, verbose=1)

plot_history(hist_D1, "Model D (ViT-B/16 transfer + fine-tune)", extra_history=hist_D2)
evaluate_and_record(model_D, "D: ViT-B/16 Fine-tuned", hist_D1,
                    notes="ViT-B/16 frozen -> last 4 encoder blocks fine-tune @1e-5, dropout .4, ES",
                    eval_train=train_ds_vit_eval, eval_test=test_ds_vit)

# Store the combined curve so the all-models comparison plot includes both phases.
combined = dict(hist_D1.history)
for k, v in hist_D2.history.items():
    if k in combined:
        combined[k] = combined[k] + v
HISTORIES["D: ViT-B/16 Fine-tuned"] = combined


# ======================================================================
# 12. Classification Metrics — Precision, Recall, Accuracy, F1
# ======================================================================

from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, auc, precision_recall_curve, average_precision_score)

y_true = y_test.astype(int)      # fixed order, matches the unshuffled test pipelines

y_prob_C = model_C.predict(test_ds,     verbose=0).ravel()
y_prob_D = model_D.predict(test_ds_vit, verbose=0).ravel()

y_pred_C = (y_prob_C >= 0.5).astype(int)
y_pred_D = (y_prob_D >= 0.5).astype(int)

print("Test set size:", len(y_true))
print("Label distribution:", dict(zip(CLASS_NAMES, np.bincount(y_true))))
print(f"Majority baseline on this test split: {max(np.bincount(y_true))/len(y_true)*100:.2f}%")

print("=" * 66)
print("MODEL C - Regularised CNN")
print("=" * 66)
print(classification_report(y_true, y_pred_C, target_names=CLASS_NAMES, digits=4))

print("=" * 66)
print("MODEL D - ViT-B/16 Fine-tuned")
print("=" * 66)
print(classification_report(y_true, y_pred_D, target_names=CLASS_NAMES, digits=4))

# Explicit metric table -- the four numbers the brief asks for, plus macro-F1.
def metric_row(y_t, y_p, name):
    return {
        "Model": name,
        "Accuracy":         round(accuracy_score(y_t, y_p), 4),
        "Precision (fire)": round(precision_score(y_t, y_p, pos_label=1, zero_division=0), 4),
        "Recall (fire)":    round(recall_score(y_t, y_p, pos_label=1, zero_division=0), 4),
        "F1 (fire)":        round(f1_score(y_t, y_p, pos_label=1, zero_division=0), 4),
        "F1 (no_fire)":     round(f1_score(y_t, y_p, pos_label=0, zero_division=0), 4),
        "F1 (macro)":       round(f1_score(y_t, y_p, average="macro", zero_division=0), 4),
        "F1 (weighted)":    round(f1_score(y_t, y_p, average="weighted", zero_division=0), 4),
    }

metrics_df = pd.DataFrame([
    metric_row(y_true, y_pred_C, "C: Regularised CNN"),
    metric_row(y_true, y_pred_D, "D: ViT-B/16 Fine-tuned"),
]).set_index("Model")

print(metrics_df)

print("Because the data is imbalanced, MACRO F1 is the number to quote:")
for m in metrics_df.index:
    print(f"  {m:<22} macro-F1 = {metrics_df.loc[m, 'F1 (macro)']:.4f}   "
          f"(accuracy = {metrics_df.loc[m, 'Accuracy']:.4f})")
print("\nMacro-F1 averages the two classes EQUALLY, so the minority class (no_fire)")
print("cannot be quietly sacrificed the way it can be under plain accuracy.")

# Metric comparison chart
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))

cols = ["Accuracy", "Precision (fire)", "Recall (fire)", "F1 (fire)", "F1 (macro)"]
xpos = np.arange(len(cols)); w = 0.36
ax[0].bar(xpos - w/2, metrics_df.loc["C: Regularised CNN", cols].values.astype(float),
          w, label="C: Regularised CNN", color="#55A868")
ax[0].bar(xpos + w/2, metrics_df.loc["D: ViT-B/16 Fine-tuned", cols].values.astype(float),
          w, label="D: ViT-B/16 Fine-tuned", color="#4C72B0")
ax[0].set_xticks(xpos); ax[0].set_xticklabels(cols, rotation=20, ha="right")
ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("score")
ax[0].set_title("Classification metrics on the held-out test split")
ax[0].legend(fontsize=9); ax[0].grid(axis="y", alpha=.3)

# Per-class F1 -- makes the imbalance effect explicit
f1_by_class = pd.DataFrame({
    "C": [metrics_df.loc["C: Regularised CNN", "F1 (no_fire)"],
          metrics_df.loc["C: Regularised CNN", "F1 (fire)"]],
    "D": [metrics_df.loc["D: ViT-B/16 Fine-tuned", "F1 (no_fire)"],
          metrics_df.loc["D: ViT-B/16 Fine-tuned", "F1 (fire)"]],
}, index=CLASS_NAMES)
f1_by_class.plot(kind="bar", ax=ax[1], color=["#55A868", "#4C72B0"], rot=0)
ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("F1 score")
ax[1].set_title("Per-class F1 (minority class is the one to watch)")
ax[1].grid(axis="y", alpha=.3)

plt.tight_layout(); plt.show()

def plot_cm(ax, y_t, y_p, title):
    cm = confusion_matrix(y_t, y_p)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(title)
    thresh = cm.max() / 2
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold")
    return cm

fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
cm_C = plot_cm(axs[0], y_true, y_pred_C, "Model C - Regularised CNN")
cm_D = plot_cm(axs[1], y_true, y_pred_D, "Model D - ViT-B/16 Fine-tuned")
plt.tight_layout(); plt.show()

for nm, cm in [("Model C", cm_C), ("Model D", cm_D)]:
    tn, fp, fn, tp = cm.ravel()
    print(f"{nm}:  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"   MISSED FIRES (false negatives) = {fn}   <-- the costly error")
    print(f"   FALSE ALARMS (false positives) = {fp}\n")

# ROC and Precision-Recall curves.
# For imbalanced data the PR curve is the more informative of the two, because
# ROC's false-positive rate has a large TN denominator that flatters the model.
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))

for nm, prob, col in [("C: Regularised CNN", y_prob_C, "#55A868"),
                      ("D: ViT-B/16 Fine-tuned", y_prob_D, "#4C72B0")]:
    fpr, tpr, _ = roc_curve(y_true, prob)
    ax[0].plot(fpr, tpr, lw=2, color=col, label=f"{nm} (AUC = {auc(fpr, tpr):.4f})")

    prec, rec, _ = precision_recall_curve(y_true, prob)
    ap = average_precision_score(y_true, prob)
    ax[1].plot(rec, prec, lw=2, color=col, label=f"{nm} (AP = {ap:.4f})")

ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="random")
ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate (recall)")
ax[0].set_title("ROC curve"); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)

ax[1].axhline(y_true.mean(), ls="--", c="k", lw=1,
              label=f"baseline (prevalence = {y_true.mean():.3f})")
ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
ax[1].set_title("Precision-Recall curve (better suited to imbalanced data)")
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)

plt.tight_layout(); plt.show()


# ======================================================================
# 13. Threshold Tuning — Encoding the Cost of a Missed Fire
# ======================================================================

best_name, best_prob = ("D: ViT-B/16 Fine-tuned", y_prob_D) \
    if metrics_df.loc["D: ViT-B/16 Fine-tuned", "F1 (macro)"] >= metrics_df.loc["C: Regularised CNN", "F1 (macro)"] \
    else ("C: Regularised CNN", y_prob_C)
print(f"Sweeping thresholds for the better model by macro-F1: {best_name}\n")

rows = []
for t in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    pred = (best_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    rows.append({
        "Threshold":       t,
        "Accuracy":        round((tp + tn) / len(y_true), 4),
        "Precision":       round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "Recall":          round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "F1 (fire)":       round(f1_score(y_true, pred, pos_label=1, zero_division=0), 4),
        "F1 (macro)":      round(f1_score(y_true, pred, average="macro", zero_division=0), 4),
        "Missed fires FN": fn,
        "False alarms FP": fp,
    })

thresh_df = pd.DataFrame(rows)
print(thresh_df)

best_f1_row = thresh_df.loc[thresh_df["F1 (macro)"].idxmax()]
print(f"Best macro-F1 threshold : {best_f1_row['Threshold']:.2f} "
      f"(macro-F1 = {best_f1_row['F1 (macro)']:.4f})")
print("\nLower threshold -> fewer missed fires, more false alarms.")
print("For wildfire detection, err on the side of recall: a false alarm costs a")
print("crew a wasted trip; a missed fire can cost a forest.")

fig, ax = plt.subplots(1, 2, figsize=(14.5, 4.6))

ax[0].plot(thresh_df["Threshold"], thresh_df["Precision"], "o-", label="precision", color="#4C72B0")
ax[0].plot(thresh_df["Threshold"], thresh_df["Recall"],    "o-", label="recall",    color="#C44E52")
ax[0].plot(thresh_df["Threshold"], thresh_df["F1 (macro)"],"o-", label="macro-F1",  color="#55A868")
ax[0].axvline(0.5, ls="--", c="grey", lw=1, label="default 0.5")
ax[0].axvline(best_f1_row["Threshold"], ls=":", c="green", lw=1.8,
              label=f"best F1 @ {best_f1_row['Threshold']:.2f}")
ax[0].set_xlabel("decision threshold"); ax[0].set_ylabel("score")
ax[0].set_title(f"Precision / recall trade-off - {best_name}")
ax[0].legend(fontsize=8.5); ax[0].grid(alpha=.3)

w = 0.035
ax[1].bar(thresh_df["Threshold"] - w/2, thresh_df["Missed fires FN"], w,
          label="missed fires (FN)", color="#C44E52")
ax[1].bar(thresh_df["Threshold"] + w/2, thresh_df["False alarms FP"], w,
          label="false alarms (FP)", color="#DD8452")
ax[1].set_xlabel("decision threshold"); ax[1].set_ylabel("count on test split")
ax[1].set_title("The error you choose to make"); ax[1].legend(); ax[1].grid(axis="y", alpha=.3)

plt.tight_layout(); plt.show()


# ======================================================================
# 14. Grad-CAM — Is the Model Actually Looking at the Fire?
# ======================================================================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Grad-CAM for a Sequential model with a single sigmoid output.
    conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = keras.models.Model(model.inputs, [conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array, training=False)
        loss = preds[:, 0]                      # single sigmoid unit

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))          # importance per channel
    heat = conv_out[0] @ pooled[..., tf.newaxis]
    heat = tf.squeeze(heat)
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)   # ReLU + normalise
    return heat.numpy()


def overlay(img_uint8, heat, alpha=0.45):
    import matplotlib.cm as cm
    h = tf.image.resize(heat[..., np.newaxis], img_uint8.shape[:2]).numpy().squeeze()
    coloured = (cm.jet(h)[..., :3] * 255).astype("uint8")
    return (img_uint8 * (1 - alpha) + coloured * alpha).astype("uint8")

# Pick a few test images: 3 fire and 3 no_fire
fire_idx   = np.where(y_true == 1)[0][:3]
nofire_idx = np.where(y_true == 0)[0][:3]
pick = np.concatenate([fire_idx, nofire_idx])

fig, axes = plt.subplots(2, len(pick), figsize=(3.05 * len(pick), 6.4))

for col, idx in enumerate(pick):
    img_raw, _ = load_and_resize(paths_test[idx], y_test[idx])
    img_uint8 = tf.clip_by_value(img_raw, 0, 255).numpy().astype("uint8")
    batch = tf.expand_dims(img_raw, 0)

    prob = float(model_C.predict(batch, verbose=0).ravel()[0])
    pred = CLASS_NAMES[int(prob >= 0.5)]
    truth = CLASS_NAMES[y_true[idx]]
    ok = "OK" if pred == truth else "WRONG"

    axes[0, col].imshow(img_uint8); axes[0, col].axis("off")
    axes[0, col].set_title(f"true: {truth}\npred: {pred} ({prob:.3f}) {ok}",
                           fontsize=9,
                           color="#55A868" if pred == truth else "#C44E52")

    try:
        heat = make_gradcam_heatmap(batch, model_C, "conv4")
        axes[1, col].imshow(overlay(img_uint8, heat))
        axes[1, col].set_title("Grad-CAM", fontsize=9)
    except Exception as e:
        axes[1, col].text(.5, .5, f"Grad-CAM failed:\n{type(e).__name__}",
                          ha="center", va="center", fontsize=8)
    axes[1, col].axis("off")

plt.suptitle("Model C - top: input with prediction   |   bottom: Grad-CAM attention",
             y=1.02, fontsize=12)
plt.tight_layout(); plt.show()


# ======================================================================
# 15. Final Comparison — Bias, Variance, and the Diagnosis of Each Model
# ======================================================================

comparison = pd.DataFrame(RESULTS).T
comparison = comparison[["Train Acc (%)", "Test Acc (%)", "Bias (%)", "Variance (%)",
                         "Train Loss", "Test Loss", "Test Precision", "Test Recall",
                         "Test AUC", "Params", "Notes"]]
print(comparison)

print(f"Majority-class baseline to beat: {majority_baseline*100:.2f}%")
print(f"5-fold CV reference (Model C arch): {CV_MEAN:.2f}% +/- {CV_STD:.2f}")

names = list(RESULTS.keys())

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

train_accs = [RESULTS[n]["Train Acc (%)"] for n in names]
test_accs  = [RESULTS[n]["Test Acc (%)"]  for n in names]
biases     = [RESULTS[n]["Bias (%)"]      for n in names]
variances  = [RESULTS[n]["Variance (%)"]  for n in names]

xpos = np.arange(len(names)); w = 0.36
ax[0].bar(xpos - w/2, train_accs, w, label="Train Acc", color="#4C72B0")
ax[0].bar(xpos + w/2, test_accs,  w, label="Test Acc",  color="#C44E52")
ax[0].axhline(majority_baseline*100, ls="--", c="grey", label="majority baseline")
ax[0].set_xticks(xpos); ax[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
ax[0].set_ylabel("accuracy (%)"); ax[0].set_ylim(0, 108)
ax[0].set_title("Training vs Testing Accuracy"); ax[0].legend(fontsize=8.5)
ax[0].grid(axis="y", alpha=.3)

ax[1].bar(xpos - w/2, biases,    w, label="Bias (100 - train acc)", color="#DD8452")
ax[1].bar(xpos + w/2, variances, w, label="Variance (train - test)", color="#55A868")
ax[1].axhline(0, c="black", lw=.8)
ax[1].set_xticks(xpos); ax[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
ax[1].set_ylabel("error (%)")
ax[1].set_title("Bias vs Variance Decomposition"); ax[1].legend(fontsize=8.5)
ax[1].grid(axis="y", alpha=.3)

plt.tight_layout(); plt.show()

# All validation-loss curves on one axis -- the clearest single figure
# for the overfitting section of a report.
plt.figure(figsize=(11, 5.8))
colours = {
    "A: Underfit CNN":     "#DD8452",
    "B: Overfit CNN":      "#C44E52",
    "C: Regularised CNN":  "#55A868",
    "L1: lambda=1e-4":     "#8172B3",
    "L2: lambda=1e-4":     "#937860",
    "D: ViT-B/16 Fine-tuned": "#4C72B0",
}

for name, h in HISTORIES.items():
    if h and "val_loss" in h:
        plt.plot(range(1, len(h["val_loss"]) + 1), h["val_loss"],
                 "o-", ms=3.5, label=name, color=colours.get(name))

plt.xlabel("epoch"); plt.ylabel("validation loss (binary crossentropy)")
plt.title("Loss vs Epoch - validation loss for every model")
plt.legend(fontsize=9); plt.grid(alpha=.3); plt.tight_layout(); plt.show()

# Train vs validation loss, every model, side by side -- the underfit/overfit story
plot_names = [n for n in names if HISTORIES.get(n) and "loss" in HISTORIES[n]]
ncol = 3
nrow = int(np.ceil(len(plot_names) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.7 * nrow))
axes = np.atleast_1d(axes).ravel()

for a, n in zip(axes, plot_names):
    h = HISTORIES[n]
    ep = range(1, len(h["loss"]) + 1)
    a.plot(ep, h["loss"],     "o-", ms=3, label="train", color="#4C72B0")
    a.plot(ep, h["val_loss"], "o-", ms=3, label="validation", color="#C44E52")
    gap = h["val_loss"][-1] - h["loss"][-1]
    a.set_title(f"{n}\nfinal gap = {gap:+.4f}", fontsize=9.5)
    a.set_xlabel("epoch"); a.set_ylabel("loss"); a.legend(fontsize=8); a.grid(alpha=.3)

for a in axes[len(plot_names):]:
    a.axis("off")

plt.suptitle("Train vs Validation Loss - a widening gap IS overfitting", y=1.01, fontsize=13)
plt.tight_layout(); plt.show()

# Automatic diagnosis of each model, using standard thresholds.
def diagnose(train_acc, test_acc, bias_thresh=15.0, var_thresh=8.0):
    bias = 100 - train_acc
    var  = train_acc - test_acc
    if bias > bias_thresh and var < var_thresh:
        return "UNDERFITTING (high bias, low variance)"
    if bias <= bias_thresh and var > var_thresh:
        return "OVERFITTING (low bias, high variance)"
    if bias > bias_thresh and var > var_thresh:
        return "HIGH BIAS *and* HIGH VARIANCE (poor architecture)"
    return "GOOD FIT (low bias, low variance)"

print(f"{'Model':<24} {'Train':>8} {'Test':>8} {'Bias':>8} {'Var':>8}   Diagnosis")
print("-" * 104)
for n in names:
    r = RESULTS[n]
    print(f"{n:<24} {r['Train Acc (%)']:>7.2f}% {r['Test Acc (%)']:>7.2f}% "
          f"{r['Bias (%)']:>7.2f}% {r['Variance (%)']:>7.2f}%   "
          f"{diagnose(r['Train Acc (%)'], r['Test Acc (%)'])}")


# ======================================================================
# 16. Save the Best Model
# ======================================================================

best_key = max(RESULTS, key=lambda k: RESULTS[k]["Test Acc (%)"])
print("Best model by test accuracy:", best_key)

model_C.save("model_C_regularised_fire.keras")
model_D.save("model_D_vit_b16_fire.keras")
print("\nSaved:")
print("  model_C_regularised_fire.keras")
print("  model_D_vit_b16_fire.keras")

# Export the tables for your report.
comparison.to_csv("model_comparison.csv")
metrics_df.to_csv("classification_metrics.csv")
cv_df.to_csv("cross_validation_5fold.csv")
thresh_df.to_csv("threshold_sweep.csv")
print("\nExported: model_comparison.csv, classification_metrics.csv,")
print("          cross_validation_5fold.csv, threshold_sweep.csv")

# In Colab, download them:
# from google.colab import files
# for f in ["model_comparison.csv", "classification_metrics.csv",
#           "cross_validation_5fold.csv", "threshold_sweep.csv"]:
#     files.download(f)


# ======================================================================
# 17. Conclusion
# ======================================================================


# ======================================================================
# Level 2: Comparative Analysis with State-of-the-Art (CO4)
# ======================================================================

from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet

def make_sota_ds(paths, labels, preprocess_fn, shuffle=False, augment=False, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_and_resize, num_parallel_calls=AUTOTUNE).batch(batch_size)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

# Prepare datasets for ResNet50 and MobileNetV2
train_ds_res = make_sota_ds(paths_train, y_train, preprocess_resnet, shuffle=True, augment=True)
val_ds_res   = make_sota_ds(paths_val, y_val, preprocess_resnet)
test_ds_res  = make_sota_ds(paths_test, y_test, preprocess_resnet)

train_ds_mob = make_sota_ds(paths_train, y_train, preprocess_mobilenet, shuffle=True, augment=True)
val_ds_mob   = make_sota_ds(paths_val, y_val, preprocess_mobilenet)
test_ds_mob  = make_sota_ds(paths_test, y_test, preprocess_mobilenet)

# --- Train ResNet50 ---
print("Training ResNet50...")
base_res = ResNet50(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
base_res.trainable = False
model_res = keras.Sequential([
    layers.Input(shape=IMG_SIZE + (3,)),
    base_res,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid"),
], name="Model_ResNet50")
model_res = compile_model(model_res, lr=1e-3)
es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
hist_res = model_res.fit(train_ds_res, validation_data=val_ds_res, epochs=15, class_weight=class_weight, callbacks=[es], verbose=1)
evaluate_and_record(model_res, "SOTA: ResNet50", hist_res, notes="ResNet50 frozen feature extraction", eval_test=test_ds_res)

# --- Train MobileNetV2 ---
print("\nTraining MobileNetV2...")
base_mob = MobileNetV2(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
base_mob.trainable = False
model_mob = keras.Sequential([
    layers.Input(shape=IMG_SIZE + (3,)),
    base_mob,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid"),
], name="Model_MobileNetV2")
model_mob = compile_model(model_mob, lr=1e-3)
hist_mob = model_mob.fit(train_ds_mob, validation_data=val_ds_mob, epochs=15, class_weight=class_weight, callbacks=[es], verbose=1)
evaluate_and_record(model_mob, "SOTA: MobileNetV2", hist_mob, notes="MobileNetV2 frozen feature extraction", eval_test=test_ds_mob)

# Compile SOTA Comparison Table
y_prob_res = model_res.predict(test_ds_res, verbose=0).ravel()
y_pred_res = (y_prob_res >= 0.5).astype(int)
y_prob_mob = model_mob.predict(test_ds_mob, verbose=0).ravel()
y_pred_mob = (y_prob_mob >= 0.5).astype(int)

comp_matrix = pd.DataFrame([
    metric_row(y_true, y_pred_C, "Proposed: Model C"),
    metric_row(y_true, y_pred_D, "SOTA 1: ViT-B/16 (Transformer)"),
    metric_row(y_true, y_pred_res, "SOTA 2: ResNet50"),
    metric_row(y_true, y_pred_mob, "SOTA 3: MobileNetV2"),
])

print("=" * 80)
print("CO4: COMPARISON WITH STATE-OF-THE-ART (SOTA)")
print("=" * 80)
print(comp_matrix)
comp_matrix.to_csv("sota_comparison.csv", index=False)


# ======================================================================
# Level 3: Innovation & Ablation Study (CO5)
# ======================================================================

from sklearn.metrics import roc_auc_score

# --- 1. Define SE Block ---
def se_block(input_tensor, ratio=8):
    channels = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(channels // ratio, activation='relu', kernel_initializer='he_normal')(se)
    se = layers.Dense(channels, activation='sigmoid', kernel_initializer='he_normal')(se)
    se = layers.Reshape((1, 1, channels))(se)
    return layers.Multiply()([input_tensor, se])

# --- 2. Define Custom Focal Loss ---
def focal_loss(gamma=2., alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) - \
               tf.reduce_sum((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0))
    return focal_loss_fixed

# --- 3. Build Ablation Architecture ---
def build_ablation_model(use_se=False, name="Model"):
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    if use_se: x = se_block(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    if use_se: x = se_block(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    if use_se: x = se_block(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name=name)

def compile_ablation(model, use_focal=False, lr=1e-3):
    loss_fn = focal_loss() if use_focal else "binary_crossentropy"
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=loss_fn,
                  metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall"), keras.metrics.AUC(name="auc")])
    return model

# --- Run Ablation Study ---
ablation_results = []

experiments = [
    {"name": "Baseline", "se": False, "focal": False},
    {"name": "Exp 1 (SE Only)", "se": True, "focal": False},
    {"name": "Exp 2 (Focal Only)", "se": False, "focal": True},
    {"name": "Proposed (SE + Focal)", "se": True, "focal": True}
]

for exp in experiments:
    print(f"\nRunning {exp['name']}...")
    keras.backend.clear_session()
    tf.random.set_seed(SEED)

    # We clean the display name so Keras doesn't complain about spaces/parentheses
    safe_model_name = exp['name'].replace(' ', '_').replace('(' , '').replace(')', '').replace('+', 'plus')
    model = build_ablation_model(use_se=exp['se'], name=safe_model_name)
    model = compile_ablation(model, use_focal=exp['focal'])

    # If using focal loss, don't use class weights (focal loss handles it implicitly)
    cw = None if exp['focal'] else class_weight

    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)
    model.fit(train_ds, validation_data=val_ds, epochs=15, class_weight=cw, callbacks=[es], verbose=0)

    y_prob = model.predict(test_ds, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    ablation_results.append({
        "Experiment": exp["name"],
        "SE Block": "✓" if exp["se"] else "✗",
        "Focal Loss": "✓" if exp["focal"] else "✗",
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 (macro)": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "AUC": round(roc_auc_score(y_true, y_prob), 4)
    })

ablation_df = pd.DataFrame(ablation_results)
print("=" * 80)
print("CO5: ABLATION STUDY RESULTS")
print("=" * 80)
print(ablation_df)
ablation_df.to_csv("ablation_study.csv", index=False)
