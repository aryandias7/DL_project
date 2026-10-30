"""
train.py

Step 3 of 4.  Train the Transformer.

Validation set = 20 % of the TRAINING ENGINES (whole engines held out), so no
window from a validation engine is ever seen during training.
"""
import json
import random
import time

import keras
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import config
from model import build_model

random.seed(config.SEED)
np.random.seed(config.SEED)
keras.utils.set_random_seed(config.SEED)

# ---------------- Data ----------------
X = np.load(config.WINDOW_DIR / "X_train.npy")
y = np.load(config.WINDOW_DIR / "y_train.npy")
groups = np.load(config.WINDOW_DIR / "g_train.npy")
print("X:", X.shape, "| y:", y.shape)

splitter = GroupShuffleSplit(n_splits=1, test_size=config.VALIDATION_SIZE, random_state=config.SEED)
tr_idx, va_idx = next(splitter.split(X, y, groups))
X_train, y_train = X[tr_idx], y[tr_idx]
X_val, y_val = X[va_idx], y[va_idx]
print(f"Train windows: {len(X_train)} ({len(np.unique(groups[tr_idx]))} engines)")
print(f"Val windows  : {len(X_val)} ({len(np.unique(groups[va_idx]))} engines)")

# ---------------- Model ----------------
model = build_model(window_size=X.shape[1], num_features=X.shape[2])
model.summary()
with open(config.RESULT_DIR / "model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda s: f.write(s + "\n"))

callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint(str(config.MODEL_DIR / "best_transformer.keras"),
                    monitor="val_loss", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
]

# ---------------- Train ----------------
start = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)
train_time = time.time() - start

model.save(config.MODEL_DIR / "final_transformer.keras")

# ---------------- Save logs ----------------
hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(config.RESULT_DIR / "history.json", "w") as f:
    json.dump(hist, f, indent=4)
with open(config.RESULT_DIR / "train_info.json", "w") as f:
    json.dump({
        "train_time_seconds": train_time,
        "epochs_run": len(hist["loss"]),
        "total_params": int(model.count_params()),
    }, f, indent=4)

print(f"\nTraining time: {train_time:.1f} s | epochs run: {len(hist['loss'])}")
