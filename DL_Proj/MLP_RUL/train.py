"""
train.py

Train the Multi-Layer Perceptron (MLP)
for Remaining Useful Life Prediction
"""

import os
import json
import numpy as np
import time

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from model import build_model


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 64
EPOCHS = 100
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

WINDOW_DIR = "data/windows"

MODEL_DIR = "saved_model"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# Load Dataset
# ============================================================

print("Loading dataset...")

X = np.load(os.path.join(WINDOW_DIR, "X_train.npy"))
y = np.load(os.path.join(WINDOW_DIR, "y_train.npy"))

print("X Shape :", X.shape)
print("y Shape :", y.shape)


# ============================================================
# Train Validation Split
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

print("\nTraining Samples :", len(X_train))
print("Validation Samples :", len(X_val))


# ============================================================
# Build Model
# ============================================================

input_dimension = X_train.shape[1]

model = build_model(input_dimension)

model.summary()

# ============================================================
# Save Model Summary
# ============================================================

with open("results/model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# ============================================================
# Callbacks
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_mlp.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)


# ============================================================
# Train Model
# ============================================================

print("\nTraining Started...\n")

start = time.time()

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# Save Final Model
# ============================================================

model.save(
    os.path.join(MODEL_DIR, "final_mlp.keras")
)


# ============================================================
# Save History
# ============================================================

history_dict = {}

for key in history.history:

    history_dict[key] = [
        float(v)
        for v in history.history[key]
    ]

with open(
    os.path.join(
        RESULT_DIR,
        "history.json"
    ),
    "w"
) as f:

    json.dump(
        history_dict,
        f,
        indent=4
    )


end = time.time()

print(f"\nTraining Time : {end-start:.2f} seconds")

print("\nTraining Finished")

print("\nSaved Files")

print("best_mlp.keras")
print("final_mlp.keras")
print("history.json")
