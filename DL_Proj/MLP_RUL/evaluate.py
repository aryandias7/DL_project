"""
evaluate.py

Evaluate the trained MLP model
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from tensorflow.keras.models import load_model


# ============================================================
# Directories
# ============================================================

WINDOW_DIR = "data/windows"
MODEL_DIR = "saved_model"
RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# Load Model
# ============================================================

print("Loading trained model...")

model = load_model(
    os.path.join(MODEL_DIR, "best_mlp.keras")
)


# ============================================================
# Load Test Data
# ============================================================

X_test = np.load(
    os.path.join(WINDOW_DIR, "X_test.npy")
)

y_test = np.load(
    os.path.join(WINDOW_DIR, "y_test.npy")
)

print("Test Samples :", len(X_test))


# ============================================================
# Prediction
# ============================================================

pred = model.predict(X_test)

pred = pred.flatten()

pd.DataFrame({
    "Actual": y_test,
    "Predicted": pred
}).to_csv(
    os.path.join(
        RESULT_DIR,
        "predictions.csv"
    ),
    index=False
)

# ============================================================
# Metrics
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred
    )
)

mae = mean_absolute_error(
    y_test,
    pred
)

r2 = r2_score(
    y_test,
    pred
)

print("\nEvaluation Results")

print("---------------------------")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# Save Metrics
# ============================================================

metrics = {
    "RMSE": float(rmse),
    "MAE": float(mae),
    "R2": float(r2)
}

with open(
    os.path.join(
        RESULT_DIR,
        "metrics.json"
    ),
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# Plot 1
# Actual vs Predicted
# ============================================================

plt.figure(figsize=(10,6))

plt.plot(
    y_test,
    label="Actual",
    linewidth=2
)

plt.plot(
    pred,
    label="Predicted",
    linewidth=2
)

plt.legend()

plt.xlabel("Engine")

plt.ylabel("Remaining Useful Life")

plt.title("Actual vs Predicted RUL")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "actual_vs_predicted.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# Scatter Plot
# ============================================================

plt.figure(figsize=(6,6))

plt.scatter(
    y_test,
    pred,
    alpha=0.7
)

plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    'r--'
)

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs Predicted Scatter")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "scatter.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# Plot 2
# Prediction Error
# ============================================================

errors = pred - y_test

plt.figure(figsize=(8,6))

plt.hist(
    errors,
    bins=25
)

plt.xlabel("Prediction Error")

plt.ylabel("Frequency")

plt.title("Prediction Error Distribution")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "error_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# Plot 3
# Training History
# ============================================================

history_file = os.path.join(
    RESULT_DIR,
    "history.json"
)

if os.path.exists(history_file):

    with open(history_file) as f:

        history = json.load(f)

    plt.figure(figsize=(10,6))

    plt.plot(
        history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training vs Validation Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "loss_curve.png"
        ),
        dpi=300
    )

    plt.close()


print("\nPlots Saved")

print("actual_vs_predicted.png")
print("error_distribution.png")
print("loss_curve.png")

print("\nMetrics Saved")

print("metrics.json")