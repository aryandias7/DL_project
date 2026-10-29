import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

WINDOW_SIZE = 30

TRAIN_FILE = "data/train_processed.csv"
TEST_FILE = "data/test_processed.csv"
RUL_FILE = "data/RUL_FD001.txt"


# ============================================================
# Load Data
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

true_rul = pd.read_csv(
    RUL_FILE,
    header=None,
    names=["RUL"]
)


# ============================================================
# Feature Columns
# ============================================================

feature_columns = [
    col for col in train_df.columns
    if col not in ["engine_id", "cycle", "RUL"]
]


# ============================================================
# Create Training Windows
# ============================================================

X_train = []
y_train = []

engine_ids = train_df["engine_id"].unique()

for engine in engine_ids:

    engine_data = train_df[
        train_df["engine_id"] == engine
    ].reset_index(drop=True)

    features = engine_data[feature_columns].values
    rul = engine_data["RUL"].values

    for i in range(len(engine_data) - WINDOW_SIZE + 1):

        window = features[i:i + WINDOW_SIZE]

        X_train.append(window.flatten())

        y_train.append(
            rul[i + WINDOW_SIZE - 1]
        )


X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)


# ============================================================
# Create Test Windows
# ============================================================

X_test = []
y_test = []

engine_ids = test_df["engine_id"].unique()

for idx, engine in enumerate(engine_ids):

    engine_data = test_df[
        test_df["engine_id"] == engine
    ].reset_index(drop=True)

    features = engine_data[feature_columns].values

    if len(engine_data) >= WINDOW_SIZE:

        window = features[-WINDOW_SIZE:]

    else:

        pad = np.repeat(
            features[0:1],
            WINDOW_SIZE - len(engine_data),
            axis=0
        )

        window = np.vstack((pad, features))

    X_test.append(window.flatten())

    y_test.append(true_rul.iloc[idx, 0])


X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)


# ============================================================
# Create Output Folder
# ============================================================

os.makedirs("data/windows", exist_ok=True)


# ============================================================
# Save Arrays
# ============================================================

np.save("data/windows/X_train.npy", X_train)
np.save("data/windows/y_train.npy", y_train)

np.save("data/windows/X_test.npy", X_test)
np.save("data/windows/y_test.npy", y_test)


# ============================================================
# Print Summary
# ============================================================

print("\nSliding Window Creation Complete\n")

print("Window Size :", WINDOW_SIZE)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

print("\nInput Shape :", X_train.shape)
print("Output Shape:", y_train.shape)

print("\nFiles Saved")

print("X_train.npy")
print("y_train.npy")
print("X_test.npy")
print("y_test.npy")
