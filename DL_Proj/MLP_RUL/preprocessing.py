import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data"

TRAIN_FILE = os.path.join(DATA_DIR, "train_FD001.txt")
TEST_FILE = os.path.join(DATA_DIR, "test_FD001.txt")
RUL_FILE = os.path.join(DATA_DIR, "RUL_FD001.txt")

MAX_RUL = 125


# ============================================================
# Column Names
# ============================================================

columns = (
    ["engine_id", "cycle"]
    + ["op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# ============================================================
# Load Dataset
# ============================================================

def load_dataset(path):

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )

    df = df.iloc[:, :26]
    df.columns = columns

    return df


train_df = load_dataset(TRAIN_FILE)
test_df = load_dataset(TEST_FILE)

print("Training Shape :", train_df.shape)
print("Testing Shape  :", test_df.shape)


# ============================================================
# Create RUL Labels
# ============================================================

max_cycle = train_df.groupby("engine_id")["cycle"].max()

train_df["RUL"] = train_df.apply(
    lambda row:
    max_cycle[row["engine_id"]] - row["cycle"],
    axis=1
)


# ============================================================
# Piecewise Linear RUL
# ============================================================

train_df["RUL"] = train_df["RUL"].clip(upper=MAX_RUL)


# ============================================================
# Remove Constant Sensors
# ============================================================

constant_columns = []

sensor_columns = [c for c in train_df.columns if "sensor" in c]

for col in sensor_columns:

    if train_df[col].std() < 1e-6:
        constant_columns.append(col)

train_df.drop(columns=constant_columns, inplace=True)
test_df.drop(columns=constant_columns, inplace=True)

print("\nRemoved Sensors")
print(constant_columns)


# ============================================================
# Feature Columns
# ============================================================

feature_columns = train_df.columns.drop(
    ["engine_id", "cycle", "RUL"]
)

print("\nNumber of Features :", len(feature_columns))


# ============================================================
# Normalization
# ============================================================

scaler = StandardScaler()

train_df[feature_columns] = scaler.fit_transform(
    train_df[feature_columns]
)

test_df[feature_columns] = scaler.transform(
    test_df[feature_columns]
)


# ============================================================
# Save Scaler
# ============================================================

os.makedirs("saved_model", exist_ok=True)

joblib.dump(
    scaler,
    "saved_model/scaler.pkl"
)


# ============================================================
# Save Processed Files
# ============================================================

train_df.to_csv(
    "data/train_processed.csv",
    index=False
)

test_df.to_csv(
    "data/test_processed.csv",
    index=False
)


print("\nProcessing Complete")

print("Processed Train Shape :", train_df.shape)
print("Processed Test Shape  :", test_df.shape)

print("\nSaved:")
print("train_processed.csv")
print("test_processed.csv")
print("scaler.pkl")
