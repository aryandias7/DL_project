"""
preprocessing.py

Step 1 of 4.  Raw C-MAPSS FD001  ->  cleaned, normalised CSVs.

  * load train / test text files
  * piece-wise linear RUL label for training data  (min(cycles_to_failure, MAX_RUL))
  * drop constant (zero-information) columns
  * StandardScaler fitted on TRAIN only, applied to train and test
"""
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_dataset(path):
    df = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, :26]
    df.columns = COLUMNS
    return df


def main():
    train_df = load_dataset(config.TRAIN_FILE)
    test_df = load_dataset(config.TEST_FILE)
    print("Train shape:", train_df.shape, "| Test shape:", test_df.shape)

    # ---- RUL labels (piece-wise linear) ----
    max_cycle = train_df.groupby("engine_id")["cycle"].transform("max")
    train_df["RUL"] = (max_cycle - train_df["cycle"]).clip(upper=config.MAX_RUL)

    # ---- Remove constant sensors (decided on train data only) ----
    sensor_cols = [c for c in train_df.columns if c.startswith("sensor")]
    constant = [c for c in sensor_cols if train_df[c].std() < 1e-6]
    train_df = train_df.drop(columns=constant)
    test_df = test_df.drop(columns=constant)
    print("Removed constant sensors:", constant)

    feature_cols = [c for c in train_df.columns if c not in ("engine_id", "cycle", "RUL")]
    print("Number of features:", len(feature_cols))

    # ---- Normalise (fit on train only -> no leakage) ----
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # ---- Save ----
    joblib.dump(scaler, config.MODEL_DIR / "scaler.pkl")
    train_df.to_csv(config.PROCESSED_DIR / "train_processed.csv", index=False)
    test_df.to_csv(config.PROCESSED_DIR / "test_processed.csv", index=False)
    print("Saved processed CSVs and scaler.pkl")


if __name__ == "__main__":
    main()
