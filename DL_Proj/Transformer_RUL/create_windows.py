"""
create_windows.py

Step 2 of 4.  Processed CSVs  ->  3-D sliding-window arrays for the Transformer.

  X_train : (num_windows, WINDOW_SIZE, num_features)   one window per cycle
  y_train : (num_windows,)                             RUL at the window's last cycle
  X_test  : (num_engines, WINDOW_SIZE, num_features)   LAST window of each test engine
  y_test  : (num_engines,)                             true RUL from RUL_FD001.txt
  g_train : (num_windows,)                             engine id of each window
                                                       (used to split train/val BY ENGINE)
"""
import numpy as np
import pandas as pd

import config

W = config.WINDOW_SIZE


def main():
    train_df = pd.read_csv(config.PROCESSED_DIR / "train_processed.csv")
    test_df = pd.read_csv(config.PROCESSED_DIR / "test_processed.csv")
    true_rul = pd.read_csv(config.RUL_FILE, header=None).iloc[:, 0].values

    feature_cols = [c for c in train_df.columns if c not in ("engine_id", "cycle", "RUL")]

    # ---------------- Train ----------------
    X_train, y_train, g_train = [], [], []
    for engine, df in train_df.groupby("engine_id"):
        feats = df[feature_cols].values
        rul = df["RUL"].values
        for i in range(len(df) - W + 1):
            X_train.append(feats[i:i + W])
            y_train.append(rul[i + W - 1])
            g_train.append(engine)

    # ---------------- Test (one window per engine) ----------------
    X_test = []
    for engine, df in test_df.groupby("engine_id"):
        feats = df[feature_cols].values
        if len(feats) >= W:
            window = feats[-W:]
        else:  # short engine: pad the start by repeating the first cycle
            pad = np.repeat(feats[:1], W - len(feats), axis=0)
            window = np.vstack([pad, feats])
        X_test.append(window)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    g_train = np.asarray(g_train, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(true_rul, dtype=np.float32)

    assert len(X_test) == len(y_test), "engine count mismatch between test file and RUL file"

    np.save(config.WINDOW_DIR / "X_train.npy", X_train)
    np.save(config.WINDOW_DIR / "y_train.npy", y_train)
    np.save(config.WINDOW_DIR / "g_train.npy", g_train)
    np.save(config.WINDOW_DIR / "X_test.npy", X_test)
    np.save(config.WINDOW_DIR / "y_test.npy", y_test)

    print("Window size :", W)
    print("X_train     :", X_train.shape, "| y_train:", y_train.shape)
    print("X_test      :", X_test.shape, "| y_test :", y_test.shape)


if __name__ == "__main__":
    main()
