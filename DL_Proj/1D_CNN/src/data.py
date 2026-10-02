"""C-MAPSS loading and preprocessing.

Pipeline (shared with the other team models so results are comparable):
  1. Load raw train/test/RUL files for a subset (FD001-FD004).
  2. Keep the 14 informative sensors (the other 7 are constant or near-constant).
  3. Normalise sensors (z-score); for multi-condition subsets (FD002/FD004) the
     scaling is done per operating condition, found by K-Means on the 3 settings.
  4. Piece-wise linear RUL target: RUL = min(max_cycle - cycle, RUL_CAP).
  5. Sliding windows of length `window` over each engine.
Scalers / clusters are fit on training engines only.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SETTINGS = ["op1", "op2", "op3"]
SENSORS_ALL = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + SETTINGS + SENSORS_ALL
# Sensors 1, 5, 6, 10, 16, 18, 19 carry no degradation information.
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11",
           "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
N_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
RUL_CAP = 125


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, sep=r"\s+", header=None, names=COLUMNS)


def load_raw(subset: str):
    train = _read(f"train_{subset}.txt")
    test = _read(f"test_{subset}.txt")
    rul = pd.read_csv(DATA_DIR / f"RUL_{subset}.txt", header=None, names=["rul"])
    rul["unit"] = np.arange(1, len(rul) + 1)
    return train, test, rul


def add_train_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (max_cycle - df["cycle"]).clip(upper=cap)
    return df


def add_test_rul(df: pd.DataFrame, rul: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """True RUL at every test cycle = RUL at last cycle + cycles remaining in the record."""
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    end_rul = df["unit"].map(rul.set_index("unit")["rul"])
    df["rul"] = (end_rul + max_cycle - df["cycle"]).clip(upper=cap)
    return df


class ConditionScaler:
    """Z-score scaling of sensors, done separately for each operating condition."""

    def __init__(self, n_conditions: int, seed: int = 0):
        self.n_conditions = n_conditions
        self.seed = seed

    def _conditions(self, df):
        if self.n_conditions == 1:
            return np.zeros(len(df), dtype=int)
        return self.kmeans.predict(df[SETTINGS].values)

    def fit(self, df):
        if self.n_conditions > 1:
            self.kmeans = KMeans(self.n_conditions, n_init=10, random_state=self.seed)
            self.kmeans.fit(df[SETTINGS].values)
        cond = self._conditions(df)
        x = df[SENSORS].values
        self.mean = np.stack([x[cond == c].mean(0) for c in range(self.n_conditions)])
        self.std = np.stack([x[cond == c].std(0) for c in range(self.n_conditions)]) + 1e-8
        return self

    def transform(self, df):
        df = df.copy()
        cond = self._conditions(df)
        df[SENSORS] = ((df[SENSORS].values - self.mean[cond]) / self.std[cond]).astype(np.float32)
        return df


def make_windows(df: pd.DataFrame, window: int):
    """All windows of every engine. Target = RUL at the window's last cycle."""
    xs, ys, units = [], [], []
    for unit, g in df.groupby("unit"):
        x = _pad(g[SENSORS].values, window)
        y = _pad(g["rul"].values[:, None], window)[:, 0]
        for end in range(window, len(x) + 1):
            xs.append(x[end - window:end])
            ys.append(y[end - 1])
            units.append(unit)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(units)


def last_windows(df: pd.DataFrame, window: int):
    """One window per engine: the last `window` cycles (official test protocol)."""
    xs, ys, units = [], [], []
    for unit, g in df.groupby("unit"):
        x = _pad(g[SENSORS].values, window)
        xs.append(x[-window:])
        ys.append(g["rul"].values[-1])
        units.append(unit)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(units)


def _pad(x: np.ndarray, window: int) -> np.ndarray:
    """Engines shorter than the window are front-padded with their first reading."""
    if len(x) >= window:
        return x
    return np.concatenate([np.repeat(x[:1], window - len(x), axis=0), x])


@dataclass
class Prepared:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_features: int = len(SENSORS)


def prepare(subset: str = "FD001", val_frac: float = 0.2, seed: int = 0) -> Prepared:
    """Scaled train / validation / test frames. Validation = held-out training engines."""
    train, test, rul = load_raw(subset)
    train = add_train_rul(train)
    test = add_test_rul(test, rul)

    units = train["unit"].unique()
    rng = np.random.default_rng(seed)
    val_units = rng.choice(units, size=int(len(units) * val_frac), replace=False)
    val = train[train["unit"].isin(val_units)]
    train = train[~train["unit"].isin(val_units)]

    scaler = ConditionScaler(N_CONDITIONS[subset], seed).fit(train)
    return Prepared(scaler.transform(train), scaler.transform(val), scaler.transform(test))
