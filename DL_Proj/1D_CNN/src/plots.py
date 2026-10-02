"""Result visualisations for the 1D-CNN experiments."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY = "#2a6fdb", "#e8743b", "#888888"


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def learning_curves(histories: list, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, h in enumerate(histories):
        ep = np.arange(1, len(h["train_rmse"]) + 1)
        ax.plot(ep, h["train_rmse"], color=BLUE, alpha=0.8, label="train" if i == 0 else None)
        ax.plot(ep, h["val_rmse"], color=ORANGE, alpha=0.8, label="validation" if i == 0 else None)
    ax.set(xlabel="epoch", ylabel="RMSE (cycles)", title=title, ylim=(0, 60))
    ax.legend()
    _save(fig, path)


def pred_vs_true(y_true, y_pred, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=18, color=BLUE, alpha=0.7, edgecolor="none")
    lim = [0, max(y_true.max(), y_pred.max()) + 5]
    ax.plot(lim, lim, "--", color=GREY, lw=1, label="perfect prediction")
    ax.set(xlim=lim, ylim=lim, xlabel="true RUL (cycles)", ylabel="predicted RUL (cycles)",
           title=title)
    ax.legend(loc="upper left")
    _save(fig, path)


def sorted_engines(y_true, y_pred, path: Path, title: str):
    order = np.argsort(y_true)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y_true[order], color=GREY, lw=2, label="true RUL")
    ax.scatter(x, y_pred[order], s=14, color=BLUE, label="predicted RUL", zorder=3)
    ax.vlines(x, y_true[order], y_pred[order], color=BLUE, alpha=0.25, lw=1)
    ax.set(xlabel="test engine (sorted by true RUL)", ylabel="RUL (cycles)", title=title)
    ax.legend()
    _save(fig, path)


def error_hist(y_true, y_pred, path: Path, title: str):
    err = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(err, bins=25, color=BLUE, alpha=0.8, edgecolor="white")
    ax.axvline(0, color=GREY, ls="--", lw=1)
    ax.axvline(err.mean(), color=ORANGE, lw=1.5, label=f"mean error = {err.mean():.1f}")
    ax.set(xlabel="prediction error = predicted − true (cycles)\n"
                  "< 0: early (safe)    > 0: late (risky)",
           ylabel="number of engines", title=title)
    ax.legend()
    _save(fig, path)


def trajectories(curves: list, path: Path, title: str):
    """curves: list of (unit, cycles, true_rul, pred_rul) for whole test-engine records."""
    n = len(curves)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.2 * rows), squeeze=False)
    for ax, (unit, cyc, yt, yp) in zip(axes.flat, curves):
        ax.plot(cyc, yt, color=GREY, lw=2, label="true RUL (capped)")
        ax.plot(cyc, yp, color=BLUE, lw=1.3, label="predicted RUL")
        ax.set(title=f"engine {unit}", xlabel="cycle", ylabel="RUL")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    axes.flat[0].legend()
    fig.suptitle(title)
    _save(fig, path)


def tuning_sensitivity(tuning_csv: Path, params: list, path: Path):
    """Validation RMSE of every random-search trial, grouped by each hyperparameter value."""
    df = pd.read_csv(tuning_csv)
    best = df.loc[df["best_val_rmse"].idxmin()]
    cols = 3
    rows = int(np.ceil(len(params) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, p in zip(axes.flat, params):
        values = sorted(df[p].unique(), key=lambda v: (isinstance(v, str), v))
        for j, v in enumerate(values):
            y = df.loc[df[p] == v, "best_val_rmse"].values
            ax.scatter(j + rng.uniform(-0.12, 0.12, len(y)), y, s=20, color=BLUE, alpha=0.6)
            ax.hlines(np.median(y), j - 0.25, j + 0.25, color=ORANGE, lw=2)
        jb = values.index(best[p])
        ax.scatter(jb, best["best_val_rmse"], s=90, marker="*", color="crimson", zorder=4)
        ax.set_xticks(range(len(values)), [str(v) for v in values])
        ax.set(title=p, ylabel="val RMSE")
    for ax in axes.flat[len(params):]:
        ax.set_visible(False)
    fig.suptitle("Hyperparameter sensitivity (dots = trials, orange = median, red star = best)")
    _save(fig, path)


def tuning_ranking(tuning_csv: Path, path: Path, trial0: str = "default baseline config"):
    df = pd.read_csv(tuning_csv).sort_values("best_val_rmse").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [ORANGE if t == 0 else BLUE for t in df["trial"]]
    ax.bar(range(len(df)), df["best_val_rmse"], color=colors)
    ax.set(xlabel="trial rank", ylabel="best validation RMSE",
           title=f"Random search results (orange = trial 0: {trial0})")
    ax.set_ylim(df["best_val_rmse"].min() * 0.9, df["best_val_rmse"].max() * 1.02)
    _save(fig, path)


def subset_comparison(summary: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(summary))
    for ax, m in zip(axes, ["rmse", "mae"]):
        ax.bar(x, summary[f"{m}_mean"], yerr=summary[f"{m}_std"], color=BLUE, capsize=4)
        for xi, v in zip(x, summary[f"{m}_mean"]):
            ax.text(xi, v + 0.6, f"{v:.2f}", ha="center")
        ax.set_xticks(x, summary["subset"])
        ax.set(ylabel=f"test {m.upper()} (cycles)", title=f"1D-CNN test {m.upper()} by subset")
    _save(fig, path)
