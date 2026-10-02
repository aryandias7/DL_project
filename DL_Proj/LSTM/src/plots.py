"""Result visualisations for the LSTM experiments."""
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


def experiments_bar(df, path: Path):
    """Validation RMSE of every training experiment, baseline highlighted."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [f"{e}: {c}" for e, c in zip(df["experiment"], df["changed"])]
    colors = [ORANGE if c == "baseline" else ("crimson" if c.startswith("combined") else BLUE)
              for c in df["changed"]]
    y = np.arange(len(df))[::-1]
    ax.barh(y, df["best_val_rmse"], color=colors)
    for yi, v in zip(y, df["best_val_rmse"]):
        ax.text(v + 0.1, yi, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    base = df.loc[df["changed"] == "baseline", "best_val_rmse"].item()
    ax.axvline(base, color=GREY, ls="--", lw=1)
    ax.set(xlabel="best validation RMSE (cycles)",
           title="LSTM training experiments (orange = baseline, red = combined best settings)")
    ax.set_xlim(min(df["best_val_rmse"]) * 0.85, max(df["best_val_rmse"]) * 1.06)
    _save(fig, path)


def experiment_curves(histories: dict, df, path: Path):
    """Validation RMSE per epoch for each experiment."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, (e, c) in enumerate(zip(df["experiment"], df["changed"])):
        h = histories[e]["val_rmse"]
        ax.plot(np.arange(1, len(h) + 1), h, color=cmap(i % 10), lw=1.3, label=f"{e}: {c}")
    ax.set(xlabel="epoch", ylabel="validation RMSE (cycles)", ylim=(10, 50),
           title="Validation RMSE during training, per experiment")
    ax.legend(fontsize=7.5, ncol=2)
    _save(fig, path)


def subset_comparison(summary: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(summary))
    for ax, m in zip(axes, ["rmse", "mae"]):
        ax.bar(x, summary[f"{m}_mean"], yerr=summary[f"{m}_std"], color=BLUE, capsize=4)
        for xi, v in zip(x, summary[f"{m}_mean"]):
            ax.text(xi, v + 0.6, f"{v:.2f}", ha="center")
        ax.set_xticks(x, summary["subset"])
        ax.set(ylabel=f"test {m.upper()} (cycles)", title=f"LSTM test {m.upper()} by subset")
    _save(fig, path)
