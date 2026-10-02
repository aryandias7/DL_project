"""Training / inference loop shared by tuning and final evaluation."""
import copy
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import Prepared, make_windows
from .metrics import rmse
from .model import CNN1D

DEFAULTS = dict(window=30, n_filters=32, kernel_size=5, n_layers=3, dropout=0.2,
                fc_units=64, pooling="flatten", lr=1e-3, batch_size=128, weight_decay=0.0,
                noise_std=0.0)
MODEL_KEYS = ("n_filters", "kernel_size", "n_layers", "dropout", "fc_units", "pooling")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg: dict, n_features: int) -> CNN1D:
    return CNN1D(n_features, cfg["window"], **{k: cfg[k] for k in MODEL_KEYS})


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(x[i:i + batch_size])) for i in range(0, len(x), batch_size)]
    return torch.cat(out).numpy()


def train(cfg: dict, data: Prepared, max_epochs: int = 100, patience: int = 12,
          seed: int = 0, verbose: bool = False):
    """Train with Adam (+ optional L2 weight decay) and MSE, LR decay on plateau,
    early stopping on validation RMSE.

    Returns (best model, history dict, info dict).
    """
    cfg = {**DEFAULTS, **cfg}
    set_seed(seed)
    x_tr, y_tr, _ = make_windows(data.train, cfg["window"])
    x_va, y_va, _ = make_windows(data.val, cfg["window"])
    loader = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
                        batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    model = build_model(cfg, data.n_features)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    history = {"train_rmse": [], "val_rmse": [], "lr": []}
    best, best_state, bad, t0 = float("inf"), None, 0, time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            if cfg["noise_std"] > 0:  # Gaussian input-noise augmentation (regulariser)
                xb = xb + cfg["noise_std"] * torch.randn_like(xb)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        tr = (total / len(x_tr)) ** 0.5
        va = rmse(y_va, predict(model, x_va))
        sched.step(va)
        history["train_rmse"].append(tr)
        history["val_rmse"].append(va)
        history["lr"].append(opt.param_groups[0]["lr"])
        if verbose:
            print(f"  epoch {epoch:3d}  train {tr:6.2f}  val {va:6.2f}  lr {history['lr'][-1]:.1e}")
        if va < best - 1e-3:
            best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    info = {"best_val_rmse": best, "epochs": epoch, "best_epoch": int(np.argmin(history["val_rmse"])) + 1,
            "train_time_s": time.time() - t0}
    return model, history, info
