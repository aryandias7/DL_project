"""LSTM training experiments (one factor at a time).

Phase 1: start from a baseline and change ONE setting per experiment, so the effect of each
setting can be seen on its own. Phase 2: combine the best value of every setting and train
that as a final experiment. All experiments are scored by validation RMSE (held-out
training engines); the test set is not used here.

    python experiments.py --workers 3
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import torch

from src.data import prepare
from src.engine import DEFAULTS, train
from src.model import count_params

BASELINE = dict(DEFAULTS)
VARIATIONS = {                    # setting -> values tried instead of the baseline value
    "window": [40, 50],
    "hidden_size": [32, 128],
    "n_layers": [2],
    "dropout": [0.4],
    "lr": [3e-3],
    "batch_size": [64],
}
OUT = Path("results") / "FD001"
_DATA = None


def _init(threads):
    global _DATA
    torch.set_num_threads(threads)
    _DATA = prepare("FD001")


def _run(name, changed, cfg, max_epochs, patience, seed):
    model, hist, info = train(cfg, _DATA, max_epochs=max_epochs, patience=patience, seed=seed)
    return {"experiment": name, "changed": changed, **cfg, **info,
            "params": count_params(model)}, hist


def run_all(jobs, workers, max_epochs, patience, seed):
    threads = max(1, torch.get_num_threads() // workers)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(threads,)) as pool:
        futs = [pool.submit(_run, n, c, cfg, max_epochs, patience, seed) for n, c, cfg in jobs]
        results = []
        for f in futs:
            row, hist = f.result()
            print(f"{row['experiment']:>4}  {row['changed']:<22} val RMSE {row['best_val_rmse']:6.2f}"
                  f"  ({row['epochs']} ep, {row['train_time_s']:.0f}s, {row['params']:,} params)",
                  flush=True)
            results.append((row, hist))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # Phase 1: baseline + one change at a time
    jobs = [("E0", "baseline", BASELINE)]
    for key, values in VARIATIONS.items():
        for v in values:
            jobs.append((f"E{len(jobs)}", f"{key} = {v}", {**BASELINE, key: v}))
    results = run_all(jobs, args.workers, args.max_epochs, args.patience, args.seed)

    # Phase 2: best value of each setting combined
    df = pd.DataFrame([r for r, _ in results])
    base_rmse = df.loc[df["experiment"] == "E0", "best_val_rmse"].item()
    combined = dict(BASELINE)
    for key in VARIATIONS:
        rows = df[df["changed"].str.startswith(f"{key} =")]
        best = rows.loc[rows["best_val_rmse"].idxmin()]
        if best["best_val_rmse"] < base_rmse:
            combined[key] = best[key].item() if hasattr(best[key], "item") else best[key]
    if combined != BASELINE:
        changed = ", ".join(f"{k}={combined[k]}" for k in VARIATIONS if combined[k] != BASELINE[k])
        results += run_all([(f"E{len(jobs)}", "combined: " + changed, combined)], 1,
                           args.max_epochs, args.patience, args.seed)

    df = pd.DataFrame([r for r, _ in results])
    df.to_csv(OUT / "experiments.csv", index=False)
    (OUT / "experiment_histories.json").write_text(
        json.dumps({r["experiment"]: h for r, h in results}))
    best = df.loc[df["best_val_rmse"].idxmin()]
    cfg = {k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in DEFAULTS}
    (OUT / "best_config.json").write_text(json.dumps(cfg, indent=2))
    print("\n" + df[["experiment", "changed", "best_val_rmse", "epochs", "params"]].to_string(index=False))
    print(f"\nBest: {best['experiment']} ({best['changed']}), val RMSE {best['best_val_rmse']:.2f}")


if __name__ == "__main__":
    main()
