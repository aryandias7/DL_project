"""Random-search hyperparameter tuning for the 1D-CNN.

Each trial trains on 80% of the training engines and is scored by RMSE on the
held-out 20% (validation engines). The test set is never touched here.

    python tune.py --subset FD001 --trials 30
"""
import argparse
import json
import random
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import torch

from src.data import prepare
from src.engine import DEFAULTS, train
from src.model import count_params

SEARCH_SPACE = {
    "window": [20, 30, 40, 50],
    "n_filters": [16, 32, 64, 128],
    "kernel_size": [3, 5, 7, 9],
    "n_layers": [2, 3, 4],
    "dropout": [0.1, 0.2, 0.3, 0.5],
    "fc_units": [32, 64, 128],
    "pooling": ["flatten", "gap"],
    "lr": [3e-4, 1e-3, 3e-3],
    "batch_size": [64, 128, 256],
}

# Stage 2: refine around the stage-1 winner and add regularisation, since stage 1 showed
# a large train/validation gap (the network memorises the 80 training engines).
SEARCH_SPACE_STAGE2 = {
    "window": [40, 50],
    "n_filters": [16, 32],
    "kernel_size": [5, 7, 9],
    "n_layers": [3, 4],
    "dropout": [0.3, 0.5],
    "fc_units": [32, 64, 128],
    "pooling": ["flatten", "gap"],
    "lr": [1e-3, 3e-3],
    "batch_size": [64, 128],
    "weight_decay": [0.0, 1e-4, 1e-3],
    "noise_std": [0.0, 0.05, 0.1],
}


_DATA = None


def _init_worker(subset: str, threads: int):
    global _DATA
    torch.set_num_threads(threads)
    _DATA = prepare(subset)


def _run_trial(i: int, cfg: dict, max_epochs: int, patience: int, seed: int):
    try:
        model, _, info = train(cfg, _DATA, max_epochs=max_epochs, patience=patience, seed=seed)
    except RuntimeError as e:  # e.g. out of memory: record it and keep the search going
        return {"trial": i, **cfg, "best_val_rmse": float("nan"), "error": str(e)[:200]}
    return {"trial": i, **cfg, **info, "params": count_params(model)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="FD001")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=3, help="trials trained in parallel")
    p.add_argument("--stage", type=int, default=1, choices=[1, 2])
    args = p.parse_args()

    out = Path("results") / args.subset
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == 1:
        csv_path, space = out / "tuning.csv", SEARCH_SPACE
        start = DEFAULTS.copy()  # trial 0 = hand-picked baseline
    else:
        csv_path, space = out / "tuning_stage2.csv", SEARCH_SPACE_STAGE2
        start = {**DEFAULTS, **json.loads((out / "best_config.json").read_text())}  # stage-1 best

    rng = random.Random(args.seed + args.stage - 1)
    configs = [start]
    configs += [{**DEFAULTS, **{k: rng.choice(v) for k, v in space.items()}}
                for _ in range(args.trials - 1)]

    # Resume: keep trials already finished in a previous (interrupted) run.
    rows = []
    if csv_path.exists():
        done = pd.read_csv(csv_path).dropna(subset=["best_val_rmse"])
        rows = done.to_dict("records")
    done_ids = {r["trial"] for r in rows}
    todo = [(i, cfg) for i, cfg in enumerate(configs) if i not in done_ids]
    if done_ids:
        print(f"Resuming: {len(done_ids)} trials already done, {len(todo)} to go")

    threads = max(1, torch.get_num_threads() // args.workers)
    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(args.subset, threads)) as pool:
        futures = [pool.submit(_run_trial, i, cfg, args.max_epochs, args.patience, args.seed)
                   for i, cfg in todo]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            pd.DataFrame(rows).sort_values("trial").to_csv(csv_path, index=False)
            if "error" in row:
                print(f"trial {row['trial']} failed: {row['error']}", flush=True)
                continue
            print(f"[{len(rows):2d}/{len(configs)}] trial {row['trial']:2d}  val RMSE "
                  f"{row['best_val_rmse']:6.2f} ({row['epochs']} ep, {row['train_time_s']:.0f}s)  "
                  + " ".join(f"{k}={row[k]}" for k in space), flush=True)

    df = pd.DataFrame(rows).dropna(subset=["best_val_rmse"]).sort_values("best_val_rmse")
    print("\nTop 5 configurations:")
    print(df.head(5)[["trial", *space, "best_val_rmse", "params"]].to_string(index=False))

    # best_config.json = best trial over all stages run so far
    stages = [pd.read_csv(p) for p in (out / "tuning.csv", out / "tuning_stage2.csv") if p.exists()]
    all_trials = pd.concat(stages, ignore_index=True).dropna(subset=["best_val_rmse"])
    top = all_trials.loc[all_trials["best_val_rmse"].idxmin()]
    best = {k: (DEFAULTS[k] if pd.isna(top.get(k, float("nan"))) else top[k]) for k in DEFAULTS}
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    best = {k: (int(v) if isinstance(DEFAULTS[k], int) else v) for k, v in best.items()}
    (out / "best_config.json").write_text(json.dumps(best, indent=2))
    print(f"\nBest overall val RMSE {top['best_val_rmse']:.2f}; config saved to {out / 'best_config.json'}")


if __name__ == "__main__":
    main()
