"""Train the chosen LSTM configuration with several seeds and evaluate it on the test set.

    python train_eval.py                      # FD001, config from experiments.py
    python train_eval.py --subsets FD001 FD003

Test metrics use the last window of each test engine against the RUL_FDxxx.txt labels
(capped at 125, as for training). Results are mean ± std over seeds; plots use the seed
with the best validation RMSE.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import plots
from src.data import RUL_CAP, last_windows, make_windows, prepare
from src.engine import DEFAULTS, build_model, predict, train
from src.metrics import all_metrics
from src.model import count_params

RESULTS = Path("results")


def _train_seed(subset, cfg, seed, max_epochs, patience, threads):
    torch.set_num_threads(threads)
    model, hist, info = train(cfg, prepare(subset), max_epochs=max_epochs,
                              patience=patience, seed=seed)
    return seed, model.state_dict(), hist, info


def evaluate_subset(subset, cfg, seeds, max_epochs, patience, workers):
    out = RESULTS / subset
    out.mkdir(parents=True, exist_ok=True)
    data = prepare(subset)
    x_te, y_te, units = last_windows(data.test, cfg["window"])

    n = min(workers, len(seeds))
    threads = max(1, torch.get_num_threads() // n)
    with ProcessPoolExecutor(n) as pool:
        runs = list(pool.map(_train_seed, *zip(*[(subset, cfg, s, max_epochs, patience, threads)
                                                 for s in seeds])))

    per_seed, models = [], {}
    for seed, state, hist, info in runs:
        model = build_model(cfg, data.n_features)
        model.load_state_dict(state)
        models[seed] = (model, hist, info)
        m = all_metrics(y_te, np.clip(predict(model, x_te), 0, None))
        per_seed.append({"subset": subset, "seed": seed, **m, **info})
        print(f"  {subset} seed {seed}: test RMSE {m['rmse']:.2f}  MAE {m['mae']:.2f}  "
              f"score {m['score']:.0f}  (val {info['best_val_rmse']:.2f}, {info['epochs']} ep)")
    per_seed = pd.DataFrame(per_seed)
    per_seed.to_csv(out / "seeds.csv", index=False)

    best_seed = int(per_seed.loc[per_seed["best_val_rmse"].idxmin(), "seed"])
    model, _, _ = models[best_seed]
    torch.save({"config": cfg, "state_dict": model.state_dict()}, out / "lstm_best.pt")
    y_hat = np.clip(predict(model, x_te), 0, None)
    pd.DataFrame({"unit": units, "true_rul": y_te, "pred_rul": y_hat.round(2),
                  "error": (y_hat - y_te).round(2)}).to_csv(out / "test_predictions.csv", index=False)

    tag = f"LSTM on {subset}"
    plots.learning_curves([h for _, h, _ in models.values()], out / "learning_curves.png",
                          f"{tag}: learning curves ({len(seeds)} seeds)")
    plots.pred_vs_true(y_te, y_hat, out / "pred_vs_true.png", f"{tag}: test set")
    plots.sorted_engines(y_te, y_hat, out / "test_engines_sorted.png", f"{tag}: per-engine predictions")
    plots.error_hist(y_te, y_hat, out / "error_hist.png", f"{tag}: error distribution")

    x_all, y_all, u_all = make_windows(data.test, cfg["window"])
    p_all = np.clip(predict(model, x_all), 0, None)
    longest = data.test.groupby("unit")["cycle"].max().nlargest(4).index
    curves = []
    for u in sorted(longest):
        mask = u_all == u
        cycles = data.test.loc[data.test["unit"] == u, "cycle"].values[-mask.sum():]
        curves.append((u, cycles, y_all[mask], p_all[mask]))
    plots.trajectories(curves, out / "engine_trajectories.png", f"{tag}: RUL over engine life")

    if (out / "experiments.csv").exists():
        df = pd.read_csv(out / "experiments.csv")
        hists = json.loads((out / "experiment_histories.json").read_text())
        plots.experiments_bar(df, out / "experiments_bar.png")
        plots.experiment_curves(hists, df, out / "experiment_curves.png")

    row = {"subset": subset, "params": count_params(model)}
    for m in ["rmse", "mae", "score"]:
        row[f"{m}_mean"] = per_seed[m].mean()
        row[f"{m}_std"] = per_seed[m].std(ddof=0)
    row["train_time_s"] = per_seed["train_time_s"].mean()
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subsets", nargs="+", default=["FD001"])
    p.add_argument("--config", default=str(RESULTS / "FD001" / "best_config.json"))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()

    cfg_path = Path(args.config)
    cfg = {**DEFAULTS, **(json.loads(cfg_path.read_text()) if cfg_path.exists() else {})}
    print("Config:", json.dumps(cfg))

    rows = []
    for subset in args.subsets:
        print(f"\n=== {subset} ===")
        rows.append(evaluate_subset(subset, cfg, args.seeds, args.max_epochs, args.patience,
                                    args.workers))

    summary = pd.DataFrame(rows)
    summary_path = RESULTS / "summary.csv"
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        summary = pd.concat([old[~old["subset"].isin(summary["subset"])], summary])
    summary = summary.sort_values("subset").reset_index(drop=True)
    summary.to_csv(summary_path, index=False)

    lines = ["| Subset | Test RMSE | Test MAE | NASA score | Params |", "|---|---|---|---|---|"]
    for r in summary.itertuples():
        lines.append(f"| {r.subset} | {r.rmse_mean:.2f} ± {r.rmse_std:.2f} | "
                     f"{r.mae_mean:.2f} ± {r.mae_std:.2f} | {r.score_mean:.0f} ± {r.score_std:.0f} | "
                     f"{r.params:,} |")
    table = "\n".join(lines)
    (RESULTS / "summary.md").write_text(
        f"# LSTM results (RUL cap {RUL_CAP}, seeds {args.seeds})\n\nConfig: `{json.dumps(cfg)}`\n\n"
        + table + "\n", encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    main()
