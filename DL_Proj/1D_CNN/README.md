# 1D-CNN for Remaining Useful Life (RUL) Prediction — NASA C-MAPSS

Part of *Comparative Analysis of Deep Learning Architectures for Predictive Maintenance of
Industrial Machines* (MIT Manipal). This folder holds the **1D-CNN** model (Siddhant Bandal):
implementation, hyperparameter tuning, evaluation and result visualisation.

## Layout

```
data/                 C-MAPSS files (train_FD00x, test_FD00x, RUL_FD00x)
src/data.py           loading, sensor selection, normalisation, RUL labels, sliding windows
src/model.py          configurable 1D-CNN
src/engine.py         training loop (Adam + MSE, LR decay, early stopping)
src/metrics.py        RMSE, MAE, NASA PHM08 score
src/plots.py          all figures
tune.py               random-search hyperparameter tuning (validation set only)
train_eval.py         train the best config with 3 seeds, evaluate on the test set, make plots
results/              tuning logs, metrics, predictions, saved models, figures
```

## How to run

```bash
pip install -r requirements.txt
python tune.py --subset FD001 --trials 30 --workers 4
python train_eval.py --subsets FD001 FD002 FD003 FD004
```

`train_eval.py` reads `results/FD001/best_config.json`, the configuration tuned on FD001.

## Preprocessing (shared protocol)

| Step | Choice |
|---|---|
| Features | 14 sensors: s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21. Sensors 1, 5, 6, 10, 16, 18 and 19 are constant, so they are dropped. |
| Normalisation | z-score, fit on training engines only. FD002/FD004 are normalised separately for each of their 6 operating conditions, found by K-Means on the 3 operating settings. |
| Target | piece-wise linear RUL, `min(cycles_left, 125)` |
| Input | sliding windows of `window` cycles, shape `(window, 14)`. Short engines are front-padded. |
| Validation | 20% of the training engines are held out, chosen by engine to avoid leakage. Used for early stopping and tuning. |
| Test | the last window of each test engine, compared with `RUL_FD00x.txt` (capped at 125) |

## Model

Each sensor is an input channel. The convolutions slide along the time axis:

```
(B, window, 14) -> transpose -> [Conv1d -> BatchNorm -> ReLU] x n_layers
                -> flatten (or global avg pool) -> Dropout -> Dense(fc_units) -> ReLU -> Dropout -> Dense(1)
```

## Hyperparameter search space

Random search, scored by validation RMSE (stage-1 space):

| Hyperparameter | Values |
|---|---|
| window | 20, 30, 40, 50 |
| n_filters | 16, 32, 64, 128 |
| kernel_size | 3, 5, 7, 9 |
| n_layers | 2, 3, 4 |
| dropout | 0.1, 0.2, 0.3, 0.5 |
| fc_units | 32, 64, 128 |
| pooling | flatten, global average |
| learning rate | 3e-4, 1e-3, 3e-3 |
| batch size | 64, 128, 256 |

## Metrics

- **RMSE** and **MAE**, in cycles.
- **NASA score**: an asymmetric penalty from the PHM08 challenge. A late prediction (predicted RUL larger than the true RUL) costs more than an early one. Lower is better.

The search ran in two stages:

- **Stage 1:** 30 random trials over the space above.
- **Stage 2:** 20 trials refining around the stage-1 winner, adding two regularisers: L2 weight decay {0, 1e-4, 1e-3} and Gaussian input noise {0, 0.05, 0.1}.

Stage 1 showed a large overfitting gap: train RMSE about 7.7 against validation about 18, because the network memorised the 80 training engines. That gap is why stage 2 adds regularisation.

## Results: FD001

**Final configuration** (`results/FD001/best_config.json`):
window 40, 4 conv layers × 16 filters, kernel 9, global average pooling, dense 64, dropout 0.3, Adam lr 1e-3, batch 64, input noise σ = 0.05. **10,273 parameters.**

| Metric (test set, 100 engines) | 1D-CNN (mean ± std, 3 seeds) |
|---|---|
| RMSE | **16.11 ± 0.37** |
| MAE | **12.45 ± 0.19** |
| NASA score | **451 ± 95** |

- For engines close to failure (true RUL ≤ 50), RMSE drops to **12.2**, against 18.3 for RUL > 50. The model is most accurate where maintenance decisions are made.
- The mean error is +3.1 cycles, and 63% of predictions are late (too high). This explains the high NASA score, since late predictions are penalised more.

### Findings from tuning

1. **Longer windows help.** Median validation RMSE was 18.5 for window 20, against 16.8 for window 50. More history gives the convolutions more degradation trend to work with.
2. **Wider kernels help.** Median validation RMSE was 20.9 for kernel 3, against 16.9 for kernel 9. Degradation is a slow trend, so wider receptive fields capture it better than very local patterns.
3. **Deeper is better, and wider is not.** 4 conv layers beat 2. The number of filters (16–128) made almost no difference, so the smallest model works.
4. **Smaller batches (64) generalise better** than 256.
5. **Regularisation:** input noise of 0.05 and weight decay of 1e-3 both lowered the median validation RMSE. With global average pooling and input noise, the final model has no train/validation gap (see `learning_curves.png`). Across trials, though, global average pooling alone was not better than flatten. The final model is about 10× smaller than the stage-1 winner (10k parameters against 110k), with the same validation RMSE (14.63 against 14.75).
6. **Early stopping:** validation RMSE plateaus around epochs 8–15. Rerunning with patience 30 (max 150 epochs) gave identical test results, so the model was not under-trained.

### Output files (`results/FD001/`)

| File | What it shows |
|---|---|
| `learning_curves.png` | train/validation RMSE per epoch, 3 seeds |
| `pred_vs_true.png` | predicted vs true RUL for the 100 test engines |
| `test_engines_sorted.png` | per-engine predictions, sorted by true RUL |
| `error_hist.png` | error distribution (early vs late predictions) |
| `engine_trajectories.png` | predicted RUL over the whole record of 4 test engines |
| `tuning_sensitivity.png`, `tuning_stage2_sensitivity.png` | validation RMSE vs each hyperparameter |
| `tuning_ranking.png`, `tuning_stage2_ranking.png` | all trials ranked |
| `tuning.csv`, `tuning_stage2.csv` | every trial's config and score |
| `seeds.csv`, `test_predictions.csv` | per-seed metrics and per-engine predictions |
| `cnn1d_best.pt` | trained weights and config |

### Notes for the team comparison

- The test RMSE is computed on the **last window** of each test engine, with true RUL capped at 125. Use the same protocol for the MLP, LSTM and Transformer.
- `src/data.py` can be imported directly, so all four models share an identical pipeline: same sensors, normalisation, RUL cap, validation engines (seed 0) and windows.
- Validation RMSE is noisy, because it comes from 20 engines. For example, the seed with the best validation score had the worst test score. Report mean ± std over seeds rather than a single run.
