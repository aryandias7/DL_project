# LSTM for Remaining Useful Life (RUL) Prediction: NASA C-MAPSS

Part of *Comparative Analysis of Deep Learning Architectures for Predictive Maintenance of
Industrial Machines* (MIT Manipal). This folder holds the **LSTM** model: implementation,
training experiments and preliminary evaluation.

## Layout

```
data/              C-MAPSS files (train_FD00x, test_FD00x, RUL_FD00x)
src/data.py        shared preprocessing (same as the 1D-CNN): 14 sensors, z-score, RUL cap 125, windows
src/metrics.py     RMSE, MAE, NASA PHM08 score
src/model.py       LSTM regressor
src/engine.py      training loop (Adam + MSE, gradient clipping, LR decay, early stopping)
src/plots.py       figures
experiments.py     one-factor-at-a-time training experiments (validation set only)
train_eval.py      train the chosen config with 3 seeds, evaluate on the test set, make plots
results/FD001/     experiment table, metrics, predictions, saved model, figures
```

## How to run

```bash
pip install -r requirements.txt
python experiments.py --workers 3
python train_eval.py
```

## Model

```
(B, 40, 14) -> LSTM(128 hidden, 1 layer) -> last hidden state -> Dropout(0.4) -> Dense(32) -> ReLU -> Dense(1)
```

- Optimiser: Adam with lr 3e-3, batch size 64.
- Gradient clipping at a norm of 1.0.
- The target is scaled to 0–1 (RUL / 125) during training.
- 77,889 parameters.

## Results: FD001 test set (100 engines, 3 seeds)

| RMSE | MAE | NASA score |
|---|---|---|
| 13.77 ± 0.82 | 9.77 ± 0.34 | 397 ± 142 |

All 10 experiments are listed in `results/FD001/experiments.csv`, and `experiments_bar.png` plots them.
