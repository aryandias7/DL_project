# Transformer for RUL Prediction (NASA C-MAPSS FD001)

Standalone project: Transformer-encoder model for Remaining Useful Life regression.
Part of "Comparative Analysis of Deep Learning Architectures for Predictive Maintenance of Industrial Machines".

## Run (from this folder, in order)

    pip install -r requirements.txt
    python preprocessing.py     # clean, RUL labels, normalise
    python create_windows.py    # 30-cycle sliding windows (3-D arrays)
    python train.py             # train Transformer
    python evaluate.py          # metrics + plots

## Structure

    config.py            all settings (window size, hyper-parameters, paths)
    preprocessing.py     step 1
    create_windows.py    step 2
    model.py             Transformer architecture
    train.py             step 3
    evaluate.py          step 4
    data/                raw FD001 files (+ generated processed/ and windows/)
    saved_model/         scaler.pkl, best_transformer.keras, final_transformer.keras
    results/             metrics.json, predictions.csv, plots, history, model summary

## Settings used (config.py)

Window 30 cycles | RUL cap 125 | d_model 64 | 4 heads | 3 encoder layers | FF 128 |
dropout 0.1 | Adam lr 1e-3 | batch 64 | early stopping (patience 15)

## Notes

- Validation = 20 % of training ENGINES held out (no window leakage between train/val).
- Scaler is fitted on training data only.
- Test set = last 30-cycle window of each of the 100 test engines vs RUL_FD001.txt.
- Metrics: RMSE, MAE, R2, NASA asymmetric score, inference time, parameter count.
