# 1D-CNN results (RUL cap 125, seeds [0, 1, 2])

Config: `{"window": 40, "n_filters": 16, "kernel_size": 9, "n_layers": 4, "dropout": 0.3, "fc_units": 64, "pooling": "gap", "lr": 0.001, "batch_size": 64, "weight_decay": 0.0, "noise_std": 0.05}`

| Subset | Test RMSE | Test MAE | NASA score | Params |
|---|---|---|---|---|
| FD001 | 16.11 ± 0.37 | 12.45 ± 0.19 | 451 ± 95 | 10,273 |
