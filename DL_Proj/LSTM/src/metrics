import numpy as np


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def nasa_score(y_true, y_pred) -> float:
    """PHM08 scoring function (Saxena et al., 2008). Late predictions (d > 0) are
    penalised more heavily than early ones, since they risk an in-service failure."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def all_metrics(y_true, y_pred) -> dict:
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred),
            "score": nasa_score(y_true, y_pred)}
