import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


# Evaluation methods for farm power prediction according to GB/T 40607-2021
def E_rmse(pred, true, C):
    return np.sqrt(np.mean(((true - pred) / C) ** 2))


def E_mae(pred, true, C):
    return np.mean(np.abs((true - pred) / C))


def E_me(pred, true, C):
    return np.mean(((true - pred) / C))


def r(pred, true, C):
    top = np.sum((true - true.mean()) * (pred - pred.mean()))
    bottom = np.sqrt(np.sum((true - true.mean()) ** 2) * np.sum((pred - pred.mean()) ** 2))
    return top / bottom


def C_R(pred, true, C):
    return (1 - E_rmse(pred, true, C)) * 100.


def Q_R(pred, true, C):
    B = np.where(np.abs(true - pred) / C < 0.25, 1., 0.)
    return B.mean() * 100.


# When the prediction accuracy is less than 85%, the power prediction accuracy is calculated according to the following formula:
def Acc_day_ahead(pred, true, Cap):
    return (1 - np.sqrt(np.sum((true - pred)**2 * (np.abs(true - pred) / np.sum(np.abs(true - pred))))) / Cap) * 100.


def Acc_hour_ahead_power(pred, true, Cap, P_N):
    return (85. - Acc_day_ahead(pred, true, Cap)) * P_N * 0.4