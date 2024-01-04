import os, sys
import json, time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler


default_path = '../results/'

subplot_layouts = {4: (2, 2), 6: (2, 3), 9: (3, 3), 12: (3, 4), 16: (4, 4),
                   20: (4, 5), 24: (4, 6), 25: (5, 5), 36: (6, 6), 49: (7, 7),}


def test_results_plot(folder, step=None):
    result_path = default_path + folder
    pred_data = np.load(result_path + '/pred.npy'); true_data = np.load(result_path + '/true.npy')
    print(pred_data.shape, true_data.shape)

    scaler = StandardScaler()
    df_raw = pd.read_csv('../dataset/WFP/Turbine_Spd_Patv_filled.csv')
    df_data = df_raw[['Patv_Total']]
    num_train = int(len(df_raw) * 0.7); num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test
    border1s = [0, num_train - 96, len(df_raw) - num_test - 96]
    border2s = [num_train, num_train + num_vali, len(df_raw)]
    train_data = df_data[border1s[0]:border2s[0]]
    scaler.fit(train_data.values)
    pred = scaler.inverse_transform(pred_data[:, -1, :])
    true = scaler.inverse_transform(true_data[:, -1, :])
    C_wt = np.ones_like(pred) * 39.95
    e = E_RMSE(pred, true, C_wt); c = C_R(pred, true, C_wt); q = Q_R(pred, true, C_wt)
    print(f'E_RMSE: {e:.3f}, C_R: {c:.3f}, Q_R: {q:.3f}')

    if step is None:
        layouts = subplot_layouts[pred_data.shape[1]]; m, n = layouts[0], layouts[1]
        fig, ax = plt.subplots(m, n, figsize=(n * 6, m * 4), dpi=110)
        for i, axi in enumerate(ax.flatten()):
            axi.plot(pred_data[:, i, 0], '-r', label='pred'); axi.plot(true_data[:, i, 0], '-k', label='true')
            axi.set_title('Pred Step: {}'.format(i + 1)); axi.legend(loc='best')
    else:
        plot_num = 2000 or -1
        fig, ax = plt.subplots(figsize=(25, 8), dpi=80)
        ax.plot(pred[:plot_num], '-r', label='pred')
        ax.plot(true[:plot_num], '-k', label='true')
        ax.set_title('Pred Step: {}'.format(step)); ax.legend(loc='best')

    plt.show()


def E_RMSE(P_mean, P_pred, C_wt):
    assert P_mean.shape == P_pred.shape == C_wt.shape
    return np.sqrt(np.sum(((P_mean - P_pred) / C_wt) ** 2) / P_mean.shape[0])


def C_R(P_mean, P_pred, C_wt):
    return (1 - E_RMSE(P_mean, P_pred, C_wt)) * 100.


def Q_R(P_mean, P_pred, C_wt):
    B = np.where(np.abs(P_mean - P_pred) / C_wt < 0.25, 1., 0.)
    return B.sum() / B.shape[0] * 100.




if __name__ == '__main__':
    m_1 = 'long_term_forecast_WFP_96_16_Test_TimesNet_custom_ftS_sl96_ll96_pl16_dm256_nh8_el3_dl2_df32_fc3_ebtimeF_dtTrue_Exp_0'

    test_results_plot(m_1, step=16)