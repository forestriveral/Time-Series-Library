import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator


from metrics import E_rmse, E_mae, E_me, r, C_R, Q_R, Acc_day_ahead, Acc_hour_ahead_power


default_path = 'results/'

subplot_layouts = {2: (2, 1), 3:(3, 1), 4: (2, 2), 6: (2, 3),
                   9: (3, 3), 12: (3, 4), 16: (4, 4), 20: (4, 5),
                   24: (4, 6), 25: (5, 5), 36: (6, 6), 49: (7, 7),}

level_points = np.array([0, 0.2, 0.5, 0.7, 1., np.inf])


def pred_true_load(case):
    preds= np.load(os.path.join(default_path, f'{case}/pred.npy'))
    trues = np.load(os.path.join(default_path, f'{case}/true.npy'))

    if os.path.exists(os.path.join(default_path, f'{case}/time.npy')):
        times = np.load(os.path.join(default_path, f'{case}/time.npy'))
    else:
        times = None

    return preds, trues, times


def slide_pred_plot(case, save=None):
    preds, trues, times = pred_true_load(case)
    assert preds.shape == trues.shape
    if times is not None:
        assert preds.shape[0] == times.shape[0]
    slide_step = preds.shape[1]
    pred = preds[:, :, 0][::slide_step].flatten()
    true = trues[:, :, 0][::slide_step].flatten()
    # time = times[::slide_step].flatten()

    farm_capacity = 39.95
    slide_num = 30
    slide_num_subplot = 7
    if pred.shape[0] <= slide_num * 1.2:
        slide_num = pred.shape[0]
    subplot_num = int(np.floor(slide_num / slide_num_subplot))
    point_num = int(slide_num * slide_step)
    print('Number of Predictions: ', slide_num)
    print('Number of Datasamples: ', point_num)
    print('Number of Subplots: ', subplot_num)
    print('Number of Predictions in Subplots: ', slide_num_subplot)

    pred = pred[:point_num]
    true = true[:point_num]
    if times is not None:
        time = time[:point_num]
    else:
        time = np.arange(point_num)
    time_split = np.concatenate((
        np.arange(0, point_num, slide_num_subplot * slide_step)[:-1],
        [point_num]))

    fig, ax = plt.subplots(subplot_num, 1, figsize=(subplot_num * 4, 12), sharey=True, dpi=100)
    for ii, axi in enumerate(ax.flatten()):
        start, end = time_split[ii], time_split[ii + 1]
        # print('Time: ', start, end)
        axi.plot(time[start:end], pred[start:end], label='pred', lw=1.5)
        axi.plot(time[start:end], true[start:end], label='true', lw=1.5)
        for split_line in np.arange(start, end, slide_step):
            axi.axvline(split_line, color='r', linestyle='--', lw=1.2)
        axi.axhline(farm_capacity, color='k', linestyle='--', lw=0.8)
        axi.set_ylim([0., 1.2 * farm_capacity])
        axi.tick_params(labelsize=15, colors='k', direction='in',
                        top=True, bottom=True, left=True, right=True)
        tick_labs = axi.get_xticklabels() + axi.get_yticklabels()
        [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
        axi.legend(loc="best")
    plt.tight_layout()
    # plt.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08, wspace=0.15, hspace=0.25)
    if save:
        plt.savefig(save + f'/slide_pred_plot.png', format='png', dpi=200, bbox_inches='tight')
    plt.show()


def slide_pred_accuracy(case, metric='C_R', save=None):
    preds, trues, _ = pred_true_load(case)
    assert preds.shape == trues.shape

    slide_step = preds.shape[1]
    pred = preds[:, :, 0][::slide_step]
    true = trues[:, :, 0][::slide_step]

    farm_capacity = 39.95
    slide_num = 30
    if pred.shape[0] <= slide_num * 1.2:
        slide_num = pred.shape[0]
    pred = pred[:slide_num]
    true = true[:slide_num]

    metric_dict = {'E_rmse': E_rmse, 'E_mae': E_mae,
                   'E_me': E_me, 'r': r, 'C_R': C_R, 'Q_R': Q_R}
    precisions = []
    for i in range(pred.shape[0]):
        precisions.append(metric_dict[metric](pred[i], true[i], farm_capacity))

    precisions = np.array(precisions)
    width = 0.7

    fig, ax = plt.subplots(figsize=(slide_num * 0.6, 8), dpi=100)
    bars = ax.bar(np.arange(len(precisions)) + 1.7 - width, precisions, width, color='skyblue')
    ax.axhline(precisions.mean(), color='b', linestyle='--', lw=1.5, label=f'Mean={precisions.mean():.2f}')
    ax.axhline(85., color='r', linestyle='--', lw=1.5, alpha=1.)
    ax.axhline(100., color='k', linestyle='--', lw=1.5, alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., yval, f'{yval:.1f}',
                va='bottom', ha='center', fontsize=12)

    ax.set_xlabel('Days', fontsize=15)
    ax.set_ylim([precisions.min() - 10., 110.])
    ax.set_ylabel('Accuracy (%)', fontsize=15)
    # ax.set_yticks([20., 40., 60., 80., 100.])
    ax.yaxis.set_major_locator(MultipleLocator(20.))
    ax.set_xticks(np.arange(1, len(precisions) + 5, 5))
    ax.set_title(f'Prediction Precision with {metric}', fontsize=15)
    ax.tick_params(labelsize=18, colors='k', direction='in',
                   top=True, bottom=True, left=True, right=True)
    tick_labs = ax.get_xticklabels() + ax.get_yticklabels()
    [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    ax.legend(loc="best", fontsize=15)
    if save:
        plt.savefig(save + f'/slide_pred_precision.png', format='png', dpi=200, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    case_1 = 'LTF_test_Informer_turbine_ftS_sl96_ll48_pl16_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0'

    # slide_pred_plot(case_1)
    slide_pred_accuracy(case_1)
