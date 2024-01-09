import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator


from utils.metrics import E_rmse, E_mae, E_me, r, C_R, Q_R, \
    Acc_day_ahead, Acc_hour_ahead_power


default_path = 'results/'

subplot_layouts = {2: (2, 1), 3:(3, 1), 4: (2, 2), 6: (2, 3),
                   9: (3, 3), 12: (3, 4), 16: (4, 4), 20: (4, 5),
                   24: (4, 6), 25: (5, 5), 36: (6, 6), 49: (7, 7),}

level_points = np.array([0, 0.2, 0.5, 0.7, 1., np.inf])


def pred_true_load(case):
    if os.path.exists(os.path.join(default_path, f'{case}/test.npy')):
        test_data = np.load(os.path.join(default_path, f'{case}/test.npy'), allow_pickle=True)
        preds, trues, times = test_data[:, :, 0], test_data[:, :, 1], test_data[:, :, 2]
    else:
        preds= np.load(os.path.join(default_path, f'{case}/pred.npy'))
        trues = np.load(os.path.join(default_path, f'{case}/true.npy'))

        if os.path.exists(os.path.join(default_path, f'{case}/time.npy')):
            times = np.load(os.path.join(default_path, f'{case}/time.npy'), allow_pickle=True)
        else:
            times = None

    return preds, trues, times


def slide_pred_plot(case, save=None):
    preds, trues, times = pred_true_load(case)
    assert preds.shape == trues.shape
    if times is not None:
        assert preds.shape == times.shape
    slide_step = preds.shape[1]
    pred = preds[:, :, 0][::slide_step]
    true = trues[:, :, 0][::slide_step]
    time = times[:, :, 0][::slide_step]

    capacity = 39.95
    slide_num = 30
    subplot_num = 4
    if pred.shape[0] <= slide_num:
        slide_num = pred.shape[0]
    slide_num_subplot = slide_num // subplot_num + 1
    point_num = int(slide_num * slide_step)
    print('Slide prediction ploting ....')
    print('Number of Subplots: ', subplot_num)
    print('Number of Predictions: ', slide_num)
    print('Number of Datasamples: ', point_num)

    if times is not None:
        time = time[:slide_num]
    else:
        time = np.arange(point_num).reshape(pred.shape)
    slide_split = np.concatenate((
        np.arange(0, slide_num, slide_num_subplot)[:subplot_num],
        [slide_num]))
    pred = pred[:slide_num]
    true = true[:slide_num]

    fig, ax = plt.subplots(subplot_num, 1, figsize=(subplot_num * 6.5, slide_num_subplot * 2),
                           sharey=True, dpi=100)
    for ii, axi in enumerate(ax.flatten()):
        start, end = slide_split[ii], slide_split[ii + 1]
        time_stamp = time[start:end].flatten().astype(np.datetime64)
        axi.plot(time_stamp, pred[start:end].flatten(), label='pred', lw=2.)
        axi.plot(time_stamp, true[start:end].flatten(), label='true', lw=2.)
        time_end = np.datetime64(time_stamp.min()) + np.timedelta64(slide_num_subplot + 1, 'D')
        time_scope = pd.date_range(time_stamp.min(), time_end, freq='15T').values
        for split_line in np.arange(0, min(slide_num_subplot, end - start) + 1):
            axi.axvline(time_scope[split_line * slide_step], color='r', linestyle='--', lw=1.2)
        axi.axhline(capacity, color='k', linestyle='--', lw=0.8)
        if ii == subplot_num - 1:
            axi.set_xlabel('Time', fontsize=15)
        for xtick in axi.get_xticklabels():
            xtick.set_rotation(30); xtick.set_horizontalalignment('right')
        axi.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        axi.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d | %H:%M"))
        axi.set_ylabel('Power (MW)', fontsize=15)
        axi.set_ylim([0., 1.2 * capacity])
        axi.tick_params(labelsize=13, colors='k', direction='in',
                        top=True, bottom=True, left=True, right=True)
        tick_labs = axi.get_xticklabels() + axi.get_yticklabels()
        [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
        axi.legend(loc="best")
    fig.suptitle(f'Prediction Plot of {case}', fontsize=18, y=0.935)
    # plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.15, hspace=0.35)
    if save:
        plt.savefig(save + f'/slide_pred_plot.png', format='png', dpi=200, bbox_inches='tight')
    plt.show()


def slide_pred_accuracy(case, metric='C_R', acc=False, save=None):
    preds, trues, times = pred_true_load(case)
    assert preds.shape == trues.shape

    slide_step = preds.shape[1]
    pred = preds[:, :, 0][::slide_step]
    true = trues[:, :, 0][::slide_step]

    acc_max = 100.
    acc_require = 85.
    capacity = 39.95
    slide_num = 30
    if pred.shape[0] <= slide_num:
        slide_num = pred.shape[0]
    pred = pred[:slide_num]
    true = true[:slide_num]

    print('Prediction Accuracy calculating ....')
    print('Evaluation Time Range: ', times.min(), 'to', times.max())

    metric_dict = {'E_rmse': E_rmse,
                   'E_mae': E_mae,
                   'E_me': E_me,
                   'r': r,
                   'C_R': C_R,
                   'Q_R': Q_R}

    precisions = []
    accuracy = []
    for i in range(pred.shape[0]):
        precisions.append(metric_dict[metric](pred[i], true[i], capacity))
        if acc:
            accuracy.append(Acc_day_ahead(pred[i], true[i], capacity))
    precisions = np.array(precisions)
    accuracy = np.array(accuracy)

    bar_colors = 'skyblue'
    bar_hatches = ''
    if acc:
        precisions = np.where(precisions >= acc_require, precisions, accuracy)
        bar_colors = np.where(precisions >= acc_require, 'skyblue', 'orange')
        # bar_hatches = np.where(precisions >= acc_require, '', '///')

    width = 0.7

    fig, ax = plt.subplots(figsize=(slide_num * 0.6, 8), dpi=100)
    bars = ax.bar(np.arange(len(precisions)) + 1.7 - width, precisions, width,
                  color=bar_colors, hatch=bar_hatches,)

    ax.axhline(precisions.mean(), color='b', linestyle='--', lw=1.5, label=f'Mean: {precisions.mean():.2f}')
    ax.axhline(acc_require, color='g', linestyle='--', lw=1.5, alpha=1., label=f'Required: {acc_require:.2f}')
    ax.axhline(acc_max, color='k', linestyle='--', lw=1.5, alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., yval, f'{yval:.1f}',
                va='bottom', ha='center', fontsize=12)

    ax.set_xlabel(f'Days ({times.min()} to {times.max()})', fontsize=15)
    # ax.set_xlim([0., len(precisions) + 3])
    ax.set_xticks(np.linspace(1, len(precisions), 9, endpoint=True, dtype=int))
    ax.set_ylabel(f'{metric} (%)', fontsize=15)
    ax.set_ylim([precisions.min() - acc_max * 0.1, acc_max * 1.06])
    # ax.set_yticks([20., 40., 60., 80., 100.])
    ax.yaxis.set_major_locator(MultipleLocator(20.))
    ax.set_title(f'Prediction Precision with {case}', fontsize=12, y=1.01)
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
    case_2 = 'LTF_test_Informer_turbine_ftS_sl256_ll128_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0'
    case_3 = 'LTF_farm_PatchTST_turbine_ftS_ti5_uf0_uh0_sl256_ll128_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0'

    slide_pred_plot(case_1)
    # slide_pred_accuracy(case_3)
