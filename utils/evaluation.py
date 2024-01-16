import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from typing import Optional, Tuple, List

try:
    from utils.divider import dataset_divider
    from utils.tools import speed_power_converter, ModelConfig
    from utils.metrics import E_rmse, E_mae, E_me, r, C_R, Q_R, \
        Acc_day_ahead, Acc_hour_ahead_power
except:
    from divider import dataset_divider
    from tools import speed_power_converter, ModelConfig
    from metrics import E_rmse, E_mae, E_me, r, C_R, Q_R, \
        Acc_day_ahead, Acc_hour_ahead_power


def pred_true_load(
    case_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    print('Loading prediction and true data ....')

    if os.path.exists(case_path + '/test.npy'):
        test_data = np.load(case_path + '/test.npy', allow_pickle=True)
        preds, trues, times = test_data[0], test_data[1], test_data[2]
    else:
        preds= np.load(case_path, '/pred.npy')
        trues = np.load(case_path, '/true.npy')

        if os.path.exists(case_path, '/time.npy'):
            times = np.load(case_path, '/time.npy', allow_pickle=True)
        else:
            times = None

    return preds, trues, times


def slide_pred_plot(
    case: Optional[str] = None,
    convert: Optional[speed_power_converter] = None,
    save: Optional[str] = None,
    data: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]] = ()
    ) -> None:
    if case is not None:
        preds, trues, times = pred_true_load(case)

        slide_step = preds.shape[1]
        f_dim = preds.shape[2]
        pred = preds[:, :, :f_dim][::slide_step]
        true = trues[:, :, :f_dim][::slide_step]
        assert preds.shape == trues.shape

        if times is not None:
            assert preds.shape == times.shape
            time = times[:, :, 0][::slide_step]
    else:
        assert len(data) == 3
        preds, trues, times = data
        assert isinstance(times, np.ndarray)

        slide_step = preds.shape[1]
        pred = preds[:, :, 0]
        true = trues[:, :, 0]
        time = times[:, :, 0]

    if convert is not None:
        pred = convert.func(pred)
        if convert.baseline is not None:
            true = convert.baseline(time)

    capacity = convert.capacity

    if convert.flag == 'power':
        ylabel = 'Power (MW)'
    else:
        ylabel = 'Wind Speed (m/s)'

    if case is not None:
        case_name = case.split('/')[-1]
        title_label = f'Prediction Plot of {case_name}'
    else:
        title_label = 'Prediction Plot'

    if case is not None:
        slide_num = 30
        if pred.shape[0] <= slide_num:
            slide_num = pred.shape[0]
    else:
        slide_num = pred.shape[0]

    subplot_num = 4
    slide_num_subplot = slide_num // subplot_num + 1
    point_num = int(slide_num * slide_step)
    print('Slide prediction ploting ....')
    print('\tSubplots: ', subplot_num)
    print('\tPredictions: ', slide_num)
    print('\tDatasamples: ', point_num)

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
        time_stamp = pd.to_datetime(time[start:end].flatten())
        axi.plot(time_stamp, pred[start:end].flatten(), label='pred', lw=2.5)
        axi.plot(time_stamp, true[start:end].flatten(), label='true', lw=2.5)
        time_end = np.datetime64(time_stamp.min()) + np.timedelta64(slide_num_subplot + 1, 'D')
        time_scope = pd.date_range(time_stamp.min(), time_end, freq='15T').values
        for split_line in np.arange(0, min(slide_num_subplot, end - start) + 1):
            axi.axvline(time_scope[split_line * slide_step], color='r', linestyle='--', lw=1.2)
        axi.axhline(capacity, color='k', linestyle='--', lw=0.8)
        if ii == subplot_num - 1:
            axi.set_xlabel('Time', fontsize=15)
        for xtick in axi.get_xticklabels():
            # xtick.set_rotation(30)
            # xtick.set_horizontalalignment('right')
            xtick.set_horizontalalignment('center')
        axi.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        axi.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d | %H:%M"))
        axi.set_ylabel(ylabel, fontsize=15)
        axi.set_ylim([0., 1.2 * capacity])
        axi.tick_params(labelsize=15, colors='k', direction='in',
                        top=True, bottom=True, left=True, right=True)
        tick_labs = axi.get_xticklabels() + axi.get_yticklabels()
        [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
        axi.legend(loc="best")
    fig.suptitle(title_label, fontsize=18, y=0.935)
    # plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.15, hspace=0.25)
    if save:
        plt.savefig(save, format='png', dpi=200, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    return None

def slide_pred_accuracy(
    case: Optional[str] = None,
    metric: str = 'Acc_day_ahead',
    convert: Optional[speed_power_converter] = None,
    save: Optional[str] = None,
    data: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]] = ()
    ) -> float:
    if case is not None:
        preds, trues, times = pred_true_load(case)

        slide_step = preds.shape[1]
        f_dim = preds.shape[2]
        pred = preds[:, :, :f_dim][::slide_step]
        true = trues[:, :, :f_dim][::slide_step]
        assert preds.shape == trues.shape

        if times is not None:
            assert preds.shape == times.shape
            time = times[:, :, 0][::slide_step]
    else:
        assert len(data) == 3
        preds, trues, times = data
        assert isinstance(times, np.ndarray)

        slide_step = preds.shape[1]
        pred = preds[:, :, 0]
        true = trues[:, :, 0]
        time = times[:, :, 0]

    if convert is not None:
        pred = convert.func(pred)
        if convert.baseline is not None:
            true = convert.baseline(time)

    capacity = convert.capacity
    acc_max = 100.
    acc_require = 85.

    if case is not None:
        case_name = case.split('/')[-1]
        title_label = f'Prediction Precision with {case_name}'
    else:
        title_label = 'Prediction Precision'

    if case is not None:
        slide_num = 30
        if pred.shape[0] <= slide_num:
            slide_num = pred.shape[0]
    else:
        slide_num = pred.shape[0]

    pred = pred[:slide_num]
    true = true[:slide_num]

    print(f'Prediction Accuracy calculating ({metric}) ....')
    time_start = pd.to_datetime(time[0, 0]).strftime('%Y-%m-%d %H:%M:%S')
    time_end = pd.to_datetime(time[-1, -1]).strftime('%Y-%m-%d %H:%M:%S')
    print(f'\tTime Range: {time_start} to {time_end}')

    metric_dict = {
        'E_rmse': E_rmse,
        'E_mae': E_mae,
        'E_me': E_me,
        'r': r,
        'C_R': C_R,
        'Q_R': Q_R,
        'Acc_day_ahead': Acc_day_ahead,
        # 'Acc_hour_ahead_power': Acc_hour_ahead_power,
        }
    assert metric in metric_dict.keys(), 'Invalid metric input'

    precisions = []; powers = []
    for i in range(pred.shape[0]):
        precisions.append(metric_dict[metric](pred[i], true[i], capacity))
        if metric == 'Acc_day_ahead':
            powers.append(Acc_hour_ahead_power(pred[i], true[i], capacity, capacity))
    precisions = np.array(precisions)
    if metric == 'Acc_day_ahead':
        powers = np.where(precisions >= acc_require, 0., np.array(powers))

    bar_colors = 'skyblue'
    bar_hatches = ''
    # if metric == 'Acc_day_ahead':
    #     precisions = np.where(precisions >= acc_require, precisions, accuracy)
    #     bar_colors = np.where(precisions >= acc_require, 'skyblue', 'orange')
    #     bar_hatches = np.where(precisions >= acc_require, '', '///')

    width = 0.7

    fig, ax = plt.subplots(figsize=(slide_num * 0.6, 8), dpi=100)
    bars = ax.bar(np.arange(len(precisions)) + 1.7 - width, precisions, width,
                  color=bar_colors, hatch=bar_hatches,)

    acc_mean = precisions.mean()
    ax.axhline(acc_mean, color='b', linestyle='--', lw=1.5, label=f'Mean: {acc_mean:.2f}')
    ax.axhline(acc_require, color='g', linestyle='--', lw=1.5, alpha=1., label=f'Required: {acc_require:.2f}')
    ax.axhline(acc_max, color='k', linestyle='--', lw=1.5, alpha=0.5)

    for i, bar in enumerate(bars):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., yval, f'{yval:.1f}',
                va='bottom', ha='center', fontsize=12)
        if metric == 'Acc_day_ahead':
            ax.text(bar.get_x() + bar.get_width() / 2., yval + acc_max * 0.025, f'({powers[i]:.0f})',
                    va='bottom', ha='center', color='r', fontsize=12)

    ax.set_xlabel(f'Days ({time_start} to {time_end})', fontsize=15)
    # ax.set_xlim([0., len(precisions) + 3])
    ax.set_xticks(np.linspace(1, len(precisions), 9, endpoint=True, dtype=int))
    ax.set_ylabel(f'{metric} (%)', fontsize=15)
    ax.set_ylim([precisions.min() - acc_max * 0.1, acc_max * 1.06])
    ax.yaxis.set_major_locator(MultipleLocator(20.))
    ax.set_title(title_label, fontsize=12, y=1.01)
    ax.tick_params(labelsize=18, colors='k', direction='in',
                   top=True, bottom=True, left=True, right=True)
    tick_labs = ax.get_xticklabels() + ax.get_yticklabels()
    [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    ax.legend(loc="best", fontsize=15)
    if save:
        plt.savefig(save, format='png', dpi=200, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    return acc_mean


def cfd_converted_data_eval(
    ref: str = 'wrf',
    idx: Optional[int] = None,
    year: int = 2022,
    month: int = 6,
    col: str = 'Patv_Total',
    step: int = 96,
    ) -> None:
    wt_df = pd.read_csv('datasets\WFP\Turbine_Patv_Spd_15min_filled.csv', index_col=None, header=0)
    if ref == 'wrf':
        cfd_df = pd.read_csv('datasets\CFD\wrf_converted_turbine_data.csv', index_col=None, header=0)
    elif ref == 'wts':
        cfd_df = pd.read_csv('datasets\CFD\wts_converted_turbine_data.csv', index_col=None, header=0)
    else:
        raise ValueError('Invalid ref input')

    if idx is None:
        start = datetime.date(year=year, month=month, day=1)
        end = datetime.date(year=year, month=month + 1, day=1)
        eval_times = pd.date_range(start, end, freq='15T', inclusive='left')
        days_num = (eval_times.max() - eval_times.min()).days + 1
    else:
        assert isinstance(idx, int) and idx >= 1 and idx <= 12
        DEFAULT_DATASET_LEN = 35136
        border1s, border2s = dataset_divider(DEFAULT_DATASET_LEN, idx)
        idx_start, idx_end = border1s[-1][0], border2s[-1][0]
        start = wt_df['date'][idx_start]; end = wt_df['date'][idx_end]
        eval_times = pd.date_range(start, end, freq='15T', inclusive='left')
        days_num = 30 if len(eval_times) >= 30 * step else len(eval_times) // step
        eval_times = eval_times[:days_num * step]
    # convert the time range to string
    time_start = eval_times.min().strftime('%Y-%m-%d %H:%M:%S')
    time_end = eval_times.max().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Time Range [{days_num} days]: {time_start} to {time_end}')

    wt_df['date'] = pd.to_datetime(wt_df['date']); wt_df.set_index('date', inplace=True)
    cfd_df['date'] = pd.to_datetime(cfd_df['date']); cfd_df.set_index('date', inplace=True)

    # extract data of evaluation time range
    time = eval_times.values[:days_num * step].reshape(days_num, step, -1)
    wt_data = wt_df.loc[eval_times, col].values.reshape(days_num, step, -1)
    cfd_data = cfd_df.loc[eval_times, col].values.reshape(days_num, step, -1)
    # print('Number of true samples: ', wt_data.shape)
    # print('Number of pred samples: ', cfd_data.shape)

    convert = speed_power_converter(col, )

    if idx is None:
        save_suffix = f'datasets/CFD/cfd_{ref}_{year}_{month}'
    else:
        save_suffix = f'datasets/CFD/cfd_{ref}_idx_{idx}'

    slide_pred_plot(
        data=(cfd_data, wt_data, time),
        convert=convert,
        save=save_suffix + '_pred_plot.png',
        )
    slide_pred_accuracy(
        data=(cfd_data, wt_data, time),
        convert=convert,
        save=save_suffix + '_pred_acc.png',
        )


if __name__ == '__main__':
    case_1 = 'LTF_test_Informer_turbine_ftS_sl96_ll48_pl16_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0'

    # slide_pred_plot(case_1)
    # slide_pred_accuracy(case_1)

    cfd_converted_data_eval(idx=1)

    # for data_idx in range(1, 13):
    #     cfd_converted_data_eval(idx=data_idx)