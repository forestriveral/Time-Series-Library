import xlsxwriter
import os, sys, copy
import json, time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# from metrics import metric

default_path = '../results/'

subplot_layouts = {2: (2, 1), 3:(3, 1), 4: (2, 2), 6: (2, 3), 9: (3, 3), 12: (3, 4), 16: (4, 4),
                   20: (4, 5), 24: (4, 6), 25: (5, 5), 36: (6, 6), 49: (7, 7),}

level_points = np.array([0, 0.2, 0.5, 0.7, 1., np.inf])


def random_pred_plot(folder, show=True, save=False):
    result_path = default_path + folder
    random_data = np.load(result_path + '/random_preds_trues.npy', allow_pickle=True)
    pred_data, true_data, time_label = random_data[0], random_data[1], random_data[2]
    # print(true_data.shape, pred_data.shape, time_label.shape)

    layouts = subplot_layouts[random_data.shape[1]]; m, n = layouts[0], layouts[1]
    fig, ax = plt.subplots(m, n, figsize=(n * 5, m * 5), sharey=True, dpi=110)
    for i, axi in enumerate(ax.flatten()):
        time_stamp = [np.datetime64(t) for t in time_label[i, :, 0]]
        axi.plot(time_stamp, true_data[i, :, 0], 'k', label='True')
        axi.plot(time_stamp, pred_data[i, :, 0], 'b', label='Pred')
        for xtick in axi.get_xticklabels():
            xtick.set_rotation(30); xtick.set_horizontalalignment('right')
        axi.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        axi.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d | %H:%M"))
        axi.format_ydata = lambda x : f'$x:.2f$'
        axi.tick_params(axis="both", direction="in", labelsize=8)
        axi.legend(loc="upper right")
        axi.xaxis.set_visible(True)
        axi.yaxis.set_visible(True)
    plt.tight_layout()
    plt.subplots_adjust(left=0.07, right=0.93, top=0.92,
                        bottom=0.08, wspace=0.15, hspace=0.25)
    if save:
        plt.savefig(folder + '/random_pred_plot.png', format='png',
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()


def single_sliding_pred_plot(folder, step=1, plot_num=2000, path='./results'):
    result_path = path + '/' + folder
    pred_data = np.load(result_path + '/sliding_pred.npy')
    true_data = np.load(result_path + '/sliding_true.npy')
    print(pred_data.shape, true_data.shape)

    pred = pred_data[:, step - 1, 0]
    true = true_data[:, step - 1, 0]
    # C_wt = np.ones_like(pred) * 39.95
    C_wt = np.ones_like(pred) * 3200
    e = E_RMSE(pred, true, C_wt); c = C_R(pred, true, C_wt); q = Q_R(pred, true, C_wt)
    # print(f'E_RMSE: {e:.3f}, C_R: {c:.3f}, Q_R: {q:.3f}')

    _, ax = plt.subplots(figsize=(25, 8), dpi=80)
    ax.plot(pred[:plot_num], '-r', label='pred', lw=2.)
    ax.plot(true[:plot_num], '-k', label='true', lw=2.)
    ax.set_title(f'Pred Step: {step} E_RMSE: {e:.3f}, C_R: {c:.2f}, Q_R: {q:.2f}')
    ax.legend(loc='best')

    plt.savefig(result_path + f'/sliding_pred_plot.png', format='png', dpi=200, bbox_inches='tight')
    print(f'Sliding Pred Plot saved to {result_path}.')
    plt.close()


def multiple_sliding_pred_plot(folder, bounds=False, show=True, save=False):
    result_path = default_path + folder
    sliding_data = np.load(result_path + '/multiple_sliding_data.npy', allow_pickle=True)
    sliding_data[:, :4, :] = np.clip(sliding_data[:, :4, :], 0., 100.)
    setting = case_setting_unpack(folder); seq_len = setting['seq_len']; pred_len = setting['pred_len']
    # print(sliding_data.shape)

    layouts = subplot_layouts[sliding_data.shape[0]]; m, n = layouts[0], layouts[1]
    fig, ax = plt.subplots(m, n, figsize=(n * 6, m * 4), sharey=True, dpi=110)
    for i, axi in enumerate(ax.flatten()):
        time_stamp = [np.datetime64(t) for t in sliding_data[i, 4, :]]
        axi.plot(time_stamp, sliding_data[i, 0, :], '-k', linewidth=1.5, label='True')
        axi.plot(time_stamp, sliding_data[i, 1, :], '-r', linewidth=1.5, label='Pred')
        if bounds:
            up_bound = sliding_data[i, 2, :][seq_len:]; low_bound = sliding_data[i, 3, :][seq_len:]
            axi.fill_between(time_stamp[seq_len:], list(low_bound), list(up_bound), alpha=.5, linewidth=0.)
        axi.axvline(time_stamp[seq_len], color='b', alpha=0.7, linestyle='--', linewidth=2.)
        axi.axvline(time_stamp[-pred_len], color='y', alpha=0.7, linestyle='--', linewidth=2.)
        for xtick in axi.get_xticklabels():
            xtick.set_rotation(30); xtick.set_horizontalalignment('right')
        # axi.xaxis.set_major_locator(mdates.HourLocator(interval=30))
        axi.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d | %H:%M"))
        axi.format_ydata = lambda x : f'$x:.2f$'
        axi.tick_params(axis="both", direction="in", labelsize=8)
        axi.legend(loc="best")
        axi.xaxis.set_visible(True)
        axi.yaxis.set_visible(True)
    plt.tight_layout()
    plt.subplots_adjust(left=0.04, right=0.96, top=0.92,
                        bottom=0.08, wspace=0.15, hspace=0.25)
    if save:
        plt.savefig(folder + '/multiple_sliding_plot.png', format='png',
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()


def multiple_sliding_eval_plot(folder, show=True, save=False):
    result_path = default_path + folder
    sliding_data = np.load(result_path + '/multiple_sliding_data.npy', allow_pickle=True)
    sliding_data[:, :4, :] = np.clip(sliding_data[:, :4, :], 0., 100.)
    setting = case_setting_unpack(folder); seq_len = setting['seq_len']; pred_len = setting['pred_len']
    # print(sliding_data.shape)

    layouts = subplot_layouts[sliding_data.shape[0]]; m, n = layouts[0], layouts[1]
    fig, ax = plt.subplots(m, n, figsize=(n * 6, m * 4), sharey=True, dpi=110)
    for i, axi in enumerate(ax.flatten()):
        time_stamp = [np.datetime64(t) for t in sliding_data[i, 4, :]]
        error = np.abs(sliding_data[i, 0, :] - sliding_data[i, 1, :]).copy(); true = sliding_data[i, 0, :].copy()
        errors, time_stamps = error_filter_divider(true, error, time_stamp)
        labels = ['Error <= 0.2', '0.2 < Error <= 0.5', '0.5 < Error <= 0.7', '0.7 < Error <= 1.0', 'Outliers']
        markers = ['o', 'o', 's', '^', 'x']; markercolors = ['g', 'b', 'y', 'purple', 'r']
        for i in range(len(errors)):
            count_percent = f'{len(errors[i]) / len(error):.2f}'
            label_with_count = labels[i] + ' (' + count_percent + ')'
            axi.plot(time_stamps[i], errors[i], c='w', lw=0., label=label_with_count, markersize=5,
                     marker=markers[i], markeredgecolor=markercolors[i], markeredgewidth=0.8)
        axi.set_ylim([0., 1.1])
        axi.axvline(time_stamp[seq_len], color='b', alpha=0.7, linestyle='--', linewidth=2.)
        axi.axvline(time_stamp[-pred_len], color='y', alpha=0.7, linestyle='--', linewidth=2.)
        for xtick in axi.get_xticklabels():
            xtick.set_rotation(30); xtick.set_horizontalalignment('right')
        # axi.xaxis.set_major_locator(mdates.HourLocator(interval=30))
        axi.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d | %H:%M"))
        axi.format_ydata = lambda x : f'$x:.2f$'
        axi.tick_params(axis="both", direction="in", labelsize=8)
        axi.legend(loc="best", edgecolor='None',)
        axi.xaxis.set_visible(True)
        axi.yaxis.set_visible(True)
    plt.tight_layout()
    plt.subplots_adjust(left=0.04, right=0.96, top=0.92,
                        bottom=0.08, wspace=0.15, hspace=0.25)
    if save:
        plt.savefig(folder + '/multiple_sliding_eval.png', format='png',
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()


def relative_sliding_pred_eval(folder, data='multiple', save=True):
    result_path = default_path + folder
    sliding_data = np.load(result_path + f'/{data}_sliding_data.npy', allow_pickle=True)
    setting = case_setting_unpack(folder); seq_len = setting['seq_len']; pred_len = setting['pred_len']
    effective_data = np.clip(sliding_data[:, :4, seq_len:-pred_len], 0., 100.)
    precision = []

    for i in range(sliding_data.shape[0]):
        error = np.abs(effective_data[i, 0, :] - effective_data[i, 1, :]).copy(); true = effective_data[i, 0, :].copy()
        errors, _ = error_filter_divider(true, error, np.zeros_like(true))
        tmp_precision = []; tmp_count = []
        for j in range(len(errors)):
            tmp_count.append(len(errors[j]) / len(error))
            tmp_precision.append(errors[j].mean())
        precision.append([tmp_precision, tmp_count])
    precision = np.array(precision); weight = precision[:, 1, :-1] / precision[:, 1, :-1].sum(axis=1)[:, None]
    overall = np.average(precision[:, 0, :-1], axis=1, weights=weight)

    if save:
        save_path = folder + f'/{data}_sliding_eval.txt'
        precision_dict = {
            'overall': np.round(overall, 3).tolist(),
            'precision': np.round(precision, 3).tolist()
            }
        with open(save_path, 'w+') as f:
            json.dump(precision_dict, f, indent=2)

    # print(precision)
    # print(overall)
    return precision, overall


def sliding_pred_eval(folder, plot='simple', save=False, path=default_path):
    result_path = path + '/' + folder
    sliding_data = np.load(result_path + '/sliding_data.npy', allow_pickle=True)
    sliding_data[:, :-2, :] = np.clip(sliding_data[:, :-2, :], 0., 100.)
    # sliding_data[:, -2, :] = np.ones((sliding_data.shape[0], sliding_data.shape[2])) * 39.95
    setting = case_setting_unpack(folder)
    seq_len = setting['seq_len']; pred_len = setting['pred_len']
    # print(sliding_data.shape)

    # precision evaluation for each prediction sequence on E_rmse, C_r, Q_r
    seq_num = sliding_data.shape[2]; eval_precision = []; eval_plot = []
    for data_idx in range(sliding_data.shape[0]):
        eval_acc_idx, eval_plot_idx = [], []
        for pred_indx in range(pred_len):
            pred_true = sliding_pred_format_index(
                sliding_data[data_idx, :-2, :], seq_len, pred_len, index=pred_indx + 1)
            eff_pred_true = pred_true[
                :, seq_len + pred_indx:seq_num - pred_len + pred_indx + 1]
            eff_C_wt = sliding_data[
                data_idx, -2, seq_len + pred_indx:seq_num - pred_len + pred_indx + 1]
            E_rmse_i, C_r_i, Q_r_i = sliding_pred_precision(
                np.concatenate((eff_pred_true, eff_C_wt[None, :]), axis=0))
            time_pred_true = np.concatenate((np.round(pred_true, decimals=3), sliding_data[data_idx, [-1], :]), axis=0)
            eval_plot_idx.append(time_pred_true); eval_acc_idx.append([E_rmse_i, C_r_i, Q_r_i])
        eval_precision.append(np.round(np.array(eval_acc_idx).T, decimals=3))
        eval_plot.append(np.array(eval_plot_idx))
    eval_plot = np.array(eval_plot); eval_precision = np.array(eval_precision)
    # average_precision = eval_precision.mean(axis=0)
    # print('Avg E_rmse: ', [float(f'{E_i:.3f}') for E_i in average_precision[0]])
    # print('Avg C_r: ', [float(f'{C_i:.3f}') for C_i in average_precision[1]])
    # print('Avg Q_r: ', [float(f'{Q_i:.3f}') for Q_i in average_precision[2]])
    if save:
        save_path = result_path + f'/sliding_eval.txt'
        with open(save_path, 'w+') as f: json.dump(eval_precision.tolist(), f, indent=2)

    plot_func_dict = {'simple': sliding_pred_simple_plot, 'all': sliding_pred_all_plot,
                      'None': lambda x, y, z, w: None}
    if plot in plot_func_dict.keys():
        precision = plot_func_dict[plot](eval_plot, eval_precision, seq_len, result_path)

    return precision


def sliding_pred_all_plot(plot_data, eval_data, seq_len, save_path, ignore_idx=[]):
    # plot evaluation for each prediction sequence
    pred_len = plot_data.shape[1]
    for data_idx in range(plot_data.shape[0]):
        if isinstance(ignore_idx, (int, list)):
            ignore_idx = [ignore_idx] if isinstance(ignore_idx, int) else ignore_idx
            if data_idx + 1 in ignore_idx: continue
        layouts = subplot_layouts[int(pred_len)]; m, n = layouts[0], layouts[1]
        fig, ax_idx = plt.subplots(m, n, figsize=(n * 6, m * 4),
                                    sharey=True, sharex=True, dpi=110)
        for i, axi in enumerate(ax_idx.flatten()):
            time_stamp = [np.datetime64(t) for t in plot_data[data_idx, i, -1, :]]
            pred_time_stamp = time_stamp[seq_len + i:-pred_len]
            pred_farm_power = plot_data[data_idx, i, 1, :][seq_len + i:-pred_len]
            axi.plot(time_stamp, plot_data[data_idx, i, 0, :], '-k', linewidth=1.5, label='True')
            axi.plot(pred_time_stamp, pred_farm_power, '-r', linewidth=1.5, label='Pred')
            axi.axvline(time_stamp[seq_len + i], color='b', alpha=0.7, linestyle='--', linewidth=2.)
            axi.axvline(time_stamp[-pred_len], color='y', alpha=0.7, linestyle='--', linewidth=2.)
            E_rmse_i, C_r_i, Q_r_i = eval_data[data_idx, :, i]
            title_i = f'Pred Step: {i + 1} | E_rmse: {E_rmse_i:.2f}, C_r: {C_r_i:.2f}, Q_r: {Q_r_i:.2f}'
            axi.set_title(title_i, fontsize=13); axi.set_ylim([0., 45.])
            for xtick in axi.get_xticklabels():
                xtick.set_rotation(30); xtick.set_horizontalalignment('right')
            # axi.xaxis.set_major_locator(mdates.HourLocator(interval=30))
            axi.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d | %H:%M"))
            axi.format_ydata = lambda x : f'$x:.2f$'
            axi.tick_params(axis="both", direction="in", labelsize=8)
            axi.legend(loc="best"); axi.xaxis.set_visible(True); axi.yaxis.set_visible(True)
        plt.tight_layout()
        plt.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08, wspace=0.15, hspace=0.25)
        plt.savefig(save_path + f'/sliding_eval_plot_{data_idx + 1}.png', format='png',
                    dpi=200, bbox_inches='tight')
        print(f'Eval All Plot {data_idx + 1} saved to {save_path}.')
        plt.close()

        return None


def sliding_pred_simple_plot(plot_data, eval_data, seq_len, save_path):
    # plot evaluation for each prediction sequence only with the 16th-step prediction
    pred_len = plot_data.shape[1]; avg_precision = []
    layouts = subplot_layouts[plot_data.shape[0]]; m, n = layouts[0], layouts[1]
    fig, ax = plt.subplots(m, n, figsize=(n * 6, m * 4), sharey=True, dpi=130)
    for data_idx, axi in enumerate(ax.flatten()):
        time_stamp = [np.datetime64(t) for t in plot_data[data_idx, -1, -1, :]]
        pred_time_stamp = time_stamp[seq_len + pred_len - 1:-pred_len]
        pred_farm_power = plot_data[data_idx, -1, 1, :][seq_len + pred_len - 1:-pred_len]
        axi.plot(time_stamp, plot_data[data_idx, -1, 0, :], '-k', linewidth=1.5, label='True')
        axi.plot(pred_time_stamp, pred_farm_power, '-r', linewidth=1.5, label='Pred')
        axi.fill_between(time_stamp[seq_len:seq_len + pred_len - 1],
                         np.zeros(pred_len - 1), np.ones(pred_len - 1) * 100.,
                         alpha=.8, linewidth=0, color='grey')
        axi.axvline(time_stamp[seq_len], color='b', alpha=0.7, linestyle='--', linewidth=2.)
        axi.axvline(time_stamp[-pred_len], color='y', alpha=0.7, linestyle='--', linewidth=2.)
        E_rmse_i, C_r_i, Q_r_i = eval_data[data_idx, :, -1]; avg_precision.append([E_rmse_i, C_r_i, Q_r_i])
        title_i = f'Pred Step: {pred_len} | E_rmse: {E_rmse_i:.2f}, C_r: {C_r_i:.2f}, Q_r: {Q_r_i:.2f}'
        axi.set_title(title_i, fontsize=12); axi.set_ylim([0., 45.])
        for xtick in axi.get_xticklabels():
            xtick.set_rotation(30); xtick.set_horizontalalignment('right')
        # axi.xaxis.set_major_locator(mdates.HourLocator(interval=30))
        axi.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d | %H:%M"))
        axi.format_ydata = lambda x : f'$x:.2f$'
        axi.tick_params(axis="both", direction="in", labelsize=8)
        axi.legend(loc="best"); axi.xaxis.set_visible(True); axi.yaxis.set_visible(True)
    avg_precision = np.array(avg_precision).mean(axis=0)
    avg_precision_txt = f'Avg: E_rmse: {avg_precision[0]:.2f} | C_r: {avg_precision[1]:.2f} | Q_r: {avg_precision[2]:.2f}'
    fig.suptitle(avg_precision_txt, fontsize=18, x=0.4, y=0.10, ha='left', va='bottom')
    plt.tight_layout()
    plt.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.20, wspace=0.12, hspace=0.4)
    plt.savefig(save_path + f'/sliding_eval_simple_plot.png', format='png', dpi=200, bbox_inches='tight')
    print(f'Eval Simple Plot saved to {save_path}.')
    plt.close()

    return avg_precision


def sliding_pred_output(folder, save=False):
    setting = case_setting_unpack(folder)
    seq_len = setting['seq_len']; pred_len = setting['pred_len']
    excel_columns = ['时间', '真实值'] + ['预测值' for _ in range(pred_len)]
    append_columns = ['time', 'actual value', '15min', '30min', '45min', '60min',
                      '1h15min', '1h30min', '1h45min', '2h', '2h15min', '2h30min',
                      '2h45min', '3h', '3h15min', '3h30min', '3h45min', '4h']
    result_path = default_path + folder
    sliding_data = np.load(result_path + '/sliding_data.npy', allow_pickle=True)
    writer = pd.ExcelWriter(result_path + '/sliding_pred_results.xlsx', engine="xlsxwriter")
    pred_array = np.zeros((sliding_data.shape[0], sliding_data.shape[1] - 2, pred_len + 2), dtype = 'object')
    for data_idx in range(sliding_data.shape[0]):
        time_idx = sliding_data[data_idx, -1, seq_len - 1:sliding_data.shape[2] - pred_len + 1]
        pred_array[data_idx, :, 0] = np.array(
            [time.strftime('%m/%d/%Y %H:%M', time.strptime(t,'%Y-%m-%d %H:%M')) for t in time_idx])
        pred_array[data_idx, :, 1] = sliding_data[data_idx, 0, seq_len - 1:sliding_data.shape[2] - pred_len + 1]
        for pred_indx in range(pred_array.shape[1] - 1):
            if pred_indx < pred_len - 1:
                pred_array[data_idx, pred_indx + 1, 2:2 + pred_indx + 1] = sliding_data[
                    data_idx, 1:1 + pred_indx + 1, seq_len + pred_indx][::-1]
            else:
                h_idx = 2 + pred_indx - pred_len
                pred_array[data_idx, pred_indx + 1, 2:] = sliding_data[
                    data_idx, h_idx:h_idx + pred_len, seq_len + pred_indx][::-1]
        preds_idx = np.concatenate((np.array(append_columns)[None, :], pred_array[data_idx]), axis=0)
        preds_idx = pd.DataFrame(preds_idx, columns=excel_columns, index=None)
        preds_idx.to_excel(writer, sheet_name=f'滑动预测表_{data_idx}', index=False)
    writer.close()
    return pred_array


def batch_runner_formatter(param, precision=None):
    report_cols = [
            'model_id', 'E_rmse', 'C_r', 'Q_r', 'mae', 'mse', 'rmse', 'mape', 'mspe',
            'is_leaking', 'is_noised', 'features', 'in_output', 'seq_len', 'label_len',
            'pred_len', 'enc_in', 'dec_in', 'c_out', 'd_model', 'n_heads', 'e_layers',
            'd_layers', 's_layers', 'd_ff', 'factor', 'padding', 'distil', 'dropout',
            'attn', 'embed', 'activation', 'output_attention', 'train_epochs', 'batch_size',
            'patience', 'learning_rate', 'loss', 'lradj'
            ]
    assert param is not None, 'param is None.'
    report_file = param.results + '/' + 'batch_train_report.csv'
    if not os.path.exists(report_file):
        pd.DataFrame(columns=report_cols).to_csv(report_file, index=True)
    report_df = pd.read_csv(report_file, index_col=0)
    add_dict = {key: None for key in report_cols}; param_dict = vars(param)
    add_dict = dict_update_formatter(add_dict, param_dict)
    if precision is not None:
        add_dict = dict_update_formatter(add_dict, precision)
    report_df = report_df.append(add_dict, ignore_index=True)
    report_df.to_csv(report_file, index=True)


def dict_update_formatter(dict_A, dict_B):
    # Two dictionaries A and B
    # dict_A = {
    #     'key1': 10,
    #     'key2': 20,
    #     'key3': 30,
    #     'key4': 40
    # }

    # dict_B = {
    #     'key2': 25,
    #     'key4': 45,
    #     'key5': 50
    # }
    A, B = copy.deepcopy(dict_A), copy.deepcopy(dict_B)
    # Iterate through keys in dictionary B
    for key, value in B.items():
        if key in A:
            A[key] = value

    return A



def sliding_pred_precision(sliding_data):
    P_mean, P_pred, C_wt = sliding_data[0, :], sliding_data[1, :], sliding_data[2, :]
    P_mean = P_mean[C_wt > 0]; P_pred = P_pred[C_wt > 0]; C_wt = C_wt[C_wt > 0]
    return E_RMSE(P_mean, P_pred, C_wt), C_R(P_mean, P_pred, C_wt), Q_R(P_mean, P_pred, C_wt)


def E_RMSE(P_mean, P_pred, C_wt):
    assert P_mean.shape == P_pred.shape == C_wt.shape
    return np.sqrt(np.sum(((P_mean - P_pred) / C_wt) ** 2) / P_mean.shape[0])


def C_R(P_mean, P_pred, C_wt):
    return (1 - E_RMSE(P_mean, P_pred, C_wt)) * 100.


def Q_R(P_mean, P_pred, C_wt):
    B = np.where(np.abs(P_mean - P_pred) / C_wt < 0.25, 1., 0.)
    return B.sum() / B.shape[0] * 100.


def next_day_accuracy(pred, true, C_wt):
    assert pred.shape == true.shape == C_wt.shape
    return 1 - np.sqrt(np.sum((true - pred)**2 * (np.abs(true - pred) / np.sum(np.abs(true - pred))))) / C_wt


def error_filter_divider(true, error, time):
    assert error.shape == true.shape; errors = []
    for i in range(error.shape[0]):
        # print(error[i], true[i])
        if true[i] == 0:
            errors.append(1.)
        else:
            errors.append(error[i] / true[i])
            # if error_percent[-1] >= 1.:
            #     print(i, 'E:', error[i], 'T:', true[i], 'EP:', error[i] / true[i])
    errors = np.clip(np.array(errors), 0., 1.); divided_errors, divided_times = [], []
    for i in range(level_points.shape[0] - 1):
        a = level_points[i]; b = level_points[i + 1]; idx = np.where((errors >= a) & (errors < b))
        divided_errors.append(errors[idx[0]]), divided_times.append(np.array(time)[idx[0]])
    # print(divided_errors, divided_times)
    return divided_errors, divided_times


def sliding_pred_format(sliding_preds, seq_len):
    # define an array to store the sliding prediction comparison results
    # 1: true; 2: pred; 3: upper boundary; 4: lower boundary
    data_seqs = np.zeros((4, sliding_preds.shape[1]))
    data_seqs[0, :] = sliding_preds[0, :]
    data_seqs[1, :seq_len] = sliding_preds[0, :seq_len]
    for step in range(sliding_preds.shape[1] - seq_len):
        step_preds = sliding_preds[1:, seq_len + step]; step_preds = step_preds[step_preds != 0]
        # data_seqs[1, seq_len + step] = sliding_preds[step + 1, seq_len + step] \
        #     if step + 1 < sliding_preds.shape[0] else sliding_preds[-1, seq_len + step]
        data_seqs[1, seq_len + step] = step_preds[-1]
        data_seqs[2, seq_len + step] = np.max(step_preds)
        data_seqs[3, seq_len + step] = np.min(step_preds)
    # print(data_seqs.shape); print(data_seqs[:, -5:]); print(sliding_preds[-5:, -5:])
    return data_seqs


def sliding_pred_format_index(sliding_preds, seq_len, pred_len, index=1):
    # define an array to store the sliding prediction comparison results
    # 1: true; 2: pred
    assert index <= pred_len and index >= 1, 'Index shlould be smaller than pred_len'
    data_num = sliding_preds.shape[1]
    data_seqs = np.zeros((2, data_num)); data_seqs[0, :] = sliding_preds[0, :]
    data_seqs[1, :seq_len + index - 1] = sliding_preds[0, :seq_len + index - 1]
    fill_step_num = sliding_preds.shape[1] - seq_len - index + 1
    for step in range(fill_step_num):
        step_idx = seq_len + step + index - 1; pred_idx = step + 1
        data_seqs[1:, step_idx] = sliding_preds[pred_idx, step_idx] \
            if step_idx <= data_num - (pred_len - index) - 1 else sliding_preds[-1, step_idx]
    # print(data_seqs)
    return data_seqs


def sliding_pred_format_index_debug():
    seq_len = 6; pred_len = 3; index = 1
    test_preds = np.zeros((8, 15)); np.random.seed(1234)
    test_preds[0, :] = np.random.randint(0, 50, 15)
    for i in range(7):
        test_preds[i + 1, i + seq_len:i + seq_len + pred_len] = \
            np.random.randint(0, 50, pred_len)
    print(test_preds)
    sliding_pred_format_index(test_preds, seq_len, pred_len, index)


def case_setting_unpack(folder):
    result_path = default_path + folder
    case_name = Path(result_path).name; setting = case_name.split('_')
    return {'model_id':setting[0], 'model':setting[1], 'data':setting[2],
            'features':setting[3][-1], 'seq_len':int(setting[4][2:]),
            'label_len':int(setting[5][2:]), 'pred_len':int(setting[6][2:]),
            'd_model':int(setting[7][2:]), 'n_heads':int(setting[8][2:]),
            'e_layers':int(setting[9][2:]), 'd_layers':int(setting[10][2:]),
            'd_ff':int(setting[11][2:]), 'attn':setting[12][2:],
            'factor':int(setting[13][2:]), 'embed':setting[14][2:],
            'distil':bool(setting[15][2:]), 'mix':bool(setting[16][2:]),
            'des':setting[17], 'ii':int(setting[-1])}


def temporal_pred_plot(model):
    path = 'D:\Documents\Wind Farm Control\WFCC\output\inflows_3s_1214_5_2_60s.csv'
    wind_data = pd.read_csv(path, header=0).values
    wind_spd, wind_dir = wind_data[:, 0], wind_data[:, 1]
    ref_plot = sliding_pred_eval(model, plot='None', save=False)
    error = ref_plot[3, :6, 1, 176:236] / ref_plot[3, :6, 0, 176:236]
    print(error.shape); print(error)

    _, ax_wd = plt.subplots(2, 3, figsize=(6 * 3, 4 * 2), dpi=80)
    time_stamp = np.arange(wind_data.shape[0]); scale_wd = [0.70, 0.66, 0.60, 0.58, 0.54, 0.48]
    for i, axi in enumerate(ax_wd.flatten()):
        axi.plot(time_stamp, wind_dir, '-k', linewidth=1.5, label='True Wind Dir')
        axi.plot(time_stamp, wind_dir * error[i] * scale_wd[i], '-r', linewidth=1.5, label='Pred Wind Dir')
        axi.set_title(f'Pred Step: {i + 1}', fontsize=13)
        axi.tick_params(axis="both", direction="in", labelsize=8)
        axi.legend(loc="best"); axi.xaxis.set_visible(True); axi.yaxis.set_visible(True)
    plt.tight_layout()

    # _, ax_ws = plt.subplots(2, 3, figsize=(6 * 3, 4 * 2), dpi=80)
    # time_stamp = np.arange(wind_data.shape[0]); scale_ws = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
    # for i, axi in enumerate(ax_ws.flatten()):
    #     axi.plot(time_stamp, wind_spd, '-k', linewidth=1.5, label='True Wind Dir')
    #     axi.plot(time_stamp, wind_spd * error[i] * scale_ws[i], '-r', linewidth=1.5, label='Pred Wind Dir')
    #     axi.set_title(f'Pred Step: {i + 1}', fontsize=13)
    #     axi.tick_params(axis="both", direction="in", labelsize=8)
    #     axi.legend(loc="best"); axi.xaxis.set_visible(True); axi.yaxis.set_visible(True)
    # plt.tight_layout()

    plt.show()


def test_result_check(folder):
    result_path = default_path + folder
    preds = np.load(result_path + '/test_pred.npy', allow_pickle=True)
    trues = np.load(result_path + '/test_true.npy', allow_pickle=True)
    print(preds.shape); print(trues.shape)
    print(preds); print(trues)



if __name__ == '__main__':
    # legacy model
    m_144 = 'informer_WFP_ftS_sl144_ll144_pl16_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0'
    m_160_0 = 'informer_WFP_ftS_sl160_ll160_pl16_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0'
    m_160_1 = 'informer_WFP_ftS_sl160_ll160_pl16_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_1'
    m_160_2 = 'informer_WFP_ftS_sl160_ll160_pl16_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_2'
    m_160_3 = 'informer_WFP_ftS_sl160_ll160_pl16_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_3'
    m_160_4 = 'informer_WFP_ftS_sl160_ll160_pl16_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_4'
    m_160_00 = 'informer_WFP_ftS_sl160_ll160_pl16_dm512_nh8_el4_dl3_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0'
    m_160_96 = 'informer_WFP_ftS_sl160_ll96_pl16_dm512_nh16_el4_dl3_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0'

    # Updated model with new training data
    nm_160_0 = 'informer_WFP_ftS_sl160_ll160_pl16_dm512_nh8_el4_dl3_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0'
    nm_160_1 = 'informer_WFP_ftS_sl160_ll160_pl16_dm512_nh8_el4_dl3_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_1'
    nm_160_2 = 'informer_WFP_ftS_sl160_ll160_pl16_dm512_nh8_el4_dl3_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_2'

    ms_160_14P_0 = 'informer_WFP_ftMS_sl160_ll160_pl16_dm512_nh8_el4_dl3_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0'
    ms_160_14P_1 = 'informer_WFP_ftMS_sl160_ll160_pl16_dm512_nh8_el4_dl3_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_1'
    ms_160_14P_2 = 'informer_WFP_ftMS_sl160_ll160_pl16_dm512_nh8_el4_dl3_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_2'
    ms_160_14S_0 = 'informer_WFP_ftMS_sl160_ll160_pl16_dm512_nh8_el4_dl3_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_3'
    ms_160_14S_1 = 'informer_WFP_ftMS_sl160_ll160_pl16_dm512_nh8_el4_dl3_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_4'
    ms_160_14S_2 = 'informer_WFP_ftMS_sl160_ll160_pl16_dm512_nh8_el4_dl3_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_5'
    ms_160_27PS_0 = 'informer_WFP_ftMS_sl160_ll160_pl16_dm512_nh8_el4_dl3_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_6'

    wts_0 = 'informer_WTS_ftS_sl168_ll96_pl24_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0'
    wts_1 = 'informer_WTS_ftS_sl168_ll96_pl24_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_1'
    wts_2 = 'informer_WTS_ftS_sl168_ll96_pl24_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_2'
    # random_pred_plot(m1)
    # single_sliding_pred_plot(m1)
    # multiple_sliding_pred_plot(m2)
    # multiple_sliding_eval_plot(m2)
    # relative_sliding_pred_precision(m2, data='multiple')
    # sliding_pred_format_index_debug()
    # sliding_pred_output(m_160_00, save=True)
    # workbook_modified(m_160_0)
    # for m in [ms_160_14S_0, ms_160_14S_1, ms_160_14S_2, ms_160_27PS_0]:
        # sliding_pred_eval(m, plot='simple', save=True)
    # test_result_check(ms_160_14P_0)
    # batch_runner_formatter(precision, param, action='add')
