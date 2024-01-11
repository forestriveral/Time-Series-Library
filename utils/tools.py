import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import yaml
import copy
from pathlib import Path
from scipy import signal
from scipy import interpolate

from typing import Any, List, Dict, Callable, Literal, Optional

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


class logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        # with open(filename, 'r+') as file: file.truncate(0)
        self.log = open(filename, "a")

    def write(self, message):
        self.log.write(message)
        self.terminal.write(message)
        self.log.flush()

    def flush(self):
        pass


class Loader(yaml.SafeLoader):
    def __init__(self, stream: str | Path) -> None:
        self._root = os.path.split(stream.name)[0]
        super().__init__(stream)

    def include(self, node: yaml.Node) -> Dict[Any, Any]:
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            return yaml.load(f, self.__class__)


Loader.add_constructor('!include', Loader.include)


class DotDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
            if isinstance(value, dict):
                value = DotDict(value)
            return value
        except:
            return super().__getattribute__(key)

    def __deepcopy__(self, memo: Any, _nil: List[Any] = []) -> Dict[Any, Any] | List[Any]:
        if memo is None:
            memo = {}
        d = id(self)
        y = memo.get(d, _nil)
        if y is not _nil:
            return y

        dict = DotDict()
        memo[d] = id(dict)
        for key in self.keys():
            dict.__setattr__(copy.deepcopy(key, memo),
                             copy.deepcopy(self.__getattr__(key), memo))
        return dict


class ModelConfig(DotDict):
    def __init__(self, input_config: str | Path | Dict) -> None:
        if isinstance(input_config, str) or isinstance(input_config, Path):
            input_config = ModelConfig.load_yaml(Path(input_config).resolve())
        super().__init__(input_config)

    @classmethod
    def load_yaml(cls, filename: str, loader: yaml.SafeLoader = Loader) -> Dict[Any, Any]:
        with open(filename) as fid:
            return yaml.load(fid, loader)

    def to_yaml(self, output_file_path: str | Path) -> None:
        with open(output_file_path, "w+") as output_file:
            yaml.dump(
                dict(self),
                output_file,
                sort_keys=False,
                default_flow_style=False
                )


def data_filter(sample, n=5, cut_off=0.15):
    b1, a1 = signal.butter(n, cut_off, 'lowpass')
    b2, a2 = signal.butter(n, cut_off, 'highpass')
    return signal.filtfilt(b1, a1, sample), signal.filtfilt(b2, a2, sample)


def df_data_filter(df_data, n, cut_off):
    highpass_data = {}
    for col in df_data.columns:
        lowpass, highpass = data_filter(
            df_data[col].values, n, cut_off)
        df_data[col] = lowpass
        highpass_data[col] = highpass
    return df_data, highpass_data


def noise_statistics(noise):
    if isinstance(noise, torch.Tensor):
        noise = noise.cpu().numpy()
    elif isinstance(noise, list):
        noise = np.array(noise)

    # noise = noise.reshape(-1)
    noise_mean = np.mean(noise, axis=1)
    noise_std = np.std(noise, axis=1)
    return noise_mean, noise_std


def config_format(config, path):
    if hasattr(config, 'short_task_name'):
        config.task_name = config.short_task_name
        # config.pop('short_task_name')
    if isinstance(config.use_filter, DotDict) or isinstance(config.use_hybrid, DotDict):
        config.use_filter = dict(config.use_filter)
        config.use_hybrid = dict(config.use_hybrid)
    return ModelConfig(dict(vars(config))).to_yaml(path + 'config.yaml')


def train_loss_plot(train_recorder, record_path):
    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(np.arange(len(train_recorder['train_loss'])),
            train_recorder['train_loss'], 'k-', label='train loss', lw=2.)
    ax.plot(np.arange(len(train_recorder['train_loss'])),
            train_recorder['val_loss'], 'b-', label='val loss', lw=2.)
    ax.plot(np.arange(len(train_recorder['train_loss'])),
            train_recorder['test_loss'], 'g-', label='test loss', lw=2.)
    plt.legend(loc='best')
    plt.savefig(record_path + '/training_history.png')
    plt.close()


def turbine_curve_loader(wt, param):
    wts = ['320', '265']; params = ['power', 'thrust', 'C_p', 'C_t']
    power_curve = pd.read_csv('datasets/WFP/Turbine_Power_Curve.csv', index_col=None, header=0)
    # print(power_curve.shape)
    # print(power_curve)

    if isinstance(wt, int):
        wt = str(wt)

    assert wt in wts, 'Invalid turbine type'
    assert param in params, 'Invalid parameter'

    col_name = '_'.join([param, wt])
    interp_func = interpolate.interp1d(
        power_curve['speed'].values,
        power_curve[col_name].values * 1e-3,
        kind='slinear',
        fill_value='extrapolate')

    print(f'Turbine {wt} {param} curve loading ...')

    return interp_func

#  Check whether the shape and datetime columns of raw and hybrid data are matched
def hybrid_data_check(raw, hybrid):
    if not (raw.shape[0] == hybrid.shape[0]):
        raise ValueError('Shape of raw and hybrid data not matched')
    if not (pd.to_datetime(raw['date']).equals(pd.to_datetime(hybrid['date']))):
        raise ValueError('Datetime of raw and hybrid data not matched')
    return True


def speed_power_converter(args, debug=False):
    DEFAULT_POWER_BASELINE = 'datasets\WFP\Turbine_Patv_Spd_15min_filled.csv'
    DEFAULT_POWER_INDEX = ['320'] * 10 + ['265'] * 3

    pred_type, pred_idx = args.target.split('_')

    data_type = 'power'
    pow_func = lambda x: x
    baseline_data = None
    pow_capacity = float(DEFAULT_POWER_INDEX[int(pred_idx) - 1]) / 100.

    if (not debug) and (pred_type == 'Wspd') and (pred_idx in [str(i) for i in range(1, 14)]):
        pow_func = turbine_curve_loader(DEFAULT_POWER_INDEX[int(pred_idx) - 1], 'power')

        # load the baseline data and extract the power baseline
        pow_baseline = pd.read_csv(DEFAULT_POWER_BASELINE, index_col=None, header=0)
        pow_baseline = pow_baseline[['date', f'Patv_{pred_idx}']]

        def baseline_data(date, data=pow_baseline):
            assert isinstance(date, np.ndarray), 'Input date should be a numpy array'
            # load the baseline data and extract the power baseline of the specific date range
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')
            data = data.loc[pd.to_datetime(date.flatten())]
            data = data.reset_index()
            print('Baseline data extracting ...')
            return data.iloc[:, 1].values.reshape(date.shape)

    elif (debug) and (pred_type == 'Wspd') and (pred_idx in [str(i) for i in range(1, 14)]):
        pow_capacity = 20.
        data_type = 'speed'
    elif (pred_type == 'Patv') and (pred_idx == 'Total'):
        pow_capacity = 39.95

    return DotDict(flag=data_type, func=pow_func, baseline=baseline_data, capacity=pow_capacity)
