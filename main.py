import os
import random
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
from utils.tools import ModelConfig, DotDict, param_list_converter, setting_formatter


def model_runner(config: str | Path | Dict, seed=None, report=False) -> None:
    args = argparse.Namespace(**ModelConfig(config))
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    args.use_filter = DotDict(args.use_filter)
    args.use_hybrid = DotDict(args.use_hybrid)

    fix_seed = 2024
    if seed is not None:
        fix_seed = seed
        args.run_seed = seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # check the target, subcol and use_hybrid target input
    # convert all of them to list
    args.target, args.subcol, args.use_hybrid.target = \
        param_list_converter([args.target, args.subcol, args.use_hybrid.target])

    # check whether the input dimension of encoder and decoder are matched or not
    subcol_len = len(args.subcol) if args.subcol is not None else len([])
    if args.features == 'S':
        args.enc_in, args.dec_in, args.c_out = 1, 1, 1
    elif args.features == 'M':
        args.enc_in, args.dec_in, args.c_out = \
            len(args.target) + subcol_len, len(args.target) + subcol_len, len(args.target)
    elif args.features == 'MS':
        args.enc_in, args.dec_in, args.c_out = \
            len(args.target) + subcol_len, len(args.target) + subcol_len, 1
    else:
        raise ValueError('Illegal input feature type!')

    # check the target of use_hybrid is empty or not
    # if empty, reset the flag of use_hybrid to False
    if args.use_hybrid.flag:
        if args.use_hybrid.target is None:
            args.use_hybrid.flag = False

    print('Args in experiment:')
    print_args(args)

    task_short_names = {'LTF': 'long_term_forecast',
                        'STF': 'short_term_forecast',
                        'IMP': 'imputation',
                        'AD': 'anomaly_detection',
                        'CLS': 'classification'}

    args.task_name = task_short_names.get(args.task_name.upper(), None) or args.task_name
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
        args.short_task_name = 'LTF'
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
        args.short_task_name = 'STF'
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
        args.short_task_name = 'IMP'
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
        args.short_task_name = 'AD'
    elif args.task_name == 'classification':
        Exp = Exp_Classification
        args.short_task_name = 'CLS'
    else:
        Exp = Exp_Long_Term_Forecast
        args.short_task_name = 'LTF'

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = setting_formatter(args, ii)

            exp = Exp(args)  # set experiments

            if args.is_logging:
                exp.logger(setting)  # set training logger for recording print info

            print('\n>>>>>>> start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('\n>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()

            print('\n>>>>>>> evaluating : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.eval(setting, report=report)

            print('\n>>>>>>> training finish : {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    else:
        setting = setting_formatter(args)

        exp = Exp(args)  # set experiments

        if args.is_logging:
                exp.logger(setting, 'testing')  # set training logger for recording print info

        print('\n>>>>>>> testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.test(setting, test=1)
        torch.cuda.empty_cache()

        print('\n>>>>>>> evaluating : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.eval(setting, report=report)

        print('\n>>>>>>> testing finish : {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))


def model_loader(case_name, mode='test', report=False) -> None:
    case_name = 'LTF_Spd6_cfd_wrf_Informer_turbine_ftM_ti5_uf0_uh1_sl384_ll96_pl96_dm512_nh8_el3_dl2_df2048_fc1_ebtimeF_dtTrue_test_0'

    # load the config file in the case folder
    config_path = Path('results') / case_name / 'config.yaml'
    config_dict = ModelConfig.load_yaml(Path(config_path).resolve())
    print(f'Loading config file from {config_path}')

    # set the mode of model loader
    config_dict.is_training = 0 if mode == 'test' else 1
    print(f'Setting model to {mode.upper()} mode')

    model_runner(config_dict, seed=2024, report=report)


if __name__ == '__main__':
    # config = 'configs/power_config.yaml'
    # config = 'configs/speed_config.yaml'
    # config = 'configs/multiple_power_config.yaml'
    config = 'configs/multiple_speed_config.yaml'

    model_runner(config, report=False)