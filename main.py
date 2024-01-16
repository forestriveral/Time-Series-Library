import os
import copy
import random
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Dict, Union, List, Tuple

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
from utils.tools import ModelConfig, DotDict, param_list_converter, \
    setting_formatter, finetune_config_generator


# define the type of config
ConfigType = str | Path | Dict | argparse.Namespace


def model_runner(
    config: ConfigType,
    seed: Optional[int] = None,
    report: bool = False,
    ) -> None:

    if not isinstance(config, argparse.Namespace):
        args = argparse.Namespace(**ModelConfig(config))
    else:
        args = config

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
        setting = setting_formatter(args, 0)

        exp = Exp(args)  # set experiments

        if args.is_logging:
                exp.logger(setting, 'testing')  # set training logger for recording print info

        print('\n>>>>>>> testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.test(setting, test=1)
        torch.cuda.empty_cache()

        print('\n>>>>>>> evaluating : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.eval(setting, report=False)

        print('\n>>>>>>> testing finish : {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))


def model_runner_from_case(
    case_path: Optional[str | Path]  = None,
    mode: str = 'test',
    seed: Optional[int] = None,
    report: bool = False,
    ) -> None:
    if case_path is None:
        case_path = './results/LTF_Wspd_cfd_wrf_Informer_turbine_ftM_ti5_uf0_uh1_sl384_ll96_pl96_dm512_nh8_el3_dl2_df2048_fc1_ebtimeF_dtTrue_test_0'

    # load the config file in the case folder
    config_path = f'{case_path}/config.yaml'
    config = argparse.Namespace(**ModelConfig(config_path))
    print(f'Loading config file from {config_path}')

    # set the mode of model loader
    if mode == 'test':
        config.is_training = 0
    elif mode == 'train':
        config.is_training = 1
    else:
        raise ValueError('Invalid mode input')
    print(f'Setting model to {mode.upper()} mode')

    model_runner(config, seed=seed, report=report)


def model_runner_for_cross_val(
    config: ConfigType,
    mode: str = 'train',
    fname: Optional[str] = None,
    seed: Optional[int] = None,
    report: bool = False,
    ) -> None:
    config = argparse.Namespace(**ModelConfig(config))

    VAL_TEST_IDX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    DEFAULT_FOLDER_NAME = fname or 'cross_val'

    # set the mode of model loader
    if mode == 'test':
        config.is_training = 0
    elif mode == 'train':
        config.is_training = 1
    else:
        raise ValueError('Invalid mode input')
    print(f'Setting model to {mode.upper()} mode')

    # set the folder name of results and checkpoints
    config.res_path = f'./results/{DEFAULT_FOLDER_NAME}/'
    config.res_path = f'./checkpoints/{DEFAULT_FOLDER_NAME}/'

    for test_idx in VAL_TEST_IDX:
        config_idx = copy.deepcopy(config)
        config_idx.test_idx = test_idx
        config_idx.model_id = f'cro_val_{test_idx}_{config.model_id}'
        print(f'\n*************** Cross validation test index: {test_idx} ***************')

        # model runner training and testing
        model_runner(config_idx, seed=seed, report=report)

        print(f'*************** Cross validation test index: {test_idx} finished! ***************\n')

    print('All cross validation finished!')

    return None


def model_runner_for_models(
    config: ConfigType,
    seed: Optional[int] = None,
    report: bool = True,
    ) -> None:
    config = argparse.Namespace(**ModelConfig(config))

    MODEL_TEST_NAMES = [
        # 'Autoformer',
        'Transformer',
        'Nonstationary_Transformer',
        'DLinear',
        'FEDformer',
        'Informer',
        'LightTS',
        'Reformer',
        'ETSformer',
        'PatchTST',
        'Pyraformer',
        'MICN',
        'Crossformer',
        'FiLM',
        'iTransformer',
        'Koopa',
        'TiDE',
        'FreTS',
    ]

    for model_i in MODEL_TEST_NAMES:
        config_idx = copy.deepcopy(config)
        config_idx.is_training = 1
        config_idx.model = model_i

        print(f'\n*************** Model Testing: {model_i} ***************')

        try:
            # model runner training and testing
            model_runner(config_idx, seed=seed, report=report)
        except:
            print(f'*************** Model Testing: {model_i} failed! ***************\n')
            continue

        print(f'*************** Model Testing: {model_i} finished! ***************\n')

    print('All model testings finished!')


def model_runner_for_finetuning(
    config: str | Path,
    hyper_config: str | Path,
    report: bool = False,
    ) -> None:

    hyper_configs = ModelConfig.load_yaml(Path(hyper_config).resolve())
    base_config = ModelConfig.load_yaml(Path(config).resolve())

    hyper_config_list = finetune_config_generator(hyper_configs)
    print('Testing model number: ', len(hyper_config_list))

    hyper_des_suffix = 'hyper'
    for hyper_i, hyper_config_i in enumerate(hyper_config_list):
        base_config_i = copy.deepcopy(base_config)
        base_config_i.update(hyper_config_i)
        base_config.des = f'{hyper_des_suffix}_{hyper_i}'

        random_seed = np.random.randint(1000, 10000)

        model_runner(base_config_i, seed=random_seed, report=report)

    print('All models have been tested!')

    return None



if __name__ == '__main__':
    # config = 'configs/power_config.yaml'
    # config = 'configs/speed_config.yaml'
    # config = 'configs/multiple_power_config.yaml'
    config = 'configs/multiple_speed_config.yaml'

    hyper_config = 'configs/hyper_config.yaml'

    model_runner(config, report=True)

    # model_runner_for_models(config, report=True)

    # model_runner_for_cross_val(config, fname='cross_val_wrf', report=True)

    # model_runner_from_case()

    # model_runner_for_finetuning(config, hyper_config)