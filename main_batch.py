import os
import random
import torch
import argparse
import numpy as np
from pathlib import Path
from collections import OrderedDict

from main import model_runner
from utils.tools import ModelConfig, finetune_config_generator


def batch_model_runner(
    config: str | Path,
    hyper_config: str | Path,
    report=False,
    log=False
    ) -> None:

    hyper_config = ModelConfig.load_yaml(Path(hyper_config).resolve())
    base_config = ModelConfig.load_yaml(Path(config).resolve())

    hyper_config_list = finetune_config_generator(hyper_config)
    print('Testing model number: ', len(hyper_config_list))

    hyper_des_suffix = 'hyper'
    for hyper_i, hyper_config in enumerate(hyper_config_list):
        base_config.update(hyper_config)
        random_seed = np.random.randint(1000, 10000, size=1)
        base_config.des = f'{hyper_des_suffix}_{hyper_i}'

        model_runner(base_config, seed=random_seed, report=True)

    print('All models have been tested!')

    return None



if __name__ == '__main__':
    # config = 'configs/power_config.yaml'
    # config = 'configs/speed_config.yaml'
    # config = 'configs/multiple_power_config.yaml'
    config = 'configs/multiple_speed_config.yaml'

    hyper_config = 'configs/hyper_config.yaml'

    batch_model_runner(config, hyper_config)