import numpy as np
import pandas as pd

from typing import List, Tuple, Optional

# extract two months data from the training datset
def dataset_divider(
    data_len: int,
    data_idx: int,
    verbose: bool = False,
    ) -> Tuple[List[List[int]], List[List[int]]]:
    split_part_num = 12
    val_part_num = 2
    test_part_num = 1
    data_idx = data_idx - 1
    # assert (data_idx <= split_part_num - val_part_num - test_part_num) and (data_idx >= 0)
    assert (data_idx <= 11) and (data_idx >= 0)

    # split the dataset into 12 parts
    split_points = np.linspace(0, data_len, split_part_num + 1, endpoint=True, dtype=int)

    if data_idx <= 1:
        extract_points = split_points[data_idx:data_idx + val_part_num + test_part_num + 1]
        val_data_borders = [extract_points[1], extract_points[-1]]
        test_data_borders = [extract_points[0], extract_points[1]]
    else:
        extract_points = split_points[data_idx - val_part_num:data_idx + 2]
        val_data_borders = [extract_points[0], extract_points[val_part_num]]
        test_data_borders = [extract_points[val_part_num], extract_points[-1]]
    # print('split_points: ', split_points)
    # print('extract_points: ', extract_points)
    # print('val_data_borders: ', val_data_borders)
    # print('test_data_borders: ', test_data_borders)

    if data_idx == 0:
        train_data_borders = [[extract_points[-1], split_points[-1]]]
    elif data_idx == 1:
        train_data_borders = [[split_points[0], extract_points[0]],
                              [extract_points[-1], split_points[-1]]]
    elif data_idx == 2:
        train_data_borders = [[extract_points[-1], split_points[-1]]]
    elif data_idx == 11:
        train_data_borders = [[split_points[0], extract_points[0]]]
    else:
        train_data_borders = [[split_points[0], extract_points[0]],
                              [extract_points[-1], split_points[-1]]]
    # print('train_data_borders: ', train_data_borders)

    border1s = [[b1 for b1, _ in train_data_borders], [val_data_borders[0]], [test_data_borders[0]]]
    border2s = [[b2 for _, b2 in train_data_borders], [val_data_borders[1]], [test_data_borders[1]]]

    if verbose:
        print('border1s: ', border1s)
        print('border2s: ', border2s)

    return border1s, border2s


def dataset_reader(
    data: pd.DataFrame | np.ndarray,
    border1s: List[int] | int,
    border2s: List[int] | int,
    ) -> pd.DataFrame | np.ndarray:
    assert isinstance(border1s, (list, int)) and isinstance(border2s, (list, int))

    if isinstance(border1s, int):
        border1s = [border1s]
    if isinstance(border2s, int):
        border2s = [border2s]

    if isinstance(data, pd.DataFrame):
        combined_data = pd.DataFrame(columns=data.columns)
        for _, (start, end) in enumerate(zip(border1s, border2s)):
            combined_data = pd.concat([combined_data, data[start:end]], axis=0)
    elif isinstance(data, np.ndarray):
        combined_data = np.array([[]])
        for i, (start, end) in enumerate(zip(border1s, border2s)):
            combined_data = np.concatenate([combined_data, data[start:end]], axis=0) \
                if i != 0 else data[start:end]
    else:
        raise ValueError('data type error')

    # print('combined_data: ', combined_data.shape)
    return combined_data


def split_calculator(
    border1s: List[List[int]],
    border2s: List[List[int]],
    ) -> List[List[int]]:
    assert isinstance(border1s, list) and isinstance(border2s, list)
    assert len(border1s) == len(border2s) == 3

    split_num = []
    for b1, b2 in zip(border1s, border2s):
        num = np.array(b2) - np.array(b1)
        split_num.append(num.tolist())

    # print('split_num: ', split_num)
    return split_num


def dataset_indexer(
    index: int,
    split_num: List[int],
    seq_pred_len: int,
    ) -> int:
    split_num = np.cumsum(np.array(split_num) - seq_pred_len + 1)
    for j in range(len(split_num)):
        if index < split_num[j]:
            s_begin = index + j * (seq_pred_len - 1)
            break
        else:
            continue
    return s_begin


if __name__ == '__main__':
    for i in range(1, 13):
        border1s, border2s = dataset_divider(35136, i, verbose=False)
    # df_raw = pd.read_csv('datasets/WFP/Farm_Patv_15min.csv')[['date', 'Patv_Total']]
    # combined_data = dataset_reader(df_raw.values, border1s[0], border2s[0])
    # split_calculator(border1s, border2s)