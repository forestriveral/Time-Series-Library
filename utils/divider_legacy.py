import random
import numpy as np
import pandas as pd


def data_divider(data_df, seq_len, type_flag='simple'):
    np.random.seed(1234)
    if type_flag == 'simple':
        num_train = int(len(data_df)*0.7)
        num_test = int(len(data_df)*0.2)
        num_vali = len(data_df) - num_train - num_test
        start_index = [0, num_train - seq_len, len(data_df) - num_test - seq_len]
        end_index = [num_train, num_train + num_vali, len(data_df)]
    elif type_flag == 'extract':
        return data_point_generator(data_df)
    elif type_flag == 'random':
        start_index, end_index = random_seqs_generator(len(data_df), seq_len, gen_num=9)
    else:
        start_index, end_index = [0, 0, 0], [len(data_df), len(data_df), len(data_df)]
    # print('start_index: ', start_index, '\nend_index: ', end_index)
    return start_index, end_index


def random_seqs_generator(data_len, seq_len, gen_num=9):
    start = 0; end = data_len - seq_len
    start_index = divide_point_generate([], seq_len, start, end, gen_num)
    start_index.sort(); end_index = [border1 + seq_len for border1 in start_index]
    assert np.all(np.array(start_index[1:]) - np.array(end_index[:-1]) >= 0)
    return start_index, end_index


def divide_point_generate(start_index, seq_len, start, end, gen_num):
    if len(start_index) < gen_num:
        start_idx = np.random.randint(start, end)
        if start_idx + seq_len <= end:
            start_index.append(start_idx)
        if start_idx - start >= seq_len:
            divide_point_generate(start_index, seq_len, start, start_idx, gen_num)
        if end - start_idx - seq_len >= seq_len:
            divide_point_generate(start_index, seq_len, start_idx + seq_len, end, gen_num)
    # print('start_index: ', start_index)
    return start_index


def data_point_generator(data, type=1):
    time_steps = {'day': 4 * 24, 'week': 4 * 24 * 7,
                  'month': 4 * 24 * 30, 'year': 4 * 24 * 365}
    data['date'] = pd.to_datetime(data['date'])
    # data['day'] = data['date'].dt.date; data['week'] = data['date'].dt.isocalendar().week
    # data['month'] = data['date'].dt.month; data['year'] = data['date'].dt.year

    if type == 1:
        train_start = [i * 4 * time_steps['week'] for i in range(12)]
        train_end = [(3 + i * 4) * time_steps['week'] for i in range(12)]
        val_test_start = [(3 + i * 4) * time_steps['week'] for i in range(12)]
        val_test_end = [(i + 1) * 4 * time_steps['week'] for i in range(12)]
        train_start.append(val_test_end[-1]); train_end.append(data.shape[0])
        val_idx = random.sample(range(0, len(val_test_start)), 6); val_idx.sort()
        val_start = [val_test_start[i] for i in val_idx]; val_end = [val_test_end[i] for i in val_idx]
        test_start = [val_test_start[i] for i in range(len(val_test_start)) if i not in val_idx]
        test_end = [val_test_end[i] for i in range(len(val_test_end)) if i not in val_idx]
    elif type == 2:
        train_start =[0, 14592, 26304]
        train_end = [11712, 23424, 35136]
        val_start =[11712, ]
        val_end = [14592, ]
        test_start = [23424, ]
        test_end = [26304, ]
    elif type == 3:
        train_start =[0, ]
        train_end = [29376, ]
        val_start =[29376, ]
        val_end = [32256, ]
        test_start = [32256, ]
        test_end = [35136, ]
    # border1s = [train_start, val_start, test_start]
    # border2s = [train_end, val_end, test_end]
    border1s = [train_start, val_start, sorted(val_start + test_start)]
    border2s = [train_end, val_end, sorted(val_end + test_end)]

    return border1s, border2s

def data_reader(data, border1, border2):
    divided_data = pd.DataFrame() if isinstance(data, pd.DataFrame) else np.array([])
    for i, (start, end) in enumerate(zip(border1, border2)):
        if isinstance(data, pd.DataFrame):
            divided_data = pd.concat([divided_data, data[start:end]], axis=0)
        else:
            divided_data = np.concatenate([divided_data, data[start:end]], axis=0) \
                if i != 0 else data[start:end]
    # print(border1); print(border2); print(divided_data)
    return divided_data


def data_indexer(index, data_num, length):
    data_num = np.cumsum(data_num)
    for j in range(len(data_num)):
        if index < data_num[j]:
            s_begin = index + j * (length - 1)
            break
        else:
            continue
    return s_begin



if __name__ == '__main__':
    power_data = pd.read_csv('../data/WFP/Turbine_Patv_Total.csv')
    start_index, end_index = data_divider(power_data, 168, type_flag='extract')
    # index_list = divide_point_generate([], 10, 0, 500)
    # print(index_list)
    # divided_data = data_loader(power_data, start_index[2], end_index[2])
    length = 5; data_num = [11, 11, 11, 11]; s_begin_0 = []
    idx_array = np.arange(44); print(idx_array)
    for idx in range(44):
        s_begin_0.append(data_indexer(idx, data_num, length))
    print(s_begin_0); print(len(s_begin_0))

    idx_array = np.arange(44); np.random.shuffle(idx_array); s_begin_1 = []
    print(idx_array)
    for idx in idx_array:
        s_begin_1.append(data_indexer(idx, data_num, length))
    print(s_begin_1); print(len(s_begin_1))