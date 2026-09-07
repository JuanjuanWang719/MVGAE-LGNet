import os
import numpy as np
import argparse
from lib.config_io import read_config
from lib.utils import compute_len_input


def search_data(sequence_length, num_of_batches, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''
    # 每小时采集的点数不能小于0
    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    # 标签开始位置+预测数  不能超出数据长度
    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    # 选择多长时间（周、天、小时）作为训练数据
    for i in range(1, num_of_batches + 1):
        # 从当前标签点往前选择指定窗口大小的数据
        start_idx = label_start_idx - points_per_hour * units * i
        # 从开始位置往后选择预测的数据数量
        end_idx = start_idx + num_for_predict  # ？？？？？？？？
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None
    # 判断可选择的数据量是否满足指定的窗口大小
    if len(x_idx) != num_of_batches:
        return None

    return x_idx[::-1]   # 从最后一个元素到第一个元素，每次往前走一步


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点
    num_for_predict: int,
                     the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None
    # 如果当前数据序号+预测数  大于  数据长度，说明数据量不够，无法生成完整一组可用数据，直接完成
    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    # 按照周选择数据的开始和结束位置
    if num_of_weeks > 0:  # 如果按照星期预测
        # 得到满足条件的start-end对
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None
        # 根据给定的 week_indices 从 data_sequence 中提取出指定的子序列，然后将它们按行连接起来，最终生成一个新的数组 week_sample。
        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)
    # 按照天选择数据的开始和结束位置
    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)
    # 按照小时选择数据的开始和结束位置
    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def read_and_generate_dataset(graph_signal_matrix_filename,
                                                     num_of_weeks, num_of_days,
                                                     num_of_hours, num_for_predict,
                                                     points_per_hour=12, save=False):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_depend * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    # 原始序列 (T, N, F)；如 PEMS04 (16992,307,*)，PEMS07 (28224,883,*)
    data_seq = np.load(graph_signal_matrix_filename)['data']  # (T, N, F)
    print('data_seq shape:', data_seq.shape)
    all_samples = []
    for idx in range(data_seq.shape[0]): # 循环每条数据
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue

        week_sample, day_sample, hour_sample, target = sample

        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)

        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(hour_sample)

        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        sample.append(target)

        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)

        all_samples.append(sample)  # sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]

    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T')
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)

    train_target = training_set[-2]  # (B,N,T)
    val_target = validation_set[-2]
    test_target = testing_set[-2]

    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]

    (stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)

    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'timestamp': test_timestamp,
        },
        'stats': {
            '_mean': stats['_mean'],
            '_std': stats['_std'],
        }
    }
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)
    print('train timestamp:', all_data['train']['timestamp'].shape)
    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)
    print('val timestamp:', all_data['val']['timestamp'].shape)
    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)
    print('test timestamp:', all_data['test']['timestamp'].shape)
    print()
    print('train data _mean :', stats['_mean'].shape, stats['_mean'])
    print('train data _std :', stats['_std'].shape, stats['_std'])

    if save:
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        dirpath = os.path.dirname(graph_signal_matrix_filename)
        filename = os.path.join(dirpath, file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '_astcgn'
        print('save file:', filename)
        np.savez_compressed(filename,
                            train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                            train_timestamp=all_data['train']['timestamp'],
                            val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                            val_timestamp=all_data['val']['timestamp'],
                            test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                            test_timestamp=all_data['test']['timestamp'],
                            mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
                            )
    return all_data


def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    mean = train.mean(axis=(0,1,3), keepdims=True)
    std = train.std(axis=(0,1,3), keepdims=True)
    print('mean.shape:',mean.shape)
    print('std.shape:',std.shape)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm


# prepare dataset
parser = argparse.ArgumentParser()
# 设置配置文件路径和名称
parser.add_argument("--config", default='configurations/PEMS04_multi_period.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
print('Read configuration file: %s' % (args.config))
config = read_config(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']  # 邻接矩阵路径csv
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']  # 数据集路径npz
# ？？？？？？？？？？？？？？
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])   # 节点数量
points_per_hour = int(data_config['points_per_hour'])   # 每小时采集时间点次数，如12次，即每5分钟次
num_for_predict = int(data_config['num_for_predict'])   # 预测数据量点，如12次，即预测1个小时的数据
len_input = int(data_config['len_input'])  # ？？？？？
dataset_name = data_config['dataset_name']  # 数据集名称
num_of_weeks = int(training_config['num_of_weeks'])  # 按周预测
num_of_days = int(training_config['num_of_days'])  # 按天预测
num_of_hours = int(training_config['num_of_hours'])  #按小时预测

expected_len = compute_len_input(num_of_weeks, num_of_days, num_of_hours, num_for_predict)
if len_input != expected_len:
    raise ValueError(
        f"len_input={len_input} 与多周期配置不一致，应为 "
        f"(num_of_weeks + num_of_days + num_of_hours) * num_for_predict = {expected_len}"
    )

print(
    f"multi-period input: weeks={num_of_weeks}, days={num_of_days}, hours={num_of_hours}, "
    f"T={expected_len}"
)

_raw = np.load(graph_signal_matrix_filename)['data']
if _raw.ndim < 2:
    raise ValueError(f"graph signal 维度异常: {_raw.shape}")
if int(_raw.shape[1]) != num_of_vertices:
    raise ValueError(
        f"num_of_vertices={num_of_vertices} 与数据不一致：npz['data'] 节点维={_raw.shape[1]} "
        f"(file={graph_signal_matrix_filename})"
    )
print(f"raw graph signal: shape={_raw.shape}, num_of_vertices={num_of_vertices}")

all_data = read_and_generate_dataset(
    graph_signal_matrix_filename,
    num_of_weeks,
    num_of_days,
    num_of_hours,
    num_for_predict,
    points_per_hour=points_per_hour,
    save=True,
)
