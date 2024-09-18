import os
import pytz
from datetime import datetime
import sys
import numpy as np
import pandas as pd


def uwb_hd_aa_pre_loader(sensor_data_dir, id_list, date = '0725', label = True):
    combined_df = pd.DataFrame()

    for cow_id in id_list:

        tag_name = f'T{cow_id:02d}'
        cow_name = f'C{cow_id:02d}'

        uwb_dir = os.path.join(sensor_data_dir, 'main_data', 'uwb', tag_name, tag_name + '_' + date + '.csv') # 1/15 Hz
        head_dir = os.path.join(sensor_data_dir, 'sub_data', 'head_direction', tag_name, tag_name + '_' + date + '.csv') # 10 Hz
        ankle_dir = os.path.join(sensor_data_dir, 'main_data', 'ankle', cow_name, cow_name + '_' + date + '.csv') # 10 Hz
        if label == True:
            label_dir = os.path.join(sensor_data_dir, 'behavior_labels', 'individual', cow_name + '_' + date + '.csv') # 1 Hz

        uwb_df = pd.read_csv(uwb_dir)
        head_df = pd.read_csv(head_dir)
        ankle_df = pd.read_csv(ankle_dir)
        if label == True:
            label_df = pd.read_csv(label_dir)

        # 10 Hz to 1 Hz
        head_df['timestamp'] = np.floor(head_df['timestamp']).astype(int)  # Round down to the nearest second
        head_df = head_df.groupby('timestamp').mean() # 1 Hz
        head_df.reset_index(inplace=True) # Required

        # 1 Hz to 1/15 Hz
        uwb_timestamps = uwb_df['timestamp'].values
        head_df = head_df[head_df['timestamp'].isin(uwb_timestamps)].astype('float64') # 1/15 Hz
        if label == True:
            label_df = label_df[label_df['timestamp'].isin(uwb_timestamps)] # 1/15 Hz

        ankle_df = ankle_df.drop(columns=['datetime']).astype('float64')
        head_df = pd.merge_asof(head_df, ankle_df, on='timestamp', direction='nearest') # ankle_dir 1/15 Hz, ankle_df 1 Hz

        if label == True:
            # Remove 0 labels
            label_df = label_df[label_df['behavior'] != 0].copy() # Remove unknown behavior
            label_timestamps = label_df.values[:,0] # Without 0s

            # Replace all occurrences of 1 with 2 in the 'behavior' column
            # label_df['behavior'] = label_df['behavior'].replace(1, 2)  

            # 1/15 Hz data without 0 labels
            uwb_df = uwb_df[uwb_df['timestamp'].isin(label_timestamps)] # 1/15 Hz
            head_df = head_df[head_df['timestamp'].isin(label_timestamps)] # 1/15 Hz

        X_df = pd.merge(uwb_df, head_df, on='timestamp')
        if label == True:
            data_df = pd.merge(X_df, label_df, on='timestamp')
            data_df = data_df.drop(columns=['datetime'])
        else:
            data_df = X_df

        data_df = data_df.dropna()
        data_df = data_df.drop(columns=['accel_norm','relative_angle'])

        combined_df = pd.concat([combined_df, data_df], ignore_index=True)
    return combined_df

# Object-wide split
def data_loader_s1(pre_loader, sensor_data_dir, fold_config, val = False, timestamp = False):
    date = '0725'
    print(pre_loader)

    # Retrieve the sub-function by its name
    sub_func = getattr(sys.modules[__name__], pre_loader)

    train_ids = [int(x) for x in fold_config['train']] 
    val_ids = [int(x) for x in fold_config['val']] 
    test_ids = [int(x) for x in fold_config['test']] 

    train_data = sub_func(sensor_data_dir, train_ids, date)

    if val == True:
        val_data = sub_func(sensor_data_dir, val_ids, date)
    else: # val_data is empty if val = True
        val_data = pd.DataFrame()
        train_data = pd.concat([train_data, sub_func(sensor_data_dir, val_ids, date)])

    test_data = sub_func(sensor_data_dir, test_ids, date)

    if timestamp == False:
        try:
            train_data = train_data.drop(columns=['timestamp'])
            test_data = test_data.drop(columns=['timestamp'])
            val_data = val_data.drop(columns=['timestamp'])
        except:
            pass

    return train_data, val_data, test_data


# Temporal split
def data_loader_s2(pre_loader, sensor_data_dir, group_1, group_2, fold, val = False, timestamp = False):

    id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # id_list = [6]
    date = '0725'
    print(pre_loader)

    # Retrieve the sub-function by its name
    sub_func = getattr(sys.modules[__name__], pre_loader)
    df = sub_func(sensor_data_dir, id_list, date)

    # Set a global time zone: Central Time
    CT_time_zone = pytz.timezone('America/Chicago')

    # Convert datetime ranges to Unix timestamps
    def datetime_to_unix(dt_str):
        dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        dt = CT_time_zone.localize(dt)  # Localize the datetime object to the specified timezone
        return int(dt.timestamp())

    # Convert the time ranges to Unix timestamps
    for group in [group_1, group_2]:
        for key, value in group.items():
            group[key] = [datetime_to_unix(value[0]), datetime_to_unix(value[1])]
            
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()

    def append_data(ranges, group, data):
        for key in ranges:
            start, end = group[key]
            data = pd.concat([data, df[(df['timestamp'] >= start) & (df['timestamp'] < end)]])
        return data

    train_data = append_data(fold['train'], group_1, train_data)
    train_data = append_data(fold['train'], group_2, train_data)

    if val == True:
        val_data = append_data(fold['val'], group_1, val_data)
        val_data = append_data(fold['val'], group_2, val_data)
    else: # val_data is empty if val = True
        train_data = append_data(fold['val'], group_1, train_data)
        train_data = append_data(fold['val'], group_2, train_data)

    test_data = append_data(fold['test'], group_1, test_data)
    test_data = append_data(fold['test'], group_2, test_data)

    if timestamp == False:
        try:
            train_data = train_data.drop(columns=['timestamp'])
            test_data = test_data.drop(columns=['timestamp'])
            val_data = val_data.drop(columns=['timestamp'])
        except:
            pass

    return train_data, val_data, test_data