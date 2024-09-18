import os
import pytz
from datetime import datetime
import sys
import numpy as np
import pandas as pd

def immu_pre_data_loader(sensor_data_dir, id_list, date = '0725'):
    combined_data_df = pd.DataFrame()

    for cow_id in id_list:
        cow_name = f'C{cow_id:02d}'
        tag_name = f'T{cow_id:02d}'
        
        accel_data = pd.read_csv(os.path.join(sensor_data_dir,'main_data','immu', tag_name, f'{tag_name}_{date}.csv'))
        accel_data = accel_data[['timestamp', 'accel_x_mps2', 'accel_y_mps2', 'accel_z_mps2']]
        # accel_data = accel_data.drop(columns=['pressure_Pa'])

        head_data = pd.read_csv(os.path.join(sensor_data_dir, 'sub_data', 'head_direction', tag_name, f'T{cow_id:02d}_{date}.csv'))
        accel_data = pd.merge(accel_data, head_data[['timestamp', 'relative_angle']], on='timestamp', how='inner')
        # accel_data = pd.merge(accel_data, head_data, on='timestamp', how='inner')
        # accel_data = head_data

        label_df = pd.read_csv(os.path.join(sensor_data_dir, 'behavior_labels', 'individual', f'C{cow_id:02d}_{date}.csv'))
        label_df = label_df[['timestamp', 'behavior']]

        # Replace all occurrences of 1 with 2 in the 'behavior' column
        # label_df['behavior'] = label_df['behavior'].replace(1, 2).astype(np.float64)  
        label_df['timestamp'] = label_df['timestamp'].astype(np.float64)  

        merged_df = pd.merge_asof(accel_data, label_df, on='timestamp', direction='nearest') # accel 10 Hz, label 1 Hz

        # Fill NaN values with preceding valid values
        merged_df.fillna(method='ffill', inplace=True)

        combined_data_df = pd.concat([combined_data_df, merged_df], ignore_index=True)
    return combined_data_df

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
    # id_list = [3]
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