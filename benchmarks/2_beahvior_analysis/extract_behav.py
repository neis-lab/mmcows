

import datetime as dt
import os

import numpy as np
import pandas as pd
import yaml

from avg_thi import get_avg_THI

def count_freq(raw_data, smoothed_data):
    assert len(np.unique(smoothed_data)) == 2, f'len(np.unique(values)) = {len(np.unique(smoothed_data))}'

    freq_list = []
    duration = 0
    prev_value = smoothed_data[0]

    for curr_raw, curr_smooth in zip( raw_data, smoothed_data):
        if curr_smooth == 1:
            if curr_raw == 1:
                duration += 15

        if curr_smooth == 0 and prev_value == 1:
            freq_list.append(duration/3600)
            duration = 0
        prev_value = curr_smooth

    return freq_list

def custom_binary_filter(data, window_size):
    if window_size < 2:
        raise ValueError("Window size must be at least 2")

    data = np.array(data)
    n = len(data)
    result = data.copy()

    for i in range(n - window_size + 1):
        window = data[i:i + window_size]
        indices = np.where(window == 1)[0]

        if len(indices) >= 2:
            for j in range(len(indices) - 1):
                start, end = indices[j], indices[j + 1]
                if end > start + 1:
                    result[i + start + 1:i + end] = 1

    return result


def extract_behav(behav_dir, date_list, id_list, window_size, behav_id):

    combined_data = []

    for cow_id in id_list:

        time_list = []
        freq_data_list = []
        mean_duration_list = []
        total_duration_list = []

        cow_name = f'C{cow_id:02d}' 
        tag_name = f'T{cow_id:02d}'

        # For each day
        for date in date_list:
            file_path = os.path.join(behav_dir, cow_name, cow_name + '_' + date + '.csv')

            behav_df = pd.read_csv(file_path) # skip the firt row, otherwise: header = True
            behav_df['behavior'] = behav_df['behavior'].replace(4, 3) 

            behav_df['behavior'] = behav_df['behavior'].replace(behav_id, 10) 

            behav_df['behavior'] = behav_df['behavior'].replace(1, 0) 
            behav_df['behavior'] = behav_df['behavior'].replace(2, 0) 
            behav_df['behavior'] = behav_df['behavior'].replace(3, 0) 
            behav_df['behavior'] = behav_df['behavior'].replace(4, 0) 
            behav_df['behavior'] = behav_df['behavior'].replace(5, 0) 
            behav_df['behavior'] = behav_df['behavior'].replace(6, 0) 
            behav_df['behavior'] = behav_df['behavior'].replace(7, 0) 

            behav_df['behavior'] = behav_df['behavior'].replace(10, 1)

            data = behav_df.to_numpy()

            timestamps = data[:,0]
            raw_data = data[:,2]

            smoothed_data = custom_binary_filter(raw_data, window_size)

            freq_list = count_freq(raw_data, smoothed_data)

            datet = dt.datetime.fromtimestamp(timestamps[0]).date()
            # print(datet)

            time_list.append(datet)
            freq_data_list.append(len(freq_list))
            mean_duration_list.append(np.mean(freq_list))
            total_duration_list.append(np.sum(raw_data)*15/3600)
        
        cow_data = {}
        cow_data['cow_id'] = cow_id
        cow_data['cow_name'] = cow_name
        cow_data['date'] = time_list
        cow_data['freq_data'] = freq_data_list
        cow_data['mean_duration'] = mean_duration_list
        cow_data['total_duration'] = total_duration_list
        cow_data['raw_data'] = raw_data
        cow_data['smoothed_data'] = smoothed_data
        cow_data['timestamps'] = timestamps

        combined_data.append(cow_data)

    return combined_data

# ===============================================
if __name__ == '__main__':

    import matplotlib.dates as md
    import matplotlib.pyplot as plt
    
    date_list = ['0725']

    id_list = range(1,11)

    current_dir = os.path.join(os.path.dirname(__file__))  # Folder
    yaml_dir = os.path.join(current_dir, 'private', "path.yaml")

    with open(yaml_dir, 'r') as file:
        file_dirs = yaml.safe_load(file)
    sensor_data_dir = file_dirs['sensor_data_dir']

    input_dir = os.path.join(sensor_data_dir, 'main_data', 'thi', 'average.csv')
    THI_timestamps, daily_THI = get_avg_THI(input_dir)

    behav_id = 3 # feeding
    window_size = 50 # 50: feeding
    combined_dict = extract_behav(os.path.join(current_dir, 'pred_behav_data'), date_list, id_list, window_size, behav_id)

    for cow_data_dict in combined_dict:

        fig, ax1 = plt.subplots(figsize=(6, 4))

        date = cow_data_dict['date']
        behavior = cow_data_dict['raw_data']
        smoothed_data = cow_data_dict['smoothed_data']

        plt.plot(behavior, label = 'raw_data', color='tab:blue')
        plt.plot(smoothed_data, label = 'smoothed_data', color='tab:orange')
        plt.title(cow_data_dict['cow_name'])
        plt.legend()

    print("\nDone\n")
    plt.show()