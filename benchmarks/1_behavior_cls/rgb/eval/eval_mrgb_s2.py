
import os

import numpy as np
import pandas as pd
import argparse

from datetime import datetime

from utils.projection import cal_cam_coord
from utils.read_proj_mat import read_projection_matrices
from mrgb_proc import mrgb_proc
from utils.cmb_eval import cmb_eval

import pytz
import yaml
import json

# Set a global time zone: Central Time
CT_time_zone = pytz.timezone('America/Chicago')

def search_files(folder_path, search_text, file_format=".csv"):
    file_names = []
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        # print(file_name)
        # Check if the file is a text file
        if file_name.endswith(file_format):
            # Check if the search text is present in the file name
            if search_text in file_name:
                file_names.append(file_name)
    return sorted(file_names)

def select_range(timestamps, point1, point2):
    assert point1 != point2, f'Invalid range {point2} = {point1}'
    # print(f'{point1} vs {point2}')
    
    if point1 > point2:
        selected = [value for value in timestamps if point2 <= value < point1]
    else:
        selected = [value for value in timestamps if point1 <= value < point2]
    return selected


# ===============================================
if __name__ == '__main__':

    print('\nMODALITY: M-RGB (S2) --------------------------------------------')
    
    date = '0725'
    behav_list = ['unknown', 'walking', 'standing', 'feed up', 'feed down', 'licking', 'drinking', 'lying']
    
    config = 's2'
    moda = 'M-RGB ' + config
    config_name = 'config_' + config + '.json'

    current_dir = os.path.join(os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--path_dir', type=str, default=os.path.join(current_dir, 'private', "path.yaml"))
    parser.add_argument('--config_dir', type=str, default=os.path.join(current_dir, 'private', config_name))   
    parser.add_argument('--lying', action='store_false', help='N/A')
    args = parser.parse_args() 
    
    yaml_dir = args.path_dir
    json_dir = args.config_dir
    lying = args.lying

    if lying == True:
        print('With lying cows')
    else:
        print('Without lying cows')

    with open(yaml_dir, 'r') as file:
        file_dirs = yaml.safe_load(file)
    sensor_data_dir = file_dirs['sensor_data_dir']

    with open(json_dir, 'r') as f:
        config = json.load(f)
    
    sensor_data_dir = file_dirs['sensor_data_dir']
    visual_data_dir = file_dirs['visual_data_dir']
    pred_label_dir = file_dirs['pred_label_dir']
    behav_dir = os.path.join(sensor_data_dir, 'behavior_labels', 'individual')
    # pred_label_dir = os.path.join(visual_data_dir, 'output_labels')
    proj_mat_dir = os.path.join(visual_data_dir, 'proj_mat')

    # print("proj mat: ", proj_mat_dir)

    Proj_cam_list = read_projection_matrices(proj_mat_dir, date)

    cam_coord = []
    for idx in range(4):
        proj_mat = Proj_cam_list[idx]
        cam_coord.append(cal_cam_coord(proj_mat))
        # print(f"Cam {idx+1} loc: {cal_cam_coord(proj_mat).astype(int)}")
    cam_coord = np.asarray(cam_coord).reshape((4,3))

    # Read behavior labels
    behav_gt_list = []
    for cow_id in range(1,17):
        csv_file = f'C{cow_id:02d}_{date}.csv' 
        behav_df = pd.read_csv(os.path.join(behav_dir, csv_file))
        assert np.shape(behav_df)[1] == 3, f'Wrong csv'
        behav_gt_list.append(behav_df)
    assert len(behav_gt_list) == 16, f'Missing behavior ground truth'

    # Extract timestamps from txt labels 
    temp_list = []
    for i in range(1,5):
        cam_name = f"cam_{i:d}"
        folder_path = os.path.join(pred_label_dir, date, cam_name)
        # print(folder_path)
        filename_list = search_files(folder_path, search_text='_', file_format=".txt")
        for single_filename in filename_list:
            temp_list.append(int(single_filename[0:10]))
    temp_list = np.asarray(temp_list)
    combined_timestamps = list(np.unique(temp_list))
    combined_timestamps.sort() # must sort here
    print('\nTotal # of timestamps:', str(len(combined_timestamps)))
    # print(combined_timestamps)


    group_1 = config['group_1']
    group_2 = config['group_2']
    folds = config['folds']

    id_list = range(1,17)

    # Convert datetime ranges to Unix timestamps
    def datetime_to_unix(dt_str):
        dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        dt = CT_time_zone.localize(dt)  # Localize the datetime object to the specified timezone
        # print(datetime.fromtimestamp(int(dt.timestamp()), CT_time_zone))
        return int(dt.timestamp())

    # Convert the time ranges to Unix timestamps
    for group in [group_1, group_2]:
        for key, value in group.items():
            group[key] = [datetime_to_unix(value[0]), datetime_to_unix(value[1])]

    # aggregated_acc = {}
    # aggregated_prec = {}
    # aggregated_recal = {}
    aggregated_f1 = {}

    # Iterate over each fold and process the data
    for fold_name, fold_config in folds.items():
        print(f"\n{fold_name}:")
            
        chunk_name = fold_config['test']
        print('  chunk', int(chunk_name[0]))
        timestamp_chunk1 = select_range(combined_timestamps, int(group_1[chunk_name[0]][0]), int(group_1[chunk_name[0]][1]))
        timestamp_chunk2 = select_range(combined_timestamps, int(group_2[chunk_name[0]][0]), int(group_2[chunk_name[0]][1]))
        selected_timestamps = list(timestamp_chunk1) + list(timestamp_chunk2)
        print('  # of timestamps:', len(selected_timestamps))

        y_pred, y_test = mrgb_proc(id_list, selected_timestamps, behav_gt_list, cam_coord, Proj_cam_list, pred_label_dir, lying = lying)
        acc_dict, prec_dict, recal_dict, f1_dict = cmb_eval(y_pred, y_test)

        # for key, value in acc_dict.items():
        #     if key not in aggregated_acc:
        #         aggregated_acc[key] = []
        #     aggregated_acc[key].append(value)

        # for key, value in prec_dict.items():
        #     if key not in aggregated_prec:
        #         aggregated_prec[key] = []
        #     aggregated_prec[key].append(value)

        # for key, value in recal_dict.items():
        #     if key not in aggregated_recal:
        #         aggregated_recal[key] = []
        #     aggregated_recal[key].append(value)

        for key, value in f1_dict.items():
            if key not in aggregated_f1:
                aggregated_f1[key] = []
            aggregated_f1[key].append(value)
            print(f'\t{key}: {value:.3f}')
    
    if 0 in aggregated_f1:
        del aggregated_f1[0]

    # Calculate mean and standard deviation for each key
    mean_std_results = {key: (np.mean(values), np.std(values)) for key, values in aggregated_f1.items()}

    print('\nRESULTS:', moda)
    for key, (mean, std) in mean_std_results.items():
        print(f"\tClass {key} F1: {mean:.3f}±{std:.3f}  ({behav_list[int(key)]})")

    # Extract the mean F1 scores and standard deviations
    mean_f1_scores = [mean for mean, std in mean_std_results.values()]
    std_f1_scores = [std for mean, std in mean_std_results.values()]
    macro_average_f1 = np.mean(mean_f1_scores)
    average_error_f1 = np.mean(std_f1_scores)

    print(f"\tMacro avg F1: {macro_average_f1:.3f}±{average_error_f1:.3f}")

    


    
        
