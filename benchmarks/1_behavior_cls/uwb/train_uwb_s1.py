# # UWB


import numpy as np
import os
import yaml
import json
import argparse

from data_loader import data_loader_s1
from rf_classifer import rf_classifer

# ===============================================
if __name__ == '__main__':
    print('\nMODALITY: UWB (S1) --------------------------------------------')
    behav_list = ['unknown', 'walking', 'standing', 'feed up', 'feed down', 'licking', 'drinking', 'lying']

    train = True
    config = 's1'
    moda = 'uwb_' + config
    config_name = 'config_' + config + '.json'

    current_dir = os.path.join(os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--path_dir', type=str, default=os.path.join(current_dir, 'private', "path.yaml"))
    parser.add_argument('--config_dir', type=str, default=os.path.join(current_dir, 'private', config_name))   
    args = parser.parse_args() 
    
    yaml_dir = args.path_dir
    json_dir = args.config_dir

    with open(yaml_dir, 'r') as file:
        file_dirs = yaml.safe_load(file)
    sensor_data_dir = file_dirs['sensor_data_dir']

    with open(json_dir, 'r') as f:
        config = json.load(f)

    folds = config['folds']

    aggregated_f1 = {}

    # Iterate over each fold and process the data
    for fold_name, fold_config in folds.items():
        print(f"\n{fold_name}:")
        pre_loader = 'uwb_pre_loader'
        train_data, val_data, test_data = data_loader_s1(pre_loader, sensor_data_dir, fold_config, val = False)
        print(f"  [train/valid/test] = {[train_data.shape[0], val_data.shape[0], test_data.shape[0]]}")

        clf, acc_dict, prec_dict, recal_dict, f1_dict = rf_classifer(moda, fold_name, train, train_data, test_data)

        for key, value in f1_dict.items():
            if key not in aggregated_f1:
                aggregated_f1[key] = []
            aggregated_f1[key].append(value)

    print('\nRESULTS:', moda)

    # Calculate mean and standard deviation for each key
    mean_std_results = {key: (np.mean(values), np.std(values)) for key, values in aggregated_f1.items()}

    for key, (mean, std) in mean_std_results.items():
        print(f"Class {key} F1: {mean:.3f}±{std:.3f}  ({behav_list[int(key)]})")

    # Extract the mean F1 scores and standard deviations
    mean_f1_scores = [mean for mean, std in mean_std_results.values()]
    std_f1_scores = [std for mean, std in mean_std_results.values()]
    macro_average_f1 = np.mean(mean_f1_scores)
    average_error_f1 = np.mean(std_f1_scores)

    print(f"Macro avg F1: {macro_average_f1:.3f}±{average_error_f1:.3f}")


