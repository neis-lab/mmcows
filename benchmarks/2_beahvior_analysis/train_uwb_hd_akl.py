
# UWB+HD+AA


import numpy as np
import os
import yaml
import argparse
import pandas as pd

from data_loader import uwb_hd_aa_pre_loader
from rf_classifer import rf_classifer


# ===============================================
if __name__ == '__main__':

    print('\nUWB+HD+AA: Behavior classification on 14-day data for cow #1 to #10 --------')

    moda = 'uwb_hd_aa'
    fold_name = 'fold_0'
    train = True

    current_dir = os.path.join(os.path.dirname(__file__))  # Folder
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--path_dir', type=str, default=os.path.join(current_dir, 'private', "path.yaml")) 
    args = parser.parse_args() 
    
    yaml_dir = args.path_dir

    with open(yaml_dir, 'r') as file:
        file_dirs = yaml.safe_load(file)
    sensor_data_dir = file_dirs['sensor_data_dir']

    id_list = range(1,11)
    date = '0725'

    train_data = uwb_hd_aa_pre_loader(sensor_data_dir, id_list, date)
    train_data = train_data.drop(columns=['timestamp'])
    test_data = train_data

    # Check the sizes of the datasets
    print(f"  Train data size:", train_data.shape)
    print("   Training")
    clf, acc_dict, prec_dict, recal_dict, f1_dict = rf_classifer(moda, fold_name, train, train_data, test_data)



