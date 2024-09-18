
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
    train = False

    current_dir = os.path.join(os.path.dirname(__file__))  # Folder
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--path_dir', type=str, default=os.path.join(current_dir, 'private', "path.yaml")) 
    args = parser.parse_args() 
    
    yaml_dir = args.path_dir

    with open(yaml_dir, 'r') as file:
        file_dirs = yaml.safe_load(file)
    sensor_data_dir = file_dirs['sensor_data_dir']
    # pred_behav_dir = file_dirs['pred_behav_dir']

    pred_behav_dir = os.path.join(current_dir, 'pred_behav_data')


    id_list = range(1,11)
    date = '0725'

    train_data = uwb_hd_aa_pre_loader(sensor_data_dir, id_list, date)
    train_data = train_data.drop(columns=['timestamp'])
    test_data = train_data

    # Check the sizes of the datasets
    # print(f"  Train data size:", train_data.shape)

    # print("   Training")
    clf, acc_dict, prec_dict, recal_dict, f1_dict = rf_classifer(moda, fold_name, train, train_data, test_data)

    date_list = ['0722','0723','0724','0725','0726','0727','0728','0729','0730','0731','0801','0802','0803']

    print("   Inferencing")
    for date in date_list:
        for cow_id in id_list:
            cow_name = f'C{cow_id:02d}'

            uwb_df = uwb_hd_aa_pre_loader(sensor_data_dir, [cow_id], date, label = False)
            # print(np.shape(uwb_df))

            full_timestamps = uwb_df.values[:,0].astype(int)

            nonzero_timestamps = uwb_df.dropna().values[:,0]
            uwb_loc = uwb_df.dropna().values[:,1:]

            X_test = uwb_loc
            y_pred = clf.predict(X_test)
            
            datetime_df = pd.to_datetime(full_timestamps - 5*3600, unit='s').strftime('%H:%M:%S')
            data_df = pd.DataFrame({'timestamp':nonzero_timestamps, 'behavior':y_pred})
            timestamps_df = pd.DataFrame({'timestamp':full_timestamps, 'datetime':datetime_df})
            merged_df = pd.merge(timestamps_df, data_df, on='timestamp', how='inner')
            # print(merged_df.head)

            output_dir = os.path.join(pred_behav_dir, cow_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            csv_output_file = cow_name + '_' + date + '.csv'
            output_dir = os.path.join(output_dir, csv_output_file)
            merged_df.to_csv(output_dir, index=False)
            print(f"{csv_output_file} saved")


