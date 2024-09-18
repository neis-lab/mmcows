# IMMU


import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import yaml, os
import json
import random
import argparse

from data_loader import data_loader_s2
from dnn_classifer import dnn_classifer
from dwt import process_dwt

# Function to segment data into 10-second windows with 50% overlap
def segment_data(data_df, window_size, overlap):
    data = data_df.drop(columns=['behavior'])
    labels = data_df[['timestamp', 'behavior']]
    mid_size = int(window_size/2)
    segments = []
    labels_list = []
    for i in range(0, len(data) - window_size, int(window_size * (1 - overlap))):
        segment = data.iloc[i:i + window_size, 1:].values  # Exclude the timestamp column
        mid_timestamp = int(data.iloc[i + mid_size, 0])
        class_id = int(labels[labels['timestamp'] == mid_timestamp]['behavior'].values[0])
        if class_id != 0:
            # if class_id != 7 or (class_id == 7 and np.random.rand() < 0.4):  # Randomly drop class 7
            # if True:
            segments.append(segment)
            labels_list.append(class_id)
    return np.array(segments), np.array(labels_list)


# Standardize each feature independently
def standardize_features(data):

    # Reshape the data to a 2D array
    num_samples, num_timesteps, num_features = data.shape
    data_reshaped = data.reshape(-1, num_features)

    # Apply StandardScaler to each feature independently
    scalers = []
    standardized_data = data_reshaped.copy()
    for i in range(num_features):
        scaler = StandardScaler()
        standardized_data[:, i] = scaler.fit_transform(data_reshaped[:, i].reshape(-1, 1)).flatten()
        scalers.append(scaler)

    # Reshape back to the original shape
    standardized_data = standardized_data.reshape(num_samples, num_timesteps, num_features)

    return standardized_data

def data_formater(data):
    window_size = 100  # 10 seconds * 10 Hz
    overlap = 0.5

    X, y = segment_data(data, window_size, overlap)
    y = y - 1

    X_train, y_train_n = X, y
    y_train = to_categorical(y_train_n, num_classes = n_classes)

    X_train = process_dwt(X_train, window_size)
    X_train = standardize_features(X_train)

    return X_train, y_train

# ===============================================
if __name__ == '__main__':

    print('\nMODALITY: IMMU (S2) --------------------------------------------')
    
    date = '0725'
    n_features = 4
    n_classes = 7

    behav_list = ['unknown', 'walking', 'standing', 'feed up', 'feed down', 'licking', 'drinking', 'lying']
    random.seed(42)

    train = False
    config = 's2'
    moda = 'immu_' + config
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

    group_1 = config['group_1']
    group_2 = config['group_2']
    folds = config['folds']

    aggregated_f1 = {}

    # Iterate over each fold and process the data
    for fold_name, fold_config in folds.items():
        print(f"\n{fold_name}:")

        pre_loader = 'immu_pre_data_loader'
        train_data, val_data, test_data = data_loader_s2(pre_loader, sensor_data_dir, group_1.copy(), group_2.copy(), fold_config, val = True, timestamp = True)
        # pre_loader = 'immu_pre_label_loader'
        # train_label, val_label, test_label = data_loader_s2(pre_loader, sensor_data_dir, group_1.copy(), group_2.copy(), fold_config, val = True, timestamp = True)

        # print(np.unique(train_data['behavior']))
        print('\tprocessing data ...')
        if train == True:
            X_train, y_train = data_formater(train_data)
            X_val, y_val = data_formater(val_data)
            X_test, y_test = data_formater(test_data)

            train_data = (X_train, y_train)
            val_data = (X_val, y_val)
            test_data = (X_test, y_test)

            print('\ttraining ...')
        else:
            X_test, y_test = data_formater(test_data)
            test_data = (X_test, y_test)
            train_data = test_data
            val_data = test_data

            print('\ttesting ...')

        acc_dict, prec_dict, recal_dict, f1_dict = dnn_classifer(moda, fold_name, train_data, val_data, test_data, num_classes = n_classes, train = train, verbose = 0)

        for key, value in f1_dict.items():
            if key not in aggregated_f1:
                aggregated_f1[key] = []
            aggregated_f1[key].append(value)
        
    print('\nRESULTS:', moda)

    # Calculate mean and standard deviation for each key
    mean_std_results = {key: (np.mean(values), np.std(values)) for key, values in aggregated_f1.items()}

    for key, (mean, std) in mean_std_results.items():
        print(f"\tClass {key} F1: {mean:.3f}±{std:.3f}  ({behav_list[int(key)]})")

    # Extract the mean F1 scores and standard deviations
    mean_f1_scores = [mean for mean, std in mean_std_results.values()]
    std_f1_scores = [std for mean, std in mean_std_results.values()]
    macro_average_f1 = np.mean(mean_f1_scores)
    average_error_f1 = np.mean(std_f1_scores)

    print(f"\tMacro avg F1: {macro_average_f1:.3f}±{average_error_f1:.3f}")


