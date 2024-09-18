# IMMU


import numpy as np
# import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
# import matplotlib.pyplot as plt
import yaml, os
from sklearn.model_selection import train_test_split

# from sklearn.utils.class_weight import compute_class_weight
import random
import argparse

from dnn_classifer import dnn_classifer
from data_loader import immu_pre_data_loader
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

    X_train, y_train = X, y
    # y_train = to_categorical(y_train_n, num_classes = n_classes)

    X_train = process_dwt(X_train, window_size)
    X_train = standardize_features(X_train)

    return X_train, y_train

# ===============================================
if __name__ == '__main__':

    print('\nMODALITY: IMMU (S0), single cow, random split for reference ------')

    moda = 'immu_s0'
    fold_name = 'fold_0'
    train = False

    # id_list = range(1,11)
    id_list = [7]
    random.seed(42)

    behav_list = ['unknown', 'walking', 'standing', 'feed up', 'feed down', 'licking', 'drinking', 'lying']

    current_dir = os.path.join(os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--path_dir', type=str, default=os.path.join(current_dir, 'private', "path.yaml")) 
    args = parser.parse_args() 
    
    yaml_dir = args.path_dir


    with open(yaml_dir, 'r') as file:
        file_dirs = yaml.safe_load(file)
    sensor_data_dir = file_dirs['sensor_data_dir']


    date = '0725'
    n_features = 4
    n_classes = 7

    print('\tprocessing data ...')

    combined_data_df = immu_pre_data_loader(sensor_data_dir, id_list, date)
    # print(np.unique(combined_data_df['behavior']))


    X, y = data_formater(combined_data_df)
    # print(np.unique(y))


    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6,  stratify = y, shuffle = True, random_state=2) 
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5,  stratify = y_test, shuffle = True, random_state=2) 

    # print(np.unique(y_train))
    # print(np.unique(y_val))
    # print(np.unique(y_test))

    # X_train = process_dwt(X_train, window_size)
    # X_train = standardize_features(X_train)

    # X_train = standardize_features(X_train)
    # X_val = standardize_features(X_val)
    # X_test = standardize_features(X_test)

    y_train = to_categorical(y_train, num_classes = n_classes)
    y_val = to_categorical(y_val, num_classes = n_classes)
    y_test = to_categorical(y_test, num_classes = n_classes)


    train_data = (X_train, y_train)
    val_data = (X_val, y_val)
    test_data = (X_test, y_test)


    # Compute class weights
    # y_train_n = np.argmax(y_train, axis=1)
    # class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_n), y=y_train_n)

    # Convert class weights to a dictionary
    # class_weight_dict = dict(enumerate(class_weights))

    # Manual dict
    # class_weight_dict = {
    #     0: 10.0,  # Class 0
    #     1: 1.0,   # Class 1
    #     2: 1.5,   # Class 2
    #     3: 2.0,   # Class 3
    #     4: 15.0,  # Class 4
    #     5: 7.5,   # Class 5
    #     6: 0.2    # Class 6
    # }

    if train == True:
        print('\ttraining ...')
    else:
        print('\ttesting ...')
    acc_dict, prec_dict, recal_dict, f1_dict = dnn_classifer(moda, fold_name, train_data, val_data, test_data, num_classes = n_classes, train = train, verbose = 0)


    behav_list = ['unknown', 'walking', 'standing', 'feed up', 'feed down', 'licking', 'drinking', 'lying']

    print('\nRESULTS:', moda)
    for key, value in f1_dict.items():
        print(f"\tClass {key} F1: {value:.3f}  ({behav_list[int(key)]})")

    # Extract the mean F1 scores and standard deviations
    mean_f1_scores = [mean for mean in f1_dict.values()]
    macro_average_f1 = np.mean(mean_f1_scores)

    print(f"\tMacro avg F1: {macro_average_f1:.3f}")


