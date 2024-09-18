
import datetime as dt
import os

# import matplotlib.dates as md
import matplotlib.pyplot as plt
# from matplotlib import cm
import numpy as np
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from matplotlib.animation import FuncAnimation

import time
from datetime import datetime
import math

from utils.plot_player import Player 
from utils.gd_uwb_loc import gd_localization, triagulation
from utils.pen_model import *

import warnings
import yaml

# Ignore UserWarning from Matplotlib
warnings.filterwarnings("ignore", category=UserWarning)

# Set NumPy print options to suppress scientific notation
np.set_printoptions(suppress=True, formatter={'float_kind':'{:.2f}'.format})


cam_coord = np.asarray([[-1189,   541,   383],     # cam 1
                      [1191,  584,  356],     # cam 2
                      [1179, -647,  379],     # cam 3
                      [-1186,  -656,   383]])    # cam 4


def update(i):
    data_point = list_data_points[i]

    print(' ')
    for i in range(8):
        print(f'{int(data_point[i,0])}\t{int(data_point[i,1])}\t{data_point[i,2]:.2f}\t{int(data_point[i,3])}\t{int(data_point[i,4])}\t{int(data_point[i,5])}')

    time_data = dt.datetime.fromtimestamp(int(data_point[0, 0]))
    ax.clear()
    ax.set_title(tag_name + ' ' + str(time_data))

    ## Localization
    prev_location = np.array([0, 0, 0]).astype(float) # first initial location
    curr_location, loss_value, n_iterations, selected_anchor_ids = localizer.gd_localization(prev_location, data_point)
    # prev_location = curr_location # do not use

    if np.isnan(n_iterations) == False:

        cow_loc = curr_location

        # The cow
        ax.scatter(cow_loc[0], cow_loc[1], cow_loc[2], marker='o', c='r', s=50, label='cow')

        anchor_list = selected_anchor_ids
        print(f'{time_data}  {anchor_list}  z:{curr_location[2]}  L:{loss_value:.1f}  iter:{n_iterations}')

        ## Triangulation lines
        for idx in anchor_list:
            ax.plot([cow_loc[0], Anchors[idx][0]], [cow_loc[1], Anchors[idx][1]], [cow_loc[2], Anchors[idx][2]], color='orange', alpha=1)  # line

    draw_pen(ax, cam_coord, legend=False)

    ax.legend()
    # plt.show()

# ===============================================
""" Main program from here """
if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
    project_dir = os.path.dirname(current_dir)   # Get the parent directory (one level up)


    tag_id = 3
    date = "0729"

    # current_dir = os.path.dirname(current_dir)   # Get the parent directory (one level up)        
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
    yaml_dir = os.path.join(current_dir, "path.yaml")
    
    with open(yaml_dir, 'r') as file:
        file_dirs = yaml.safe_load(file)
    sensor_data_dir = file_dirs['sensor_data_dir']
    visual_data_dir = file_dirs['visual_data_dir'] 
    image_dir = os.path.join(visual_data_dir, 'images')
    label_dir = os.path.join(visual_data_dir, 'labels', 'combined')
    proj_mat_dir = os.path.join(visual_data_dir, 'proj_mat')


    # cow_data_dir = os.path.join(dataset_dir, "cow_data", "combined_data")
    # proj_data_dir = os.path.join(dataset_dir, 'visual_data', 'cam_cal_mat')

    tag_name = f'T{tag_id:02d}' 
    print('\nTag: ' + tag_name)

    # Get UWB distance data

    input_path = os.path.join(sensor_data_dir, 'sub_data','uwb_distance', tag_name , tag_name + '_' + date + ".csv")
    uwb_data = pd.read_csv(input_path).to_numpy()

    timestamps = uwb_data[:,0]
    n_row = len(timestamps)

    idx = 0
    list_data_points = [] 
    while idx < n_row:
        data_point = uwb_data[idx: idx + 8, :]

        ##  Check if all eight datapoints have the same timestamp
        timestamp_point = data_point[:, 0].astype(int)
        if  np.all(timestamp_point == timestamp_point[0]) == False:
            print("wrong timestamp: " + str(uwb_data[idx:idx+16, 0]))
            break
        list_data_points.append(data_point)
        idx += 8

    localizer = gd_localization()
    # localizer = triagulation()

    # Initialize a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # animation = FuncAnimation(fig, update, frames=len(df), repeat=False, interval=1)
    animation = Player(fig, update, maxi=len(list_data_points)-1)

    # plt.tight_layout()

    # Display the animation (you can save it to a file as well)
    plt.show()

    print("\nDone\n")

