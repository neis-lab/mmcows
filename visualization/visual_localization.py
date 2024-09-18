
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

from datetime import datetime
import sys
import csv
import datetime as dt

from utils.plot_player import Player 
from utils.pen_model import draw_pen
from utils.projection import cal_cam_coord, project_image2world
from utils.draw_bbox import read_bbox_labels, ratio_to_pixel
from utils.handle_xml import read_projection_matrices
from utils.line_geometry import cal_line_equation, cal_dist_point_to_line
from utils.AdaGrad_visual_loc import visual_localization

import pytz
import yaml
from tqdm import tqdm

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

def create_folder(folder_dir):
    # Check if the folder already exists
    if not os.path.exists(folder_dir):
        # Create the folder if it doesn't exist
        os.makedirs(folder_dir)

# def progress_bar(iterable, length=50):
#     total = len(iterable)
#     progress = 0
#     for i, item in enumerate(iterable, 1):
#         progress = i / total
#         bar = "[" + "=" * int(length * progress) + " " * (length - int(length * progress)) + "]"
#         sys.stdout.write(f"\r{bar} {int(progress * 100)}%  {i:d} ")
#         sys.stdout.flush()
#         yield item

def find_closest_row(scalar, array):
    first_column = array[:, 0]
    
    differences = np.abs(first_column - scalar) # Calculate the absolute differences
    min_index = np.argmin(differences) # Find the index of the minimum difference
    closest_row = array[min_index, :] # Return the corresponding row
    
    return closest_row

def interpolate_missing_row(array_1D, array_2D):
    # Find missing values in the first column
    missing_values = np.setdiff1d(array_1D, array_2D[:, 0])

    row_length = len(array_2D[0,:])

    # Fill rows with missing values with -1
    for missing_value in missing_values:
        filled_row = -np.ones(row_length)
        filled_row[0] = missing_value
        array_2D = np.vstack([array_2D, filled_row])

    # Sort the 2D array based on the first column
    array_2D = array_2D[array_2D[:, 0].argsort()]
    return array_2D

def get_row_by_unix_timestamp(arr, unix_timestamp):
    row_index = np.where(arr[:, 0] == unix_timestamp)[0]
    if len(row_index) == 0:
        return []  # Timestamp not found in the array
    else:
        return arr[row_index[0]]
    
def update(i):
    curr_timestamp = int(combined_timestamps[i])
    datetime_var = datetime.fromtimestamp(curr_timestamp, CT_time_zone)
    image_name = f'{curr_timestamp:d}_{datetime_var.hour:02d}-{datetime_var.minute:02d}-{datetime_var.second:02d}.jpg'
    curr_datetime = f'{datetime_var.hour:02d}-{datetime_var.minute:02d}-{datetime_var.second:02d}'
    date = f'{datetime_var.month:02d}{datetime_var.day:02d}' 

    global single_view, behav_dict_list
    global Proj_cam_list, label_dir, cam_coord, n_lying, n_nonlying
    global frame_height, frame_width, image_scale, no_print

    resolution = (frame_width, frame_height)

    # uwb_locs = np.empty((0, 3))
    # resting_refs = []
    
    # for cow_data in cow_data_list:
    #     # timestamp       = cow_data[i,0]
    #     uwb_loc         = cow_data[i,1:4] / 100
    #     rest_ref    = cow_data[i,4] 

    #     uwb_locs = np.concatenate((uwb_locs, uwb_loc.reshape(1,3)), axis=0) # in cm
    #     resting_refs.append(rest_ref)

    if no_print == False:
        print('------')

    ## uwb_locs_w_rest: [uwb_id] [xyz] [rest_ref]
    # uwb_locs = np.concatenate((np.asarray(range(uwb_locs.shape[0])).reshape((-1,1))+1, uwb_locs), axis=1)
    # uwb_locs_w_rest = np.concatenate((uwb_locs, np.asarray(resting_refs).reshape((-1,1))), axis=1)
        
    ## Go through each camera
    bbox_dict_list = []
    for cam_idx, cam_name in zip(range(4), cam_list):
        cam_view_dict = {}
        proj_mat = Proj_cam_list[cam_idx]
        cam_view_dict['cam_idx'] = cam_idx
        n_rays = 0

        text_file_name = image_name[-25:-4]
        timestamp = text_file_name[0:10]
        file_dir = os.path.join(label_dir, date, cam_name, text_file_name + '.txt')

        ## Dict structure:
        #  cam_view_dict {'cam_idx', 'n_rays', 'list_dict'}
        #    +--list_dict [data_point1, data_point2, ...]
        #         +---data_point {'cow_id', 'bbox', 'line_eq', 'pixel_loc'}

        dummy_data_point = {'cow_id':-1, 'bbox':np.zeros(6), 'line_eq':np.zeros(6)}
        dummy_cam_view_dict = {'cam_idx':cam_idx, 'n_rays':0, 'list_dict':[dummy_data_point]}

        try:
        # if True:
            bboxes_data = read_bbox_labels(file_dir)
            
            if len(bboxes_data.flatten()) > 0:
                # bboxes_data[:,0] = np.ones(bboxes_data.shape[0])*(0) # change all bbox ids to 0
                n_rays = bboxes_data.shape[0]
                cam_view_dict['n_rays'] = n_rays
                # projected_lines = []
                list_dict = []
                for idx, row in enumerate(bboxes_data):
                    data_point = {}
                    data_point['cow_id'] = row[0]
                    # print(row[0])

                    bbox_xyxyn = row[1:5]
                    bboxes_data[idx, 1:3] = ratio_to_pixel(resolution, bbox_xyxyn[0:2])
                    # bboxes_data[idx, 3:5] = ratio_to_pixel(resolution, bbox_xyxyn[2:4]) # Width, height
                    data_point['bbox_loc'] = bboxes_data[idx, 1:5]
                    data_point['pixel_loc'] = bboxes_data[idx, 1:3]

                    # Projecting center of bbox to the 3D ground
                    point2 = project_image2world(bboxes_data[idx, 1:3], proj_mat, 30)
                    line_eq = cal_line_equation(cam_coord[cam_idx], point2)
                    data_point['line_eq'] = line_eq

                    list_dict.append(data_point)

                ## bbox_data_w_proj_lines: [idx: 0] [xyxy] [yx] [conf] [ABC DEF line]
                cam_view_dict['list_dict'] = list_dict
            else:
                # print('Warning: no bbox in the frame')
                cam_view_dict = dummy_cam_view_dict
        except:
            cam_view_dict = dummy_cam_view_dict
            # print("label not found: " + file_dir)

        bbox_dict_list.append(cam_view_dict)

    # for my_dict in bbox_dict_list:
    #     print(my_dict)
    #     # for key, value in my_dict.items():
    #     #     print(key, ':', value)
    #     print('---')
    
    ## Create the dict to store the data for each cow
    all_cows_line_set = []
    for i in range(16):
        single_cow_line_set = {'cow_id':i+1, 
                               'line_list': [], 
                               'cam_idx_list': [], 
                               'timestamp':curr_timestamp, 
                               'location':np.empty((3)) * np.nan, 
                               'pixel_list':[]
                               }
        all_cows_line_set.append(single_cow_line_set)

    for cam_view_dict in bbox_dict_list:
        # print(cam_view_dict)
        for single_cow_dict in all_cows_line_set:
            for single_bbox_dict in cam_view_dict['list_dict']:
                # print(f"{single_cow_dict['cow_id']} vs {single_bbox_dict['cow_id']}")
                if single_cow_dict['cow_id'] == single_bbox_dict['cow_id']:
                    single_cow_dict['line_list'].append(single_bbox_dict['line_eq'])
                    single_cow_dict['cam_idx_list'].append(cam_view_dict['cam_idx'])
                    single_cow_dict['pixel_list'].append(single_bbox_dict['pixel_loc'])
                    # print('found')
            
            # print(single_cow_dict)
    
    # print(curr_datetime)

    for single_cow_dict in all_cows_line_set:
        if len(single_cow_dict['line_list']) > 0:
            line_eqs = np.asarray(single_cow_dict['line_list'])

            # Multi-view localization
            if line_eqs.shape[0] > 1:
                ## Localization using gradient descent   
                # print(line_eqs.astype(int))
                nearest_point, total_distance, iter, gradient = visual_localization(line_eqs)
                nearest_point = nearest_point.astype(int)
                single_cow_dict['location'] = nearest_point
                if no_print == False:
                    print(f"{single_cow_dict['cow_id']:2d}  {nearest_point}\td:{int(total_distance)/100:.2f}\t#{iter}\tg:{gradient:.2f}")

                ## Find outliners
                # Find distance from nearest_point to the lines
                for i, line_eq in enumerate(line_eqs):
                    dist = int(cal_dist_point_to_line(line_eq, nearest_point))
                    # print(dist)
                    if dist > 120 or i > 3:
                        cam_id = single_cow_dict['cam_idx_list'][i] + 1
                        print(f"==> Outlier: {curr_datetime} cow {single_cow_dict['cow_id']}, cam_{cam_id} ({line_eqs.shape[0]} cams)")

            # Single-view localization
            elif line_eqs.shape[0] == 1:
                if single_view == True:
                    for behav_dict in behav_dict_list:
                        if behav_dict['cow_id'] == single_cow_dict['cow_id']:
                            row = get_row_by_unix_timestamp(behav_dict['behav'], curr_timestamp).astype(int)
                            # print(single_cow_dict['cow_id'], row)
                            behav = row[1] # timestamp, behav
                    if behav == 7: # lying
                        Z_set = 55
                        n_lying += 1
                        # print(n_lying)
                    else:
                        Z_set = 80
                        n_nonlying += 1
                        # print(n_nonlying)

                    cam_idx = int(single_cow_dict['cam_idx_list'][0])
                    p_mat = Proj_cam_list[cam_idx]
                    image_coord = single_cow_dict['pixel_list'][0]
                    nearest_point = project_image2world(image_coord, p_mat, Z=Z_set) # standing cows
                    nearest_point = nearest_point.astype(int)
                    single_cow_dict['location'] = nearest_point
                    
                    if no_print == False:
                        print(f"{single_cow_dict['cow_id']:2d}  {nearest_point}\tbehav:{behav}")
            

    ## Plotting ========================================================

    ax1.clear()
    ax1.set_title(str(datetime_var)[0:19] + f"  {int(curr_timestamp):d}")

    for single_cow_dict in all_cows_line_set:
        cow_id = single_cow_dict['cow_id']
        est_cow_loc = single_cow_dict['location']/100

        if cow_id < 11:
        # if True:
        
            # Plot points and titles
            if np.isnan(est_cow_loc[0]) == False:
                # print(f"{cow_id} {est_cow_loc}")
                ax1.scatter(est_cow_loc[0], est_cow_loc[1], est_cow_loc[2], marker='o', c=colors[cow_id], s=30) # point
                ax1.text(est_cow_loc[0], est_cow_loc[1], est_cow_loc[2] + 0.3, f'{cow_id}', fontsize=13, color=colors[cow_id], ha='center', va='bottom') # title

            # Plot lines
            line_eqs = single_cow_dict['line_list']
            if len(line_eqs) > 0 and cow_id != 0:
                for line in line_eqs:
                    plot_line_3d(ax1, line, colors[int(cow_id)], alpha=0.35)
        
        # # Plot uwb title
        # for row in uwb_locs_w_rest:
        #     cow_id = int(row[0])
        #     cow_loc = row[1:4]
        #     rest_ref = row[4]
        #     if np.isnan(cow_loc[0]) == False and rest_ref == 0:
        #         ax1.text(cow_loc[0], cow_loc[1], cow_loc[2] + 0.3, f'{cow_id}', fontsize=13, color=colors[cow_id], ha='center', va='bottom')
    
    draw_pen(ax1, cam_coord, anchor=False, structure=False, legend=False)
    
    # print(f'n_lying: {n_lying}, n_nonlying: {n_nonlying}')

    return all_cows_line_set


def plot_line_3d(ax, line_eq, color, alpha):
    origin, direction = line_eq[0:3]/100, line_eq[3:6]/100

    ax.plot3D([origin[0], origin[0] + direction[0]], 
            [origin[1], origin[1] + direction[1]], 
            [origin[2], origin[2] + direction[2]], c=color, alpha=alpha)


# ===============================================
""" Main program from here """
def main(args):

    print('Date: ' + str(args.date))

    global ax1, combined_timestamps, curr_timestamp, cow_data_list
    global frame_height, frame_width, behav_dict_list
    global cam_list, Proj_cam_list, label_dir, n_lying, n_nonlying
    global cam_coord, no_print, single_view

    date = args.date
    frame_height = args.frame_height
    frame_width = int(frame_height * 1.6)
    no_print = args.no_print
    single_view = args.single_view

    if single_view == True:
        print("Visual location with min of one view")

    cam_list = ['cam_1','cam_2','cam_3','cam_4']
    
    if args.freeze == True:
        run_status = False
    else:
        run_status = True

    # image_scale = frame_width/1920
    n_lying = 0
    n_nonlying = 0

    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
    yaml_dir = os.path.join(current_dir, "path.yaml")
    
    with open(yaml_dir, 'r') as file:
        file_dirs = yaml.safe_load(file)

    sensor_data_dir = file_dirs['sensor_data_dir']
    visual_data_dir = file_dirs['visual_data_dir'] 
    # image_dir = os.path.join(visual_data_dir, 'images')
    label_dir = os.path.join(visual_data_dir, 'labels', 'combined')
    proj_mat_dir = os.path.join(visual_data_dir, 'proj_mat')
    behav_label_dir = os.path.join(sensor_data_dir, 'behavior_labels', 'individual')

    print("proj mat: ", proj_mat_dir)

    Proj_cam_list = read_projection_matrices(proj_mat_dir, date)

    cam_coord = []
    for idx in range(4):
        proj_mat = Proj_cam_list[idx]
        cam_coord.append(cal_cam_coord(proj_mat))
        print(f"Cam {idx+1} loc: {cal_cam_coord(proj_mat).astype(int)}")
    cam_coord = np.asarray(cam_coord).reshape((4,3))

    # Extract timestamps from txt labels 
    timestamp_list = []
    for i in range(1,5):
        try:
            cam_name = f"cam_{i:d}"
            folder_path = os.path.join(label_dir, date, cam_name)
            set_list = search_files(folder_path, search_text='_', file_format=".txt")
            timestamp_list = list(set(timestamp_list).union(set_list))
        except:
            pass
    timestamp_list.sort() # must sort here
    combined_timestamps = []
    for single_time in timestamp_list:
        combined_timestamps.append(int(single_time[0:10]))
    combined_timestamps = np.asarray(combined_timestamps)
    print('Combined: ' + str(len(combined_timestamps)))

    # Read standing reference
    ## Create the dicts to store the data for each cow
    behav_dict_list = []
    for i in range(1,17):
        behav_dict = {'cow_id':i, 'behav': []}
        behav_dict_list.append(behav_dict)
    # stand_ref_array = np.zeros((len(combined_timestamps), 16))
    for behav_dict in behav_dict_list:
        # try:
        if True:
            cow_id = behav_dict['cow_id']
            cow_name = f'C{cow_id:02d}' 
            df = pd.read_csv(os.path.join(behav_label_dir, f"{cow_name}_{date}.csv"))
            df = df.drop(columns=['datetime'])
            behav_df = df[df['timestamp'].isin(combined_timestamps)] # Filter rows where the timestamp exists in the list
            behav_dict['behav'] = behav_df.values # Two columns
            assert behav_dict['behav'].shape[1] == 2, f"Wrong shape {behav_dict['behav'].shape[1]}"
            # stand_ref_array[:, cow_id-1] = stand_ref[:,2]
        # except Exception as e:
        #     print(e)

    ## Initialize the first figure and a 3D axis
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    # plt.tight_layout() # Do not use

    ani = Player(fig=fig1, func=update, maxi=len(combined_timestamps)-1, run_status=run_status, interval_ms=3000) 
    plt.show()


if __name__ == '__main__':

    date_list = ["0721", "0722", "0723", "0724","0725","0726","0727","0728","0729","0730","0731","0801","0802", "0803","0804"]
    colors = ['grey','blue', 'green', 'red', 'orange', 'black', 'purple', 'teal','maroon','hotpink','darkgreen','aqua','blue', 'green', 'red', 'orange','black', 'purple', 'teal','maroon','hotpink']

    # settings
    parser = argparse.ArgumentParser(description='CowLoc visualization')
    parser.add_argument('--freeze', action='store_true', help='Use --freeze if the annimation is laggy')
    parser.add_argument('--date', type=str, default='0725', choices=date_list, help='The date of the experiment for displaying the data in MMDD')
    parser.add_argument('--frame_ratio', type=float, default=1.6)
    parser.add_argument('--frame_height', type=int, default=2800, help='Height of the frame being displayed')
    parser.add_argument('--single_view', action='store_true', help='Visual location with min one view')
    parser.add_argument('--no_print', action='store_true', help='Stop printing out')

    args = parser.parse_args()

    main(args)


