
import os
import numpy as np
# import argparse
import pandas as pd

from datetime import datetime
from collections import defaultdict

from utils.projection import project_image2world
from utils.line_geometry import cal_line_equation, cal_dist_point_to_line
from utils.AdaGrad_visual_loc import visual_localization
from utils.cmb_eval import cmb_eval
import warnings

import pytz
from tqdm import tqdm

# Set a global time zone: Central Time
CT_time_zone = pytz.timezone('America/Chicago')

def ratio_to_pixel(disp_resolution, xyn_ratio):
    # down_scale = label_resolution[0]/disp_resolution[0]
    width_loc = int(disp_resolution[0] * xyn_ratio[0])
    height_loc = int(disp_resolution[1] * xyn_ratio[1])
    return width_loc, height_loc

def read_bbox_labels(text_file_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_array = np.loadtxt(text_file_dir)
    # print(type(data_array))
    # print(np.shape(data_array))
    return np.atleast_2d(data_array)

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

def find_behav(dataframe, unix_timestamp):
    # Assuming the dataframe has columns named 'column1' and 'column2'
    result = dataframe[dataframe['timestamp'] == unix_timestamp]['behavior'].values
    if len(result) > 0:
        return result[0]
    else:
        return None  # Or handle the case when the number is not found

def delete_files_in_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # Delete all files within the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print("All files within the folder have been deleted.") 

def mrgb_proc(id_list, selected_timestamps, gt_behav_full_16_list, cam_coord, Proj_cam_list, pred_label_dir, date = '0725', lying = False, save_csv = False, report_outliner = False, no_print_detail = True):

    frame_height = 2800
    resolution = (int(frame_height*1.6), frame_height)

    current_dir = os.path.join(os.path.dirname(__file__))  # Folder

    if save_csv == True:
        for cow_id in range(1,17):
            out_behav_dir = os.path.join(current_dir, 'behaviors')
            cow_name = f'C{cow_id:02d}'
            output_dir = os.path.join(out_behav_dir, cow_name)

            # delete the old files
            # delete_files_in_folder(output_dir)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # csv_output_file = cow_name + '_' + date + '.csv'
            # output_dir = os.path.join(output_dir, csv_output_file)
            # if not os.path.exists(output_dir):
            #     data = pd.DataFrame(columns=['timestamp', 'datetime', 'behavior'])
            #     data.to_csv(output_dir, index=False)
                # print(f"{csv_output_file} saved")

    output_dict = {
        'C01':np.empty((0,3)),
        'C02':np.empty((0,3)),
        'C03':np.empty((0,3)),
        'C04':np.empty((0,3)),
        'C05':np.empty((0,3)),
        'C06':np.empty((0,3)),
        'C07':np.empty((0,3)),
        'C08':np.empty((0,3)),
        'C09':np.empty((0,3)),
        'C10':np.empty((0,3)),
        'C11':np.empty((0,3)),
        'C12':np.empty((0,3)),
        'C13':np.empty((0,3)),
        'C14':np.empty((0,3)),
        'C15':np.empty((0,3)),
        'C16':np.empty((0,3))
    }

    cow_data = np.empty((0,2)).astype(int)
    cam_list = ['cam_1','cam_2','cam_3','cam_4']

    n_missing = 0
    
    for curr_timestamp in tqdm(selected_timestamps):

        # Create gt_id_list
        gt_id_list = []
        for gt_id in range(1,17):
            gt_behav = int(find_behav(gt_behav_full_16_list[gt_id-1], curr_timestamp))
            if gt_behav > 0:
                if lying == False:
                    if gt_behav != 7:
                        gt_id_list.append(gt_id)
                else:
                    gt_id_list.append(gt_id)

        ## Go through each camera
        bbox_dict_list = []
        for cam_idx, cam_name in zip(range(4), cam_list):
            cam_view_dict = {}
            proj_mat = Proj_cam_list[cam_idx]
            cam_view_dict['cam_idx'] = cam_idx
            n_rays = 0

            datetime_var = datetime.fromtimestamp(curr_timestamp, CT_time_zone)
            text_file_name = f'{curr_timestamp:d}_{datetime_var.hour:02d}-{datetime_var.minute:02d}-{datetime_var.second:02d}.txt'
            # timestamp = text_file_name[0:10]
            pred_file_dir = os.path.join(pred_label_dir, date, cam_name, text_file_name)

            ## Dict structure:
            #  cam_view_dict {'cam_idx', 'n_rays', 'list_dict'}
            #    +--list_dict [data_point1, data_point2, ...]
            #         +---data_point {'cow_id', 'weight', 'line_eq', 'behav'}

            dummy_data_point = {'cow_id':-1, 'bbox':np.zeros(6), 'line_eq':np.zeros(6)}
            dummy_cam_view_dict = {'cam_idx':cam_idx, 'n_rays':0, 'list_dict':[dummy_data_point]}

            try:
            # if True:
                bboxes_data = read_bbox_labels(pred_file_dir)
                
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
                        width, height = ratio_to_pixel(resolution, bbox_xyxyn[2:4])
                        data_point['weight'] = width # + height
                        data_point['behav'] = int(row[5])

                        # Projecting center of bbox to the 3D ground
                        point2 = project_image2world(bboxes_data[idx, 1:3], proj_mat, 50)
                        line_eq = cal_line_equation(cam_coord[cam_idx], point2)
                        data_point['line_eq'] = line_eq

                        if row[0] in id_list:
                            if lying == False:
                                behav = int(row[5])
                                if behav != 7:
                                    list_dict.append(data_point)
                            else:
                                list_dict.append(data_point)

                    ## bbox_data_w_proj_lines: [idx: 0] [xyxy] [yx] [conf] [ABC DEF line]
                    cam_view_dict['list_dict'] = list_dict
                else:
                    # print('Warning: no bbox in the frame')
                    cam_view_dict = dummy_cam_view_dict
            except:
                cam_view_dict = dummy_cam_view_dict
                print("missing", text_file_name)
                n_missing+=1
                assert n_missing < 1000, f'Wrong label directory: {pred_file_dir}'

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
                                'behav_list':[],
                                'weight_list':[]
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
                        single_cow_dict['behav_list'].append(single_bbox_dict['behav'])
                        single_cow_dict['weight_list'].append(single_bbox_dict['weight'])
                        # print('found')

        single_ts_pred_data = np.empty((0,2))

        # Considering all lines of a single cow
        for single_cow_dict in all_cows_line_set:
            if len(single_cow_dict['line_list']) > 0:
                line_eqs = np.asarray(single_cow_dict['line_list'])
                curr_timestamp = single_cow_dict['timestamp']
                pred_id = single_cow_dict['cow_id']

                valid_lines = np.empty((0,2))
      
                # Multi-view localization
                if line_eqs.shape[0] > 1:
                    ## Localization using gradient descent   
                    # print(line_eqs.astype(int))
                    nearest_point, total_distance, iter, gradient = visual_localization(line_eqs)
                    nearest_point = nearest_point.astype(int)
                    single_cow_dict['location'] = nearest_point
                    if no_print_detail == False:
                        print(f"{single_cow_dict['cow_id']:2d}  {nearest_point}\td:{int(total_distance)/100:.2f}\t#{iter}\tg:{gradient:.2f}")
                    
                    # Convert the dict to [cow_id, weight, behav, line_eq[6]]
                    for i in range(line_eqs.shape[0]):
                        weight = single_cow_dict['weight_list'][i]
                        behav = single_cow_dict['behav_list'][i]
                        line_eq = single_cow_dict['line_list'][i]
                        # print(f'{cow_id}, {weight}, {behav}, {line_eq.astype(int)}')

                        ## Find outliners
                        # Find distance from nearest_point to the lines
                        dist = int(cal_dist_point_to_line(line_eq, nearest_point))
                        # print(dist)
                        if dist > 200 or i > 3:
                            cam_id = single_cow_dict['cam_idx_list'][i] + 1
                            if report_outliner == True:
                                print(f"==> Outlier: cow {single_cow_dict['cow_id']}, cam_{cam_id} ({line_eqs.shape[0]} cams)")
                        else:
                            valid_line = np.array([weight, behav])
                            valid_lines = np.vstack((valid_lines, valid_line))

                # When the cow is only available in a single view
                else:
                    weight = single_cow_dict['weight_list'][0]
                    behav = single_cow_dict['behav_list'][0]
                    valid_line = np.array([weight, behav])
                    valid_lines = np.atleast_2d(valid_line)
                        
                # print(valid_lines)

                ## Weighted_majority_voting
                class_weights = defaultdict(float)
                for row in valid_lines:
                    weight, behav_id = row
                    # print(behav_id)
                    class_weights[behav_id] += weight
                
                try:
                    # Find the class with the highest total weight
                    pred_behav = int(max(class_weights, key=class_weights.get))

                    # gt_behav = int(find_behav(gt_behav_full_16_list[pred_id-1], curr_timestamp))
                    datapoint = np.array([int(pred_id), int(pred_behav)])
                    single_ts_pred_data = np.vstack((single_ts_pred_data, datapoint)).astype(int)

                    # ## For saving into csv
                    # if save_csv == True:
                    #     behav = valid_line[1]
                    #     datetime_obj = datetime.fromtimestamp(curr_timestamp, CT_time_zone)
                    #     formatted_time = datetime_obj.strftime('%H:%M:%S')

                    #     behav_datapoint = np.hstack((curr_timestamp, formatted_time, behav))

                    #     cow_name = f'C{pred_id:02d}'
                    #     output_dict[cow_name] = np.vstack((output_dict[cow_name], behav_datapoint))
                except:
                    pass
                    # print('\t__ max() arg is an empty sequence')

        # if len(gt_behav_full_16_list) > 0:
        
        pred_id_list = single_ts_pred_data[:,0].astype(int)
        # pred_behav_list = single_ts_pred_data[:,1].astype(int)
        # print('pred_id_list', pred_id_list)
        # print('gt_id_list', gt_id_list)
        # print('pred_behav_list', pred_behav_list)
        # print('gt_behav_list', gt_behav_list)
        for gt_id in gt_id_list:
            if gt_id in pred_id_list:
                gt_behav = int(find_behav(gt_behav_full_16_list[gt_id-1], curr_timestamp))
                for pred_id, pred_behav in np.atleast_2d(single_ts_pred_data):
                    if pred_id == gt_id:
                        datapoint = np.array([pred_behav, gt_behav])
                        cow_data = np.vstack((cow_data, datapoint))
            else:
                pred_behav = 0
                datapoint = np.array([pred_behav, gt_behav])
                cow_data = np.vstack((cow_data, datapoint))
            
        # print(behav_datapoint)
        

    # # Append to csv
    # if save_csv == True:
    #     for cow_id in range(1,17):
    #         out_behav_dir = os.path.join(current_dir, 'behaviors')
    #         cow_name = f'C{cow_id:02d}'
    #         csv_output_file = cow_name + '_' + date + '.csv'
    #         output_dir = os.path.join(out_behav_dir, cow_name, csv_output_file)

    #         single_dict = output_dict[cow_name]
    #         if single_dict.size > 0:
    #             row_data = pd.DataFrame(single_dict, columns=['timestamp', 'datetime', 'behavior'])
    #             row_data.to_csv(output_dir, header=True, index=False)

    #             print(f'{csv_output_file} saved')

    # print(np.shape(cow_data))

    y_pred = cow_data[:,0]
    y_test = cow_data[:,1]

    return y_pred, y_test