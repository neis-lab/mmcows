
import os
import numpy as np

import pytz
from datetime import datetime

import warnings

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

def find_behav(dataframe, unix_timestamp):
    # Assuming the dataframe has columns named 'column1' and 'column2'
    result = dataframe[dataframe['timestamp'] == unix_timestamp]['behavior'].values
    if len(result) > 0:
        return result[0]
    else:
        return None  # Or handle the case when the number is not found

def select_range(timestamps, point1, point2):
    assert point1 != point2, f'Invalid range {point2} = {point1}'
    # print(f'{point1} vs {point2}')
    
    if point1 > point2:
        selected = [value for value in timestamps if point2 <= value < point1]
    else:
        selected = [value for value in timestamps if point1 <= value < point2]
    return selected

def find_in_first_col_return_last_col_value(array, scalar):
    # Check if the array is 2D
    if array.ndim != 2:
        raise ValueError("The input array must be 2D.")
    
    # Find the index of the row where the scalar is in the first column
    for i in range(array.shape[0]):
        if array[i, 0] == scalar:
            # Return the value from the last column of the same row
            return array[i, -1]
    
    # If the scalar is not found, return None or an appropriate value
    return None

def srgb_proc(selected_timestamps, id_list, behav_gt_list, pred_label_dir, gt_label_dir, date, lying = False):
    cow_data = np.empty((0,2)).astype(int)
    n_missing = 0
    pred_file_dir = []
    gt_file_dir = []
    for i in range(1,5):
        cam_name = f"cam_{i:d}"
        # folder_path = os.path.join(pred_label_dir, date, cam_name)
        # print(folder_path)
        # filename_list = search_files(folder_path, search_text='_', file_format=".txt")
        for curr_timestamp in selected_timestamps:
            # curr_timestamp = int(single_filename[0:10])
            datetime_var = datetime.fromtimestamp(curr_timestamp, CT_time_zone)

            try:
            # if True:
                text_file_name = f'{curr_timestamp:d}_{datetime_var.hour:02d}-{datetime_var.minute:02d}-{datetime_var.second:02d}.txt'

                gt_file_dir = os.path.join(gt_label_dir, date, cam_name, text_file_name)
                # print(gt_file_dir)
                gt_bbox_data = np.atleast_2d(read_bbox_labels(gt_file_dir))
                if len(gt_bbox_data.flatten()) > 0:
                    gt_id_list = gt_bbox_data[:,0].astype(int)

                    pred_file_dir = os.path.join(pred_label_dir, date, cam_name, text_file_name)
                    # print(pred_file_dir)
                    pred_bbox_data = np.atleast_2d(read_bbox_labels(pred_file_dir))
                    if len(pred_bbox_data.flatten()) > 0:
                        pred_id_list = pred_bbox_data[:,0]

                        if len(pred_id_list) != len(np.unique(pred_id_list)):
                            print(f'dupplication in cam {cam_name} {text_file_name}')

                        for gt_id in gt_id_list:
                            gt_behav = int(find_behav(behav_gt_list[gt_id-1], curr_timestamp))
                            if gt_id in pred_id_list:
                                for pred_data_row in pred_bbox_data:
                                    pred_id, _, _, _, _, pred_behav = pred_data_row
                                    if pred_id == gt_id:
                                        if lying == False:
                                            if gt_behav != 7:
                                                datapoint = np.array([int(pred_behav), gt_behav])
                                                cow_data = np.vstack((cow_data, datapoint))
                                        else:
                                            datapoint = np.array([int(pred_behav), gt_behav])
                                            cow_data = np.vstack((cow_data, datapoint))
                            else:
                                if lying == False:
                                    if gt_behav != 7:
                                        pred_behav = 0
                                        datapoint = np.array([pred_behav, gt_behav])
                                        cow_data = np.vstack((cow_data, datapoint))
                                else:
                                    pred_behav = 0
                                    datapoint = np.array([pred_behav, gt_behav])
                                    cow_data = np.vstack((cow_data, datapoint))

                            # for row in bboxes_data:
                            #     cow_id = int(row[0])
                            #     if cow_id in gt_id_list:
                            #         pred_behav = int(row[5])
                            #         # cow_name = f'C{cow_id:02d}'
                            #         test_behav = int(find_behav(behav_gt_list[cow_id-1], curr_timestamp))
                            #         # if test_behav != pred_behav:
                            #         #     print(f'{text_file_name}  {cow_name} {test_behav}')
                            #         if lying == False:
                            #             if test_behav != 7:
                            #                 datapoint = np.array([cow_id, pred_behav, test_behav])
                            #                 cow_data = np.vstack((cow_data, datapoint))
                            #         else:
                            #             datapoint = np.array([cow_id, pred_behav, test_behav])
                            #             cow_data = np.vstack((cow_data, datapoint))
                                    
                            #     else: # if the cow is not in the gt
                            #         datapoint = np.array([cow_id, pred_behav, test_behav])

                            # cow_data = np.vstack((cow_data, datapoint))
            except:
                print('missing', text_file_name)
                n_missing+=1
                assert n_missing < 1000, f'Wrong label directory: {gt_file_dir} and {pred_file_dir}'
                pass

    print(np.shape(cow_data))

    y_pred = cow_data[:,0]
    y_test = cow_data[:,1]

    return y_pred, y_test
