
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import datetime as dt

from datetime import datetime
import sys
import cv2

from utils.draw_bbox import *

import pytz
import yaml
from tqdm import tqdm

from multiprocessing import Pool
from functools import partial
import shutil

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

from PIL import Image

def delete_eaDir(folder_path):
    # for folder_name in os.listdir(root_dir):
    #     folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        eaDir_path = os.path.join(folder_path, "@eaDir")
        if os.path.exists(eaDir_path):
            shutil.rmtree(eaDir_path)
            print(f"{eaDir_path} => deleted")
        else:
            print(f"{eaDir_path} not found")

def cam_crop(combined_timestamps, visual_data_dir, set, cam_id):

    # print(f"visual_data_dir:: {visual_data_dir}")

    date = '0725'

    height = 2800
    width = int(height*1.6)

    # current_dir = os.path.join(os.path.dirname(__file__))  # Folder
    # yaml_dir = os.path.join(current_dir, "config.yaml")

    # with open(yaml_dir, 'r') as file:
        # file_dirs = yaml.safe_load(file)
    # label_dir = file_dirs['standing_label_dir']
    # input_dir = file_dirs['visual_data_dir']
    # dump_dir = file_dirs['dump_dir']
    # label_dir = os.path.join(input_dir, 'labels')

    # ## Go through each camera
    # bbox_dict_list = []
    # for cam_id in range(1,5):
    cam_name = f"cam_{cam_id}"

    for curr_timestamp in tqdm(combined_timestamps):
        datetime_var = datetime.fromtimestamp(curr_timestamp, CT_time_zone)
        text_file_name = f'{curr_timestamp:d}_{datetime_var.hour:02d}-{datetime_var.minute:02d}-{datetime_var.second:02d}.txt'
        in_img_name = f'{curr_timestamp:d}_{datetime_var.hour:02d}-{datetime_var.minute:02d}-{datetime_var.second:02d}.jpg'
        out_img_name = f'{curr_timestamp:d}_{datetime_var.hour:02d}-{datetime_var.minute:02d}-{datetime_var.second:02d}_{cam_id}.jpg'

        file_dir = os.path.join(visual_data_dir, 'labels', set, date, cam_name, text_file_name)
        # print(f"file_dir:: {file_dir}")

        image_path = os.path.join(visual_data_dir, 'images', date, cam_name, in_img_name) 
        image = cv2.imread(image_path)

        try:
            bboxes_data = read_bbox_labels(file_dir)
            if len(bboxes_data.flatten()) > 0:
                for row in bboxes_data:
                    cow_id = int(row[0])
                    class_folder = f'{cow_id:d}'

                    center_x = int(row[1] * width)
                    center_y = int(row[2] * height)
                    b_width = int(row[3] * width)
                    b_height = int(row[4] * height)

                    output_path = os.path.join(visual_data_dir, 'cropped_bboxes', set, class_folder, out_img_name)
                    
                    # Calculate the top-left corner of the bounding box
                    x1 = int(center_x - b_width / 2)
                    y1 = int(center_y - b_height / 2)
                    
                    # Calculate the bottom-right corner of the bounding box
                    x2 = int(center_x + b_width / 2)
                    y2 = int(center_y + b_height / 2)
                    
                    # Crop the image
                    cropped_image = image[y1:y2, x1:x2]
                    
                    # Save the cropped image as JPEG
                    cv2.imwrite(output_path, cropped_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    # print('saved', output_path)
        except Exception as e:
            print(e) # Print out the error message

# ===============================================
""" Main program from here """
if __name__ == '__main__':

    date = '0725'

    current_dir = os.path.join(os.path.dirname(__file__))  # Folder
    yaml_dir = os.path.join(current_dir, "path.yaml")

    with open(yaml_dir, 'r') as file:
        file_dirs = yaml.safe_load(file)

    visual_data_dir = file_dirs['visual_data_dir']

    set_list = ['standing','lying']#,'combined']
    for set in set_list:
        print('Set', set)

        # Create output folders
        for cow_id in range(1,17):
            cow_name = f'{cow_id:d}'
            output_dir = os.path.join(visual_data_dir, 'cropped_bboxes', set, cow_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        # Extract timestamps from txt labels 
        temp_list = []
        for i in range(1,5):
            cam_name = f"cam_{i:d}"
            folder_path = os.path.join(visual_data_dir, 'images', date, cam_name)
            print(folder_path)
            filename_list = search_files(folder_path, search_text='_', file_format=".jpg")
            for single_filename in filename_list:
                temp_list.append(int(single_filename[0:10]))
        temp_list = np.asarray(temp_list)
        combined_timestamps = list(np.unique(temp_list))
        combined_timestamps.sort() # must sort here
        print('Combined: ' + str(len(combined_timestamps)))


        cam_list = [1, 2, 3, 4]

        func = partial(cam_crop, combined_timestamps, visual_data_dir, set)

        if len(cam_list) > 1:
            pool = Pool(processes=len(cam_list))
            pool.map(func, cam_list)
        else:
            for cam_id in cam_list:
                cam_crop(combined_timestamps, visual_data_dir, set, cam_id)
        
        # Delete @eaDir
        root_directory = os.path.join(visual_data_dir, 'cropped_bboxes', set)

        for i in range(1, 17):
            sub_dir = os.path.join(root_directory, str(i))
            if os.path.isdir(sub_dir):
                delete_eaDir(sub_dir)
            else:
                print(f"{sub_dir} not exist")
        
        
