import os
import shutil
from datetime import datetime
import json
import pytz
import argparse


# Set a global time zone: Central Time
CT_time_zone = pytz.timezone('America/Chicago')

def config_to_datetime(datetime_str, timezone):
    # Parse the human-readable datetime string into a naive datetime object
    naive_datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    # Localize the naive datetime object to the specific time zone
    localized_datetime = timezone.localize(naive_datetime)
    return localized_datetime

def parse_time_ranges(time_ranges):
    parsed_ranges = {}
    for key, time_range in time_ranges.items():
        start = config_to_datetime(time_range[0], CT_time_zone)
        end = config_to_datetime(time_range[1], CT_time_zone)
        parsed_ranges[key] = (start, end)
    return parsed_ranges

def get_datetime_from_filename(file_name):
    file_timestamp = int(file_name[0:10]) # How is this the timestamp?
    # print('filename_timestamp', file_timestamp)
    file_timestamp = datetime.fromtimestamp(file_timestamp, CT_time_zone)
    # print('file_timestamp', file_timestamp, '\n')
    return file_timestamp



# Function to create output directories if they don't exist
def create_output_dirs(fold_name, output_dir, experiment_type):
    for split in ['train', 'val', 'test']:
        # for behav in ['1','2','3','4','5','6','7']
        if experiment_type == 'cow_id':
            for cow in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']:
                os.makedirs(os.path.join(output_dir, fold_name, split, cow), exist_ok=True)
        
        elif experiment_type == 'behavior':
            for behav in ['1','2','3','4','5','6','7']: 
                os.makedirs(os.path.join(output_dir, fold_name, split, behav), exist_ok=True)
        # os.makedirs(os.path.join(output_dir, fold_name, split, 'labels'), exist_ok=True)


def in_time_ranges(file_datetime, keys, group_1_ranges, group_2_ranges):
    for key in keys:
        start1, end1 = group_1_ranges[key]
        start2, end2 = group_2_ranges[key]
        if (start1 <= file_datetime < end1) or (start2 <= file_datetime < end2):
            return True
    return False

# Function to check if a file is an image file
def is_image_file(filename):
    return filename.endswith(('.jpg', '.jpeg', '.png', '.bmp'))

# Function to find subfolders in a given directory
def find_subfolders(directory):
    return [f.name for f in os.scandir(directory) if f.is_dir()]


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process annotations with specific splits.')

    # Define the arguments
    parser.add_argument('--data_splits_config_file', type=str, required=True, help='Path to the data splits configuration file.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing the images.')
    # parser.add_argument('--label_dir', type=str, required=True, help='Directory containing the labels.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output.')
    parser.add_argument('--experiment_type', type=str, required=True, help='Behavior or Cow_id')

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments (or use them in your application)
    print(f"Data Splits Config File: {args.data_splits_config_file}")
    print(f"Image Directory: {args.image_dir}")
    # print(f"Label Directory: {args.label_dir}")
    print(f"Output Directory: {args.output_dir}")

    # Load the configuration file
    with open(args.data_splits_config_file, 'r') as file:
        config = json.load(file)    

    # Parse the time ranges for both groups
    group_1_ranges = parse_time_ranges(config['group_1'])
    group_2_ranges = parse_time_ranges(config['group_2'])

    image_dir = args.image_dir
    # label_dir = args.label_dir
    output_dir = args.output_dir
    experiment_type = args.experiment_type

    # Find subfolders in the images and labels directories
    # cam_dirs = find_subfolders(image_dir)
    behav_main_dir = image_dir
    behv_dirs = find_subfolders(behav_main_dir) 
    print(behv_dirs)
    # behv_dirs = cam_dirs # 1, 2, 3, ..., 7

    # Organize data for each fold
    for fold_name, fold_data in config['folds'].items():
        create_output_dirs(fold_name, output_dir, experiment_type)

        # Organize images
        for behav_dir in behv_dirs: #for behav_dir in beh_dirs # 1, 2, 3, ..
            print(f'Organizing {behv_dirs} data...')
            # cam_image_dir = os.path.join(image_dir, behav_dir)
            # cam_label_dir = os.path.join(label_dir, behav_dir)
            beh_image_dir = os.path.join(behav_main_dir, behav_dir) # joined into behv_main/1,2,...
            
            for filename in (sorted(os.listdir(beh_image_dir))): # inside behav subfolders -> cropped bboxes
                if is_image_file(filename):
                    file_path = os.path.join(beh_image_dir, filename) 
                    # print(f'This is filepath {file_path}')
                    file_datetime = get_datetime_from_filename(filename)
                    # print(f'This is file_datetime {file_datetime}')
                    # prefixed_filename = f"{cam_dir}_{filename}" # No need to, it's already formatted
                    prefixed_filename = filename
                    # print(filename)

                    if in_time_ranges(file_datetime, fold_data['train'], group_1_ranges, group_2_ranges):
                        split = 'train'
                        # print('Going into train')
                    elif in_time_ranges(file_datetime, fold_data['val'], group_1_ranges, group_2_ranges):
                        split = 'val'
                    elif in_time_ranges(file_datetime, fold_data['test'], group_1_ranges, group_2_ranges):
                        split = 'test'
                    else:
                        continue

                    copied_filepath = os.path.join(output_dir, fold_name, split, behav_dir, prefixed_filename)
                    shutil.copyfile(file_path, copied_filepath)
                    
                    # print(f'This is copied path {copied_filepath}')

    print("Data organized successfully.")


if __name__ == "__main__":
    main()
