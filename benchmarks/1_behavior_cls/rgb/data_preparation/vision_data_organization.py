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
    file_timestamp = int(file_name[0:10])
    # print('filename_timestamp', file_timestamp)
    file_timestamp = datetime.fromtimestamp(file_timestamp, CT_time_zone)
    # print('file_timestamp', file_timestamp, '\n')
    return file_timestamp



# Function to create output directories if they don't exist
def create_output_dirs(fold_name, output_dir):
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, fold_name, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, fold_name, split, 'labels'), exist_ok=True)


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
    parser.add_argument('--label_dir', type=str, required=True, help='Directory containing the labels.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output.')

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments (or use them in your application)
    print(f"Data Splits Config File: {args.data_splits_config_file}")
    print(f"Image Directory: {args.image_dir}")
    print(f"Label Directory: {args.label_dir}")
    print(f"Output Directory: {args.output_dir}")

    # Load the configuration file
    with open(args.data_splits_config_file, 'r') as file:
        config = json.load(file)    

    # Parse the time ranges for both groups
    group_1_ranges = parse_time_ranges(config['group_1'])
    group_2_ranges = parse_time_ranges(config['group_2'])

    image_dir = args.image_dir
    label_dir = args.label_dir
    output_dir = args.output_dir

    # Find subfolders in the images and labels directories
    cam_dirs = find_subfolders(image_dir)

    # Organize data for each fold
    for fold_name, fold_data in config['folds'].items():
        create_output_dirs(fold_name, output_dir)

        # Organize images
        for cam_dir in cam_dirs:
            print(f'Organizing {cam_dir} data...')
            cam_image_dir = os.path.join(image_dir, cam_dir)
            cam_label_dir = os.path.join(label_dir, cam_dir)
            
            for filename in os.listdir(cam_image_dir):
                if is_image_file(filename):
                    file_path = os.path.join(cam_image_dir, filename)
                    file_datetime = get_datetime_from_filename(filename)
                    prefixed_filename = f"{cam_dir}_{filename}"

                    if in_time_ranges(file_datetime, fold_data['train'], group_1_ranges, group_2_ranges):
                        split = 'train'
                    elif in_time_ranges(file_datetime, fold_data['val'], group_1_ranges, group_2_ranges):
                        split = 'val'
                    elif in_time_ranges(file_datetime, fold_data['test'], group_1_ranges, group_2_ranges):
                        split = 'test'
                    else:
                        continue

                    shutil.copy(file_path, os.path.join(output_dir, fold_name, split, 'images', prefixed_filename))

            # Organize labels
            for filename in os.listdir(cam_label_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(cam_label_dir, filename)
                    file_datetime = get_datetime_from_filename(filename)
                    prefixed_filename = f"{cam_dir}_{filename}"

                    if in_time_ranges(file_datetime, fold_data['train'], group_1_ranges, group_2_ranges):
                        split = 'train'
                    elif in_time_ranges(file_datetime, fold_data['val'], group_1_ranges, group_2_ranges):
                        split = 'val'
                    elif in_time_ranges(file_datetime, fold_data['test'], group_1_ranges, group_2_ranges):
                        split = 'test'
                    else:
                        continue

                    shutil.copy(file_path, os.path.join(output_dir, fold_name, split, 'labels', prefixed_filename))

    print("Data organized successfully.")


if __name__ == "__main__":
    main()