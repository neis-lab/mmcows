
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import cv2
from datetime import datetime
import math

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import rgb2hex

from utils.plot_player import Player 
from utils.pen_model import *
from utils.projection import *
from utils.mask_profiles import *
from utils.draw_bbox import *
from utils.handle_xml import read_projection_matrices
from utils.line_geometry import *

import warnings
import zipfile
from io import BytesIO
from PIL import Image

import pytz
import yaml

# Set a global time zone: Central Time
CT_time_zone = pytz.timezone('America/Chicago')

# Ignore UserWarning from Matplotlib
warnings.filterwarnings("ignore", category=UserWarning)

# def load_directories_from_yaml(yaml_file):
#     with open(yaml_file, 'r') as file:
#         data = yaml.safe_load(file)
#     return data

def create_folder(folder_path):
    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # Create the folder if it doesn't exist
        os.makedirs(folder_path)

def save_uwb_label_txt(output_text_path, bboxes_data):
    with open(output_text_path, 'w') as f:
        for row in bboxes_data:
            line = ' '.join(map(str, row))
            f.write(line + '\n')
    pass

def mask_image(image, points):
    mask = np.zeros_like(image) # Create a blank mask
    points = np.array([points], dtype=np.int32) # Convert the points to a numpy array
    cv2.fillPoly(mask, points, (255, 255, 255)) # Fill the mask with the polygon formed by the points
    result = cv2.bitwise_and(image, mask) # Bitwise AND operation to mask out the area outside the shape
    return result

def image_masking(idx, pil_image):
    global cam_bound_list
    cam_bound = cam_bound_list[idx]
    points = cam_bound.tolist()

    opencv_image = np.array(pil_image)
    result_image = mask_image(opencv_image, points)
    pil_image = Image.fromarray(result_image)

    return pil_image

# /images/date/cam_x/*.jpg
def read_images(file_path, date, image_name):
    global cam_list, masking, disp_resolution
    image_list = []

    for idx, cam in enumerate(cam_list):
        try:
            img_dir = os.path.join(file_path , date , cam , image_name)
            img = Image.open(img_dir)

            if masking == True:
                img = image_masking(idx, img)
            
            img = img.resize(disp_resolution)
            image_list.append(img) # Use PIL to open the image from the file-like object
        
        except:
            image_list.append(gen_black_image(disp_resolution))
            print('Warning: image not found: ' + img_dir)

    return image_list[0], image_list[1], image_list[2], image_list[3]

def euler_to_rotation_matrix(roll_rad, pitch_rad, yaw_rad):
    # Rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]])

    R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

    R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]])

    # Total rotation matrix
    R_total = np.dot(R_z, np.dot(R_y, R_x))

    return R_total

def rotate_vector(vector, roll, pitch, yaw):
    R_total = euler_to_rotation_matrix(roll, pitch, yaw)
    result_vector = np.dot(R_total, vector)
    return result_vector

def plot_images(ax, first_imgs, curr_timestamp):

    img_1, img_2, img_3, img_4, _,_,_,_ = first_imgs
    timestamp = int(curr_timestamp)

    global black_image, cam_list, standing_bbox_en, lying_bbox_en, en_ground_boundary
    global uwb_locs  # Unit: cm
    global Proj_cam_list, image_dir, label_dir, prior_rest_y_loc, lying_refs, uwb_label_dir
    global label_resolution, disp_resolution, color_bbox

    disp_scale = label_resolution[0] / disp_resolution[0]
        
    # Reading individual images offers no improverment compared to importing from the zip file

    datetime_var = datetime.fromtimestamp(timestamp, CT_time_zone)
    image_name = f'{timestamp:d}_{datetime_var.hour:02d}-{datetime_var.minute:02d}-{datetime_var.second:02d}.jpg'
    date = f'{datetime_var.month:02d}{datetime_var.day:02d}' 

    img_cam_list = read_images(image_dir, date, image_name)

    ## Drawing the camera name
    out_img_list = []
    for idx, cam_name, cam_img in zip(range(4), cam_list, img_cam_list):
        out_img_list.append(draw_cam_name(disp_resolution, cam_img, idx))
    img_cam_list = out_img_list

    ## Drawing bbox for standing cows from yolo's inference results
    if standing_bbox_en == True:
        file_name = image_name[-25:-4]
        out_img_list = []

        for cam_idx, cam_name, cam_img in zip(range(4), cam_list, img_cam_list):
            file_dir = os.path.join(label_dir, date, cam_name, file_name + '.txt')
            try:
                # bboxes_data: [idx] [x_center, y_center, width, height]
                bboxes_data = read_bbox_labels(file_dir)
            except:
                bboxes_data = np.empty((0,5))
                print("Warning: label not found: " + file_dir)
            proj_mat = Proj_cam_list[cam_idx]

            if bboxes_data.size > 0:
        
                # projected_lines = []
                for idx, row in enumerate(bboxes_data):
                    bboxes_data[idx, 0] = row[0]
                    bbox_center = ratio_to_pixel(disp_resolution, row[1:3])
                    width_height = ratio_to_pixel(disp_resolution, row[3:5])
                    bboxes_data[idx, 1:5] = convert_bbox_format(bbox_center, width_height)

            out_img_list.append(draw_standing_bbox(cam_img, bboxes_data, 0, color_bbox))

        img_cam_list = out_img_list


    ## Drawing the ground grid
    if en_ground_grid == True:
        out_img_list = []
        for idx, cam_name, cam_img in zip(range(4), cam_list, img_cam_list):
            proj_mat = Proj_cam_list[idx]
            out_img_list.append(draw_ground_grid(cam_img, proj_mat, disp_scale))
        img_cam_list = out_img_list

    ## Drawing the ground boundary
    if en_ground_grid == True or en_ground_boundary == True:
        out_img_list = []
        for idx, cam_name, cam_img in zip(range(4), cam_list, img_cam_list):
            proj_mat = Proj_cam_list[idx]
            out_img_list.append(draw_ground_boundary(cam_img, proj_mat, disp_scale))
        img_cam_list = out_img_list

    ## Annotating uwb location points
    if en_uwb_points == True:
        out_img_list = []
        for idx, cam_name, cam_img in zip(range(4), cam_list, img_cam_list):
            proj_mat = Proj_cam_list[idx]
            out_img_list.append(draw_uwb_points(cam_img, uwb_locs, proj_mat, disp_scale))
        img_cam_list = out_img_list

    img_cam_1, img_cam_2, img_cam_3, img_cam_4 = img_cam_list

    img_1.set_array(img_cam_1)
    img_2.set_array(img_cam_2)
    img_3.set_array(img_cam_3)
    img_4.set_array(img_cam_4)

    bbox_props = dict(boxstyle="round", alpha=1, edgecolor='none', facecolor='white')
    axins = inset_axes(ax, width="20%", height="20%", loc='upper left')
    axins.text(2.4, 1.1, str(datetime_var)[0:19], ha="center", va="center", bbox=bbox_props, fontsize=12)
    axins.axis('off')

            
def img_view_formater(ax):
    global plot_scale

    profile_image_2 = (0.0, 0, plot_scale, plot_scale)
    profile_image_3 = (0.0, -0.03, plot_scale, plot_scale)
    profile_image_1 = (0.0, 0, plot_scale, plot_scale)
    profile_image_4 = (0.0, -0.03, plot_scale, plot_scale)

    """ Sub plots """
    axins_2 = inset_axes(ax, width="50%", height="50%", loc='upper right', bbox_to_anchor=profile_image_2, bbox_transform=ax.transAxes)
    axins_3 = inset_axes(ax, width="50%", height="50%", loc='lower right', bbox_to_anchor=profile_image_3, bbox_transform=ax.transAxes)
    axins_1 = inset_axes(ax, width="50%", height="50%", loc='upper left', bbox_to_anchor=profile_image_1, bbox_transform=ax.transAxes)
    axins_4 = inset_axes(ax, width="50%", height="50%", loc='lower left', bbox_to_anchor=profile_image_4, bbox_transform=ax.transAxes)
    
    axins_1.axis('off')
    axins_2.axis('off')
    axins_3.axis('off')
    axins_4.axis('off')

    ax.set_visible(False)

    return axins_1, axins_2, axins_3, axins_4

def first_images(ax, first_imgs, curr_timestamp):

    axins_1, axins_2, axins_3, axins_4 = img_view_formater(ax)

    global black_image
    img_cam_1, img_cam_2, img_cam_3, img_cam_4 = first_imgs

    img_1 = axins_1.imshow(img_cam_1) 
    img_2 = axins_2.imshow(img_cam_2) 
    img_3 = axins_3.imshow(img_cam_3) 
    img_4 = axins_4.imshow(img_cam_4) 
        
    return img_1, img_2, img_3, img_4, axins_1, axins_2, axins_3, axins_4


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

def gen_black_image(size):
    black_image = Image.new('RGB', size, color='black')
    return black_image

def gen_black_images(size):
    black_image = gen_black_image(size)
    return black_image, black_image, black_image, black_image

def downsampling_by_second(sensor_data):
    # Downsampling from 10 Hz to 1 Hz by averaging data points within the same second
    n_rows, n_cols = np.shape(sensor_data)
    current_timestamp = np.floor(sensor_data[0,0])

    accumulated_data = sensor_data[0,1:]  # sensor_data[0,1:7] 
    n_similar_timestamp = 1.0

    downsampled_data = []

    for row in sensor_data[1:,:]:
        row_timestamp = np.floor(row[0])
        if current_timestamp == row_timestamp:

            accumulated_data += row[1:] # row[1:7]
            n_similar_timestamp += 1.0
        else:
            # Save the row data
            average_data = np.round(accumulated_data/n_similar_timestamp, 3)
            sampled_row_data = np.insert(average_data, 0, current_timestamp)
            downsampled_data.append(sampled_row_data)

            # Record new second
            current_timestamp = np.floor(row[0])
            accumulated_data = row[1:] # row[1:7]

            n_similar_timestamp = 1.0

    downsampled_data = np.asarray(downsampled_data).reshape(-1,n_cols)

    return downsampled_data

def update(i):
    # global curr_timestamp
    curr_timestamp = combined_timestamps[i]
    time_data = datetime.fromtimestamp(int(curr_timestamp), CT_time_zone)
    ax1.clear()
    # ax1.set_title(f'#{i} : ' + str(time_data))
    ax1.set_title(str(time_data)[0:19] + f"  {int(curr_timestamp):d}")

    ## draw the cow
    global uwb_locs, prior_rest_y_loc_list, lying_refs, cam_coord, hide_ticks
    uwb_locs = np.empty((0, 3))
    # prior_rest_y_loc_list = []
    lying_refs = []
    for cow_idx, cow_data in enumerate(cow_data_list):
        timestamp       = cow_data[i,0]
        uwb_loc         = cow_data[i,1:4] 
        roll, pitch, yaw = cow_data[i,4:7]
        relative_angle  = cow_data[i,7]
        rest_ref        = cow_data[i,8] 
        CBT             = cow_data[i,9]

        uwb_locs = np.concatenate((uwb_locs, uwb_loc.reshape(1,3)), axis=0) # in cm
        # prior_rest_y_loc_list.append(prior_rest_y_loc)
        lying_refs.append(rest_ref)

        cow_loc = uwb_loc/100
        
        cow_id = cow_idx + 1
        # if cow_id != -1:

        if rest_ref == 0:
            neck_color = 'b'
        else:
            neck_color = 'g'

        if CBT > 39:
            head_color = rgb2hex(red)
        elif CBT >= 38.75:
            head_color = rgb2hex(medium_red)
        elif CBT >= 38.5:
            head_color = rgb2hex(medium_brown)
        elif CBT >= 38.25:
            head_color = rgb2hex(light_brown)
        else:
            head_color = rgb2hex(brown)
        
        # The cow
        if np.isnan(cow_loc[0]) == False:
            ax1.scatter(cow_loc[0], cow_loc[1], cow_loc[2], marker='o', c=neck_color, s=50)
            ax1.text(cow_loc[0], cow_loc[1], cow_loc[2] + 0.3, f'{cow_id}', fontsize=16, color='b', ha='center', va='bottom')

            if relative_angle > 12:
                
                phi = np.deg2rad(roll)
                theta = np.deg2rad(pitch)
                psi = np.deg2rad(yaw)

                # phi_temp = np.copy(phi)
                # theta_temp = np.copy(theta)

                # phi = - theta_temp
                # if phi_temp >= 0:
                #     theta = phi_temp - math.pi
                # else:
                #     theta = phi_temp + math.pi
                
                # psi = psi - math.pi/2
                # if psi < - math.pi:
                #     psi = psi + math.pi*2

                # psi += north_offset_angle

                vector = rotate_vector(north_vector, theta, phi, -psi) * 1.3
                roll_vector = rotate_vector(north_vector, phi, theta, -psi + math.pi/2) * 0.5

                ## Plot compass
                ax1.plot([cow_loc[0], cow_loc[0] + vector[0]], [cow_loc[1], cow_loc[1] + vector[1]], [cow_loc[2], cow_loc[2] + vector[2]], color=head_color) 
                ax1.plot([cow_loc[0] - roll_vector[0], cow_loc[0] + roll_vector[0]], [cow_loc[1] - roll_vector[1], cow_loc[1] + roll_vector[1]], [cow_loc[2] - roll_vector[2], cow_loc[2] + roll_vector[2]], color=head_color) # Roll line
                ax1.plot([cow_loc[0] - roll_vector[0], cow_loc[0] + vector[0]], [cow_loc[1] - roll_vector[1], cow_loc[1] + vector[1]], [cow_loc[2] - roll_vector[2], cow_loc[2] + vector[2]], color=head_color) 
                ax1.plot([cow_loc[0] + vector[0], cow_loc[0] + roll_vector[0]], [cow_loc[1] + vector[1], cow_loc[1] + roll_vector[1]], [cow_loc[2] + vector[2], cow_loc[2] + roll_vector[2]], color=head_color)

    draw_pen(ax1, cam_coord, hide_ticks=hide_ticks)

    if showing_images == True:
        global img_1, img_2, img_3, img_4
        plot_images(ax2, first_imgs, curr_timestamp)
        fig2.canvas.draw_idle()


# ===============================================
""" Main program from here """
def main(args):

    print('Date: ' + str(args.date))

    global ax1, ax2, fig2, combined_timestamps, curr_timestamp, cow_data_list, masking
    global first_imgs, showing_images, en_uwb_points, en_ground_grid, label_resolution
    global standing_bbox_en, lying_bbox_en, cam_list, Proj_cam_list, image_dir, label_dir, uwb_label_dir
    global en_ground_boundary, disp_resolution, cam_coord, hide_ticks, plot_scale, color_bbox

    date = args.date
    showing_images = args.no_image
    en_uwb_points = args.uwb_points
    en_ground_grid = args.ground_grid
    en_ground_boundary = args.boundary
    frame_height = args.frame_height
    frame_ratio = args.frame_ratio
    frame_width = int(frame_ratio * frame_height)
    label_resolution = (frame_width, frame_width)
    masking = args.masking
    standing_bbox_en = args.bbox
    color_bbox = args.color_bbox
    hide_ticks= args.hide_ticks
    plot_scale = 1

    disp_resolution = (1920, 1200)
    # disp_resolution = (960, 600)

    cam_list = ['cam_1','cam_2','cam_3','cam_4']
    
    if args.freeze == True:
        run_status = False
    else:
        run_status = True


    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
    yaml_dir = os.path.join(current_dir, "path.yaml")
    
    with open(yaml_dir, 'r') as file:
        file_dirs = yaml.safe_load(file)

    sensor_data_dir = file_dirs['sensor_data_dir']
    visual_data_dir = file_dirs['visual_data_dir'] 
    image_dir = os.path.join(visual_data_dir, 'images')
    label_dir = os.path.join(visual_data_dir, 'labels', 'combined')
    proj_mat_dir = os.path.join(visual_data_dir, 'proj_mat')

    Proj_cam_list = read_projection_matrices(proj_mat_dir, date)
    cam_coord = []
    for idx in range(4):
        proj_mat = Proj_cam_list[idx]
        cam_coord.append(cal_cam_coord(proj_mat).astype(int))
        # print(f"Cam {idx+1} loc: {cal_cam_loc(proj_mat).astype(int)}")
    cam_coord = np.asarray(cam_coord).reshape((4,3))

    print("Combining the data...")
    combined_cow_list = []
    set_list = []
    for tag_idx in range(10):
        tag_name = f'T{tag_idx+1:02d}' 
        cow_name = f'C{tag_idx+1:02d}' 

        # Read location
        # timestmap x_m y_m z_m
        location_data_dir = os.path.join(sensor_data_dir, 'main_data', 'uwb', tag_name, tag_name + "_" + date + ".csv")
        loc_df = pd.read_csv(location_data_dir) 
        locations = loc_df[['coord_x_cm', 'coord_y_cm', 'coord_z_cm']].values
        loc_timestamp = loc_df[['timestamp']].values.flatten()

        # Read head direction
        # timestamp	roll	pitch	yaw	accel_norm	mag_norm	relative_angle
        input_file_dir = os.path.join(sensor_data_dir, 'sub_data', 'head_direction', tag_name, tag_name + "_"+ date + ".csv")
        heading_data = pd.read_csv(input_file_dir).values
        heading_data = downsampling_by_second(heading_data)

        # Read lying reference
        input_file_dir = os.path.join(sensor_data_dir, 'main_data', 'ankle', cow_name, cow_name + "_"+ date +  ".csv")
        data = pd.read_csv(input_file_dir).values
        lying_ref = np.column_stack((data[:,0], data[:,2]))

        # Read CBT
        input_file_dir = os.path.join(sensor_data_dir, 'main_data', 'cbt', cow_name + ".csv")
        CBT_data = pd.read_csv(input_file_dir).values

        temp_sel_data = []
        heading_sel_data = []
        lying_sel_data = []
        THI_sel_data = []

        for single_loc_timestamp in loc_timestamp:
            temp_sel_data.append(find_closest_row(single_loc_timestamp, CBT_data))
            heading_sel_data.append(find_closest_row(single_loc_timestamp, heading_data))
            lying_sel_data.append(find_closest_row(single_loc_timestamp, lying_ref))

        temp_sel_data = np.asarray(temp_sel_data)
        heading_sel_data = np.asarray(heading_sel_data)
        lying_sel_data = np.asarray(lying_sel_data)
        THI_sel_data = np.asarray(THI_sel_data)

        temperature_data = temp_sel_data[:, 1]
        heading_data = heading_sel_data[:,1:4]
        relative_angle = heading_sel_data[:, -1]
        lying_data = lying_sel_data[:,1]

        data = np.column_stack((loc_timestamp, 
                                       locations, 
                                       heading_data, 
                                       relative_angle, 
                                       lying_data,
                                       temperature_data
                                       ))

        ## Limiting the location range when the cow is eating
        for idx, uwb_loc in enumerate(data[:,1:4]):
            # uwb_loc = uwb_loc*100
            uwb_loc[2] = max(0, uwb_loc[2]) # z
            uwb_loc[2] = min(uwb_loc[2], 250) # z
            if uwb_loc[1] < - s_y:
                uwb_loc[1] = max(- s_y - 40, uwb_loc[1]) # y: lower bound of a negative value
                uwb_loc[2] = min(uwb_loc[2], 130) # z: upper bound of a positive value
            # print(f"{int(cow_loc[1]*100)} {int(uwb_loc[1])} {uwb_loc.astype(int)}")
            data[idx, 1:4] = uwb_loc
        
        ## Smooth out uwb locaiton
        # Apply a moving average filter for smoothing
        window_size = 3
        data[:,1] = np.convolve(data[:,1], np.ones(window_size)/window_size, mode='same')
        data[:,2] = np.convolve(data[:,2], np.ones(window_size)/window_size, mode='same')
        data[:,3] = np.convolve(data[:,3], np.ones(window_size)/window_size, mode='same')

        # if tag_id < 6:
        #     data[:,0] += 1
        if args.print_out == True:
            print(len(data[:,0]))
            # print((data[-1,0]))

        set_list.append(data[:,0])
        combined_cow_list.append(data)

    timestamp_list = list(set(set_list[0]).union(set_list[1], set_list[2], set_list[3], set_list[4], set_list[5], set_list[6], set_list[7], set_list[8], set_list[9]))
    timestamp_list.sort() # must sort here
    combined_timestamps = np.asarray(timestamp_list)

    if args.print_out == True:
        print('Combined: ' + str(len(combined_timestamps)))

    cow_data_list = []
    for tag_data in combined_cow_list:
        cow_data_list.append(interpolate_missing_row(combined_timestamps, tag_data))

    ## Initialize the first figure and a 3D axis
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    # plt.tight_layout() # Do not use

    ## Initialize the second figure
    if showing_images == True:
        fig2, ax2 = plt.subplots()
        curr_timestamp = 0
        first_imgs = gen_black_images(disp_resolution)
        first_imgs = first_images(ax2, first_imgs, curr_timestamp)
        plt.tight_layout()

    ani = Player(fig=fig1, func=update, maxi=len(combined_timestamps)-1, run_status=run_status, interval_ms=args.disp_intv) 

    plt.show()

# --date 0725 --uwb_points --bbox --boundary --freeze
# --date 0725 --bbox --boundary

if __name__ == '__main__':

    # print("Tips: Use --help for detailed configurations")

    date_list = ["0721", "0722", "0723", "0724","0725","0726","0727","0728","0729","0730","0731","0801","0802", "0803","0804"]
    
    # settings
    parser = argparse.ArgumentParser(description='CowLoc visualization')
    parser.add_argument('--freeze', action='store_true', help='Stop the annimation at run. Use if the annimation is laggy')
    parser.add_argument('--date', type=str, default='0725', choices=date_list, help='The chosen date to be visualized in MMDD')
    parser.add_argument('--no_image', action='store_false', help='Disabling the second window that displays the images')
    parser.add_argument('--print_out', action='store_true', help='N/A')
    parser.add_argument('--uwb_points', action='store_true', help='Showing 3D UWB locations in the camera views')
    parser.add_argument('--ground_grid', action='store_true', help='Showing the ground grid and the pen boundary in the camera views')
    parser.add_argument('--boundary', action='store_true', help='Showing the pen boundary in the camera views')
    parser.add_argument('--masking', action='store_true', help='Masking the view from other pens')
    parser.add_argument('--bbox', action='store_true', help='Drawing bounding boxes from the cow ID labels')
    parser.add_argument('--color_bbox', action='store_true', help='Drawing bounding boxes from the cow ID labels with unique colors')
    parser.add_argument('--hide_ticks', action='store_true', help='Hide x, y, and z ticks')
    parser.add_argument('--frame_ratio', type=float, default=1.6, help='Ratio of the frame being displayed')
    parser.add_argument('--frame_height', type=int, default=2800, help='Height of the frame being displayed')
    parser.add_argument('--disp_intv', type=int, default=3000, help='Set display interval of the animation') 

    # north_vector = np.array([3, 9, 0]).astype(float)
    north_vector = np.array([0, 9, 0]).astype(float)
    # north_offset_angle = math.atan2(north_vector[0], north_vector[1])
    radius = np.linalg.norm(north_vector)
    north_vector /= radius

    args = parser.parse_args()

    main(args)


