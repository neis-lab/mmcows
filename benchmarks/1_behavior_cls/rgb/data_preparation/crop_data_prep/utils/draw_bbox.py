
import os
import numpy as np
import cv2
import zipfile
from io import TextIOWrapper
from PIL import Image
from utils.projection import *
import math
import warnings

np.set_printoptions(suppress=True)

c_green = (0, 255, 0)
c_red = (255, 0, 0)
c_blue = (0, 0, 255)
c_white = (255, 255, 255)
c_aqua = (0, 255, 255)
c_black = (0, 0, 0)

# Define colors in BGR format
colors_bgr = [
    (128, 128, 128),  # grey
    (255, 0, 0),      # blue
    (0, 255, 0),      # green
    (0, 0, 255),      # red
    (0, 165, 255),    # orange
    (128, 0, 128),    # purple
    # (0, 0, 128),      # maroon
    (180, 105, 255),  # hotpink
    (51, 169, 0),     # darkgreen
    (255, 255, 0),     # aqua
    (128, 128, 128),  # grey
    (255, 0, 0),      # blue
    (0, 255, 0),      # green
    (0, 0, 255),      # red
    (0, 165, 255),    # orange
    (128, 0, 128),    # purple
    # (0, 0, 128),      # maroon
    (180, 105, 255),  # hotpink
    (51, 169, 0),     # darkgreen
    (255, 255, 0)     # aqua
]

text_thickness = 2

s_y = 646

meter_scale = 1 # cm
# meter_scale = 1/100 # m

min_x, max_x = -879, 1042
min_y, max_y = -646, 533

# def _read_txt_to_numpy(file_path):
#     try:
#         data_array = np.loadtxt(file_path)
#         return data_array
#     except Exception as e:
#         print(f"Error reading the file: {e}")
#         return None

# def _sort_rows_by_column(arr, column_idx):
#     """Sort the rows of a 2D numpy array by the values in the second column in decreasing order."""
#     # Get the indices that would sort the array by the second column in decreasing order
#     sorted_indices = np.argsort(arr[:, column_idx])[::-1]
#     # Sort the array based on the sorted indices
#     sorted_arr = arr[sorted_indices]
#     return sorted_arr

# def _distance(point1, point2):
#     """Calculate the Euclidean distance between two points."""
#     return np.linalg.norm(point1 - point2)

def _draw_rect_bbox(image, corner1, corner2, color=(0, 255, 0), thickness=2):
    # img_copy = image.copy()
    corner1 = tuple(map(int, corner1))
    corner2 = tuple(map(int, corner2))
    cv2.rectangle(image, corner1, corner2, color, thickness)
    return image

def _annotate_dot(image, center, radius=5, color=(0, 255, 0), thickness=-1):
    # img_copy = image.copy()
    center = tuple(map(int, center))
    cv2.circle(image, center, radius, color, thickness)
    return image

def _annotate_points(image, pixel_locations, radius=5, color=(0, 255, 0), thickness=-1):
    # img_copy = image.copy()
    # Iterate through pixel locations and draw circles
    for pixel in pixel_locations:
        center = tuple(map(int, pixel))
        cv2.circle(image, center, radius, color, thickness)
    return image

def _annotate_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 255, 0), thickness=2):
    # img_copy = image.copy()
    position = tuple(map(int, position))
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    return image

def _draw_rectangle(image, top_left, bottom_right, color=(255, 255, 255), alpha=0.1):
    # Draw a partially transparent rectangle on the image
    cv2.rectangle(image, top_left, bottom_right, (*color, int(255 * alpha)), thickness=cv2.FILLED)

def ratio_to_pixel(disp_resolution, xyn_ratio):
    # down_scale = label_resolution[0]/disp_resolution[0]
    width_loc = int(disp_resolution[0] * xyn_ratio[0])
    height_loc = int(disp_resolution[1] * xyn_ratio[1])
    return width_loc, height_loc

def pixel_to_ratio(res_width_height, xy_loc):
    width_ratio = xy_loc[0]/ res_width_height[0]
    height_ratio = xy_loc[1] / res_width_height[1]
    return width_ratio, height_ratio

def _read_zip_text_file(zip_filename, text_filename):
    with zipfile.ZipFile(zip_filename, 'r') as zip_file:
        with zip_file.open(text_filename) as text_file:
            # Use TextIOWrapper to decode the bytes into text
            text_content = TextIOWrapper(text_file, encoding='utf-8').readlines()

    # Assuming the data is space-separated in a tabular format
    # Convert each line to a list of elements and then to a NumPy array
    data_array = np.array([line.split() for line in text_content], dtype=float)

    return data_array

def read_bbox_labels(text_file_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_array = np.loadtxt(text_file_dir)
    # print(type(data_array))
    # print(np.shape(data_array))
    return np.atleast_2d(data_array)

# ==============================================================================

def draw_cam_name(resolution, img, idx): 
    img = np.array(img)
    width, height = resolution
    # x = int(0.05 * width)
    # y = int(0.9 * height)
    x = int(0.007 * width)
    y = int(0.94 * height)
    rectangle_coords = ((x, y), (x+int(width*0.12), y+int(height*0.05)))
    text_coords = (x+int(width*0.0052), y+int(height*0.04166))

    # Call the draw_rectangle function
    _draw_rectangle(img, *rectangle_coords, color=c_white)
    img = _annotate_text(img, f"Cam {idx+1:d}", text_coords, font_scale=2, color=c_black, thickness=int(4))
    return img

## bbox_data [idx] [xyxy] [yx] [conf]
def _draw_bbox(img, bbox_data, dconf = False, color_bbox = False): 
    img = np.array(img)
    idx = int(bbox_data[0])
    corner1_x, corner1_y  = bbox_data[1:3]
    corner2_x, corner2_y = bbox_data[3:5]
    # dot_x, dot_y = bbox_data[5:7]
    # conf = bbox_data[7]

    if dconf == False:
        text_content = str(idx) # bbox without conf
    # else:
        # text_content = f"{idx:d} {conf:.2f}" # bbox with conf

    if color_bbox == False:
        color = c_green
    else:
        color = colors_bgr[int(idx)-1]

    img = _draw_rect_bbox(img, (corner1_x, corner1_y), (corner2_x, corner2_y), color=color, thickness=int(2)) # bbox
    img = _annotate_text(img, text_content, (corner1_x, corner1_y-5), font_scale=1.2, color=color, thickness=int(text_thickness)) # bbox text

    # img = _annotate_dot(img, (dot_x, dot_y), radius=int(5), color=c_green, thickness=3) # keypoint
    # img = _annotate_text(img, str(idx), (dot_x-10, dot_y-10), font_scale=1.2, color=c_green, thickness=int(text_thickness)) # keypoint

    return Image.fromarray(img)

def draw_standing_bbox(img, matched_bbox_data, dconf=False, color_bbox=False): 
    img = np.array(img)
    if matched_bbox_data.size > 0:

        for bbox_data in matched_bbox_data:
            ## bbox_data [idx] [xyxy] [yx] [conf] [xyz coord]
            img = _draw_bbox(img, bbox_data, dconf=dconf, color_bbox=color_bbox)
    return img

# convert bbox from [x_center, y_center, width, height] to [x1, y1, x2, y2]
def convert_bbox_format(bbox_center, width_height):
    x_center, y_center = bbox_center
    width, height = width_height
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return [x1, y1, x2, y2]

# #  python3.11 /Users/mac/Library/CloudStorage/OneDrive-UW-Madison/cps_data_exp1/uwb_localization/cowloc_view.py --original_res --uwb_bbox --boundary
# def draw_lying_bbox(resolution, 
#                         img, 
#                         Proj_mat,
#                         uwb_locs, 
#                         prior_rest_y_loc_list, 
#                         standing_refs,
#                         cam_id): 
#     img = np.array(img)
#     resolution = np.array(resolution)
#     prior_rest_y_loc_list = np.array(prior_rest_y_loc_list)
#     # print(f"uwb_locs {np.shape(uwb_locs)}") 
#     # print(f"prior_rest_y_loc {np.shape(prior_rest_y_loc_list)}")
#     uwb_locs[:,1] = prior_rest_y_loc_list
#     cow_pixel_locs = project_world2imgage(uwb_locs.T, Proj_mat)

#     if uwb_locs.size > 0:
#         for idx, cow_pixel_loc, cow_coord, stand_ref in zip(range(10), cow_pixel_locs.T, uwb_locs, standing_refs): # Each point
#             if stand_ref == 0 and np.isnan(cow_coord[0]) == False:
#                 dot_x, dot_y = cow_pixel_loc 
#                 x_loc, y_loc, _ = cow_coord
    
#                 if y_loc > -150 and y_loc < 150 \
#                     and x_loc > -500 and x_loc < 700: # Account for false positive standing_ref
                    
#                     bbox_x1 = 200
#                     bbox_x2 = 70
#                     bbox_y1 = 80
#                     bbox_y2 = 15
#                     if y_loc > 0: # North
#                         bbox_loc_cor_tn = (x_loc, y_loc + bbox_x1, bbox_y2+bbox_y1)
#                         bbox_loc_cor_bn = (x_loc, y_loc + bbox_x1, bbox_y2)
#                         bbox_loc_cor_ts = (x_loc, y_loc - bbox_x2, bbox_y2+bbox_y1)
#                         bbox_loc_cor_bs = (x_loc, y_loc - bbox_x2, bbox_y2)
#                     else: # South
#                         bbox_loc_cor_tn = (x_loc, y_loc + bbox_x2, bbox_y2+bbox_y1)
#                         bbox_loc_cor_bn = (x_loc, y_loc + bbox_x2, bbox_y2)
#                         bbox_loc_cor_ts = (x_loc, y_loc - bbox_x1, bbox_y2+bbox_y1)
#                         bbox_loc_cor_bs = (x_loc, y_loc - bbox_x1, bbox_y2)
#                     bbox_cor_pixel_loc = project_world2imgage(np.vstack((bbox_loc_cor_tn, 
#                                                                          bbox_loc_cor_bn,
#                                                                          bbox_loc_cor_ts,
#                                                                          bbox_loc_cor_bs
#                                                                          )).T, Proj_mat)

#                     bbox_cor_pixel_loc = bbox_cor_pixel_loc.T
#                     bbox_cor_pixel_loc_tn = bbox_cor_pixel_loc[0,:]
#                     bbox_cor_pixel_loc_bn = bbox_cor_pixel_loc[1,:]
#                     bbox_cor_pixel_loc_ts = bbox_cor_pixel_loc[2,:]
#                     bbox_cor_pixel_loc_bs = bbox_cor_pixel_loc[3,:]
#                     if cam_id == 1:
#                         corner1_x, corner1_y = bbox_cor_pixel_loc_ts
#                         corner2_x, corner2_y = bbox_cor_pixel_loc_bn
#                     if cam_id == 2:
#                         corner1_x, corner1_y = bbox_cor_pixel_loc_ts
#                         corner2_x, corner2_y = bbox_cor_pixel_loc_bn
#                     if cam_id == 3:
#                         corner1_x, corner1_y = bbox_cor_pixel_loc_tn
#                         corner2_x, corner2_y = bbox_cor_pixel_loc_bs
#                     if cam_id == 4:
#                         corner1_x, corner1_y = bbox_cor_pixel_loc_tn
#                         corner2_x, corner2_y = bbox_cor_pixel_loc_bs

#                     img = _draw_rect_bbox(img, (corner1_x, corner1_y), (corner2_x, corner2_y), color=c_aqua, thickness=int(2)) # bbox
#                     img = _annotate_dot(img, (dot_x, dot_y), radius=int(5), color=c_aqua, thickness=3)
#                     # img = _annotate_text(img, str(idx), (dot_x-10, dot_y-15), font_scale=1, color=(255, 0, 0), thickness=3)
#                     img = _annotate_text(img, str(idx+1), (corner1_x, corner1_y-5), font_scale=1.2, color=c_aqua, thickness=int(text_thickness)) # bbox

#     return img

def draw_uwb_points(img, uwb_locs, Proj_mat, disp_scale): 
    img = np.array(img)
    uwb_loc_coor = project_world2imgage(uwb_locs.T, Proj_mat)
    uwb_loc_coor = (uwb_loc_coor / disp_scale).astype(int)
    for i, loc in enumerate(uwb_loc_coor.T):
        if np.isnan(loc[0]) == False:
            img = _annotate_dot(img, loc, radius=int(5), color=c_red, thickness=3)
            img = _annotate_text(img, str(i+1), (loc[0]-10, loc[1]-10), font_scale=1.2, color=c_red, thickness=int(text_thickness))
    return img

def draw_ground_grid(img, Proj_mat, disp_scale): 
    img = np.array(img)
    world_grid = generate_ground_grid()
    img_coor_grid = project_world2imgage(world_grid, Proj_mat)
    img_coor_grid = (img_coor_grid / disp_scale).astype(int)
    img = _annotate_points(img, img_coor_grid.T, radius=int(3), color=c_white, thickness=1)
    return img

def draw_ground_boundary(img, Proj_mat, disp_scale): 
    img = np.array(img)
    world_grid = generate_ground_boundary()
    img_coor_grid = project_world2imgage(world_grid, Proj_mat)
    img_coor_grid = (img_coor_grid / disp_scale).astype(int)
    img = _annotate_points(img, img_coor_grid.T, radius=int(5), color=c_white, thickness=1)
    return img

def generate_ground_boundary(meter_scale=1):
    # Define the ranges for x, y, and z
    xi = np.arange(min_x, max_x+1, 10*meter_scale)
    yi = np.arange(min_y, max_y+1, 10*meter_scale)

    # Create a meshgrid
    x, y = np.meshgrid(xi, yi)

    # Specify the rectangular boundary
    x_min, x_max = min_x+10*meter_scale, max_x-10*meter_scale
    y_min, y_max = min_y+10*meter_scale, max_y-10*meter_scale

    # Mask the points outside the rectangular boundary
    mask = np.logical_or(np.logical_or(x < x_min, x > x_max), np.logical_or(y < y_min, y > y_max))

    # Extract points inside the rectangular boundary
    x_rectangular = x[mask]
    y_rectangular = y[mask]

    # Set z value
    z_value = 0
    z_rectangular = np.full_like(x_rectangular, z_value)

    # Create world_grid by stacking x, y, and z
    world_grid = np.vstack([x_rectangular.flatten(), y_rectangular.flatten(), z_rectangular.flatten()])

    return world_grid

def generate_ground_grid(meter_scale=1):
    xi = np.arange(min_x, max_x+1, 50*meter_scale)
    yi = np.arange(min_y, max_y+1, 50*meter_scale)
    world_grid = np.stack(np.meshgrid(xi, yi, indexing='ij')).reshape([2, -1])
    world_grid = np.vstack([world_grid, np.zeros(world_grid.shape[1])])
    return world_grid

# def _generate_ground_grid():
#     # Define the ranges for x, y, and z
#     xi = np.arange(-1000, 1001, 50)
#     yi = np.arange(-600, 601, 50)

#     # Create a meshgrid
#     x, y = np.meshgrid(xi, yi)

#     # Set z value
#     z_value = 0
#     z_rectangular = np.full_like(x, z_value)

#     # Create world_grid by stacking x, y, and z
#     world_grid = np.vstack([x.flatten(), y.flatten(), z_rectangular.flatten()])

#     return world_grid

# def _generate_ground_boundary():
#     # Define the ranges for x, y, and z
#     xi = np.arange(-1000, 1001, 10)
#     yi = np.arange(-600, 601, 10)

#     # Create a meshgrid
#     x, y = np.meshgrid(xi, yi)

#     # Specify the rectangular boundary
#     x_min, x_max = -1000+5, 1001-5
#     y_min, y_max = -601+5, 601-5

#     # Mask the points outside the rectangular boundary
#     mask = np.logical_or(np.logical_or(x < x_min, x > x_max), np.logical_or(y < y_min, y > y_max))

#     # Extract points inside the rectangular boundary
#     x_rectangular = x[mask]
#     y_rectangular = y[mask]

#     # Set z value
#     z_value = 0
#     z_rectangular = np.full_like(x_rectangular, z_value)

#     # Create world_grid by stacking x, y, and z
#     world_grid = np.vstack([x_rectangular.flatten(), y_rectangular.flatten(), z_rectangular.flatten()])

#     return world_grid


    

    