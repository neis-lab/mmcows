
# import os
import numpy as np
# import cv2
# import zipfile
# from io import TextIOWrapper
# from PIL import Image
from utils.projection import *
# import math
import warnings

def read_bbox_labels(text_file_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_array = np.loadtxt(text_file_dir)
    # print(type(data_array))
    # print(np.shape(data_array))
    return np.atleast_2d(data_array)