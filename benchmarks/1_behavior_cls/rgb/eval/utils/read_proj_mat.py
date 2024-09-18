# import cv2
import numpy as np
import xml.etree.ElementTree as ET
# from xml.dom import minidom
# import xml.dom.minidom


def read_cal_mat_xml(xml_file, name1, name2):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Read Projection Matrix
    projection_matrix_element = root.find(".//" + name1)
    rows = int(projection_matrix_element.find("rows").text)
    cols = int(projection_matrix_element.find("cols").text)
    data_text = projection_matrix_element.find("data").text
    projection_matrix = np.fromstring(data_text, sep=' ').reshape(rows, cols)

    # Read Dummy Matrix
    dummy_element = root.find(".//" + name2)
    rows = int(dummy_element.find("rows").text)
    cols = int(dummy_element.find("cols").text)
    data_text = dummy_element.find("data").text
    dummy_matrix = np.fromstring(data_text, sep=' ').reshape(rows, cols)

    return projection_matrix, dummy_matrix

def read_projection_matrices(data_dir, date = '0725'):
    P_cam_list = []
    for i in range(4):
        cam_id = i + 1
        try:
            file_dir = data_dir +'/' + date + "/proj_mat_cam" + str(cam_id) + ".xml"  
            P, _ = read_cal_mat_xml(file_dir, 'projection_matrix', 'camera_matrix') 
            P_cam_list.append(P)
        except:
            print('File not found: ' + file_dir)
            P_cam_list.append(np.ones((3,4)))
    return P_cam_list