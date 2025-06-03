import cv2
import numpy as np
import xml.etree.ElementTree as ET
# from xml.dom import minidom
import xml.dom.minidom


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def save_to_xml(var1, var2, var1_name, var2_name, filename):
    root = ET.Element("opencv_storage")

    # Save var1
    var1_element = ET.SubElement(root, var1_name, type_id="opencv-matrix")
    add_matrix_data(var1_element, var1)

    # Save var2
    var2_element = ET.SubElement(root, var2_name, type_id="opencv-matrix")
    add_matrix_data(var2_element, var2)

    # Save the indented XML to the file
    with open(filename, "w") as f:
        f.write(prettify(root))

def add_matrix_data(parent_element, matrix):
    rows, cols = matrix.shape

    rows_element = ET.SubElement(parent_element, "rows")
    rows_element.text = str(rows)

    cols_element = ET.SubElement(parent_element, "cols")
    cols_element.text = str(cols)

    dt_element = ET.SubElement(parent_element, "dt")
    dt_element.text = "d"

    data_element = ET.SubElement(parent_element, "data")
    data_element.text = " ".join(format(num, ".16e") for num in matrix.flatten())

def save_cal_mat_opencv(output_path, cam_id, retval, camera_matrix, dist_coeffs, rvecs, tvecs):
    f = cv2.FileStorage(output_path + f'/intr_Camera{cam_id}.xml', flags=cv2.FILE_STORAGE_WRITE)
    f.write(name='camera_matrix', val=camera_matrix)
    f.write(name='distortion_coefficients', val=dist_coeffs)
    f.release()
    f = cv2.FileStorage(output_path + f'/extr_Camera{cam_id}.xml', flags=cv2.FILE_STORAGE_WRITE)
    f.write(name='rvec', val=rvecs[0])
    f.write(name='tvec', val=tvecs[0])
    f.release()
    print('mat saved')

# def export_cal_mat_opencv(output_path, cam_id, retval, camera_matrix, dist_coeffs, rvecs, tvecs):
#     print('save ---')

#     # Decompose the projection matrix
#     camera_matrix = camera_matrix
#     rvec = rvecs[0]
#     tvec = tvecs[0]
#     distortion_coefficients = dist_coeffs

#     intrinsic_matrix = camera_matrix
#     # print(f"Intrinsic:\n{intrinsic_matrix}")

#     rvec1, _ = cv2.Rodrigues(rvec)
#     # Construct extrinsic matrix [R | t]
#     extrinsic_matrix = np.hstack((rvec1, tvec))
#     # print(f"Extrinsic:\n{extrinsic_matrix}")

#     projection_matrix = intrinsic_matrix @ extrinsic_matrix
#     # print(f"Projection:\n{M}")

#     save_to_xml(projection_matrix, np.array([[0,0],[0,0]]), 'projection_matrix', 'dummy', output_path + '/projection_matrix_cam'+str(cam_id)+'.xml')
#     save_to_xml(rvec, tvec, 'rvec', 'tvec', output_path + '/extr_Camera'+str(cam_id)+'.xml')
#     save_to_xml(camera_matrix, distortion_coefficients.T, 'camera_matrix', 'distortion_coefficients', output_path + '/intr_Camera'+str(cam_id)+'.xml')
#     save_to_xml(intrinsic_matrix, extrinsic_matrix, 'intrinsic_matrix', 'extrinsic_matrix', output_path + '/intr_extr_cam'+str(cam_id)+'.xml')

#     # return M
#     return intrinsic_matrix, extrinsic_matrix
#     # return projection_matrix

# def export_cal_mat_mvdet(output_path, projection_matrix, cam_id):
#     print('save ---')
#     print(f"Original Projection:\n{projection_matrix}")

#     # Decompose the projection matrix
#     camera_matrix, rvec1, tvec, _, _, _, distortion_coefficients = cv2.decomposeProjectionMatrix(projection_matrix)

#     tvec = tvec[:3] / tvec[3] * (-1) # HV: added (-1) to get similar result to MVDet
#     intrinsic_matrix = camera_matrix
#     print(f"Intrinsic:\n{intrinsic_matrix}")

#     # Construct extrinsic matrix [R | t]
#     extrinsic_matrix = np.hstack((rvec1, tvec))
#     print(f"Extrinsic:\n{extrinsic_matrix}")

#     M = intrinsic_matrix @ extrinsic_matrix
#     print(f"Projection mat for MVDet:\n{M}")

#     # Step 1: Encode using cv2.Rodrigues
#     rvec, _ = cv2.Rodrigues(rvec1)
#     # print(rvec)

#     # Step 2: Decode and recover using cv2.Rodrigues
#     # recovered_matrix, _ = cv2.Rodrigues(rvec)
#     # print(recovered_matrix)

#     save_to_xml(projection_matrix, np.array([[0,0],[0,0]]), 'projection_matrix', 'dummy', output_path + '/proj_mat_cam'+str(cam_id)+'.xml')
#     save_to_xml(rvec, tvec, 'rvec', 'tvec', output_path + '/extr_Camera'+str(cam_id)+'.xml')
#     save_to_xml(camera_matrix, distortion_coefficients.T, 'camera_matrix', 'distortion_coefficients', output_path + '/intr_Camera'+str(cam_id)+'.xml')

#     # return M
#     return intrinsic_matrix, extrinsic_matrix
#     # return projection_matrix

def export_cal_mat(output_path, projection_matrix, cam_id):
    # Decomposition of P = KR[I|-X0] = KRI|-KRX0 = H|h
    H = projection_matrix[:,:3] # H = KR

    # Getting Projection Center X0 = -H^-1 h
    H_inv = np.linalg.inv(H)

    # QR Decomposition of H^-1 gives Rotation matrix and K(Camera) matrix
    rot_inv, k_inv = np.linalg.qr(H_inv)
    rmat = rot_inv.T / rot_inv.T[-1,-1]

    k = np.linalg.inv(k_inv)
    cam_mat = k / k[-1,-1]

    # tvec = tvec[:3] / tvec[3] * (-1) # HV: added (-1) to get similar result to MVDet
    
    # intrinsic_matrix = cam_mat
    # print(f"Intrinsic:\n{intrinsic_matrix}")

    # # Construct extrinsic matrix [R | t]
    # extrinsic_matrix = np.hstack((rmat, tvec))
    # print(f"Extrinsic:\n{extrinsic_matrix}")

    # M = intrinsic_matrix @ extrinsic_matrix
    # print(f"Projection mat for MVDet:\n{M}")

    # Step 1: Encode using cv2.Rodrigues
    # rvec, _ = cv2.Rodrigues(rmat)
    # print(rvec)

    # Step 2: Decode and recover using cv2.Rodrigues
    # recovered_matrix, _ = cv2.Rodrigues(rvec)
    # print(recovered_matrix)

    save_to_xml(projection_matrix, cam_mat, 'projection_matrix', 'camera_matrix', output_path + '/proj_mat_cam'+str(cam_id)+'.xml')
    # save_to_xml(rvec, tvec, 'rvec', 'tvec', output_path + '/extr_Camera'+str(cam_id)+'.xml')
    # save_to_xml(cam_mat, distortion_coefficients.T, 'camera_matrix', 'distortion_coefficients', output_path + '/intr_Camera'+str(cam_id)+'.xml')

    print('mat saved')

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

# ======================================================
    
# Function to add matrix to the XML structure
def add_matrix(parent, name, matrix):
    matrix_elem = ET.SubElement(parent, name)
    rows, cols = matrix.shape
    matrix_elem.set("rows", str(rows))
    matrix_elem.set("cols", str(cols))
    matrix_elem.text = " ".join(map(str, matrix.flatten()))

# def export_cal_mat(output_path, projection_matrix, cam_id):
#     # Decompose the projection matrix
#     camera_matrix, rotation_matrix, translation_matrix, _, _, _, distortion_coefficients = cv2.decomposeProjectionMatrix(projection_matrix)

#     translation_matrix = translation_matrix[:3] / translation_matrix[3]

#     # Create an XML structure
#     root = ET.Element("opencv_storage")

#     # Add matrices to the XML structure
#     add_matrix(root, "projection_matrix", projection_matrix)
#     add_matrix(root, "camera_matrix", camera_matrix)
#     add_matrix(root, "rotation_matrix", rotation_matrix)
#     add_matrix(root, "translation_matrix", translation_matrix)
#     add_matrix(root, "distortion_coefficients", distortion_coefficients)

#     # Create and save the XML file with proper indentation
#     tree = ET.ElementTree(root)
#     xml_str = ET.tostring(root, encoding="utf-8")
#     xml_pretty_str = minidom.parseString(xml_str).toprettyxml(indent="  ")  # Adjust the indentation as needed

#     with open(output_path + '/cal_mat_cam'+str(cam_id)+'.xml', "w") as xml_file:
#         xml_file.write(xml_pretty_str)

    # np.savez(current_dir + "/camera_calib_matrices/calibration_matrices_cam" + str(cam_id) + ".npz", 
    #          projection_matrix=projection_matrix, 
    #          camera_matrix=camera_matrix, 
    #          rotation_matrix=rotation_matrix, 
    #          translation_matrix=translation_matrix)
        
# ======================================================
        
# current_dir = os.path.join(os.path.dirname(__file__)) # Folder

# # Barn cam 1
# cam_id = 1
# # projection_matrix = np.array(
# # [[-3.18398598e-05,  1.32439515e-03,  1.68915503e-04, -9.00846863e-01],
# #  [ 7.10132472e-05,  7.11002707e-05,  1.09689491e-03, -4.34131422e-01],
# #  [-6.99795537e-07,  4.43792219e-07,  2.99750732e-07, -1.35711776e-03]])
# # recombined
# projection_matrix = np.array(
# [[ 3.61323280e+01, -1.50188901e+03, -1.91377817e+02,  1.02153371e+06],
#  [-8.05868475e+01, -8.04653928e+01, -1.24404266e+03,  4.92202939e+05],
#  [ 7.94137974e-01, -5.03232145e-01, -3.39898262e-01,  1.53972918e+03]])
        
if __name__ == "__main__":
    rvec = np.array([
        [-1.2615197302646406e+00],
        [4.1726883239727519e-01],
        [5.4379507535231220e-01]
    ])

    tvec = np.array([
        [-3.0791218013940238e+00],
        [-2.9619545805836935e+00],
        [2.5540128107375155e+00]
    ])

    save_to_xml(rvec, tvec, 'tem1', 'tem2', "output.xml")



