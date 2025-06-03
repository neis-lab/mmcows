import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
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

    save_to_xml(projection_matrix, cam_mat, 'projection_matrix', 'camera_matrix', output_path + '/proj_mat_cam'+str(cam_id)+'.xml')
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



