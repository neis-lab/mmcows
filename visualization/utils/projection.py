
import numpy as np
import math

import os

current_dir = os.path.join(os.path.dirname(__file__)) # Folder

    
def project_point2image(points, K):
    '''
    Compute projection of points onto the image plane
    
    Parameters
    -----------
    points - np.ndarray, shape - (3, n_points)
        points we want to project onto the image plane
        the points should be represented in the camera coordinate system
    K - np.ndarray, shape - (3, 3)
        camera intrinsic matrix
        
    Returns
    -------
    points_i - np.ndarray, shape - (2, n_points)
        the projected points on the image
    '''
        
    h_points_i = K @ points
    
    h_points_i[0, :] = h_points_i[0, :] / h_points_i[2, :]
    h_points_i[1, :] = h_points_i[1, :] / h_points_i[2, :]

    points_i = h_points_i[:2, :]    
    
    return points_i


def project_world2imgage(world_points, Proj_mat):
    # convert to homogeneous coordinates
    points_h = np.vstack((world_points, np.ones(world_points.shape[1])))

    h_points_i = Proj_mat @ points_h

    h_points_i[0, :] = h_points_i[0, :] / h_points_i[2, :]
    h_points_i[1, :] = h_points_i[1, :] / h_points_i[2, :]
    points_i = h_points_i[:2, :]    
    return points_i

def project_image2world(image_coord, Proj_mat, Z):
    # ref: https://stackoverflow.com/questions/53394418/2d-to-3d-projection-with-given-z-in-world
    u, v = image_coord[0], image_coord[1]
    m11, m12, m13, m14 = Proj_mat[0,:]
    m21, m22, m23, m24 = Proj_mat[1,:]
    m31, m32, m33, m34 = Proj_mat[2,:]
    A = (m12-m32*u)/(m22-m32*v)
    B = (m31*u-m11)/(m31*v-m21)
    X = (Z*((m23-m33*v)*A-m13+m33*u) + (m24-m34*v)*A-m14+m34*u ) / (A*(m31*v-m21)-m31*u+m11)
    Y = (Z*((m13-m33*u)-B*(m23-m33*v)) + m14-m34*u-B*(m24-m34*v)) / (B*(m22-m32*v)-m12+m32*u)
    return np.array([X, Y, Z])

def cal_cam_coord(proj_mat):
    ## Camera position
    # ref https://math.stackexchange.com/questions/2237994/back-projecting-pixel-to-3d-rays-in-world-coordinates-using-pseudoinverse-method?newreg=2dc63084a6d04f9d88bbb60dd8a78e35
    # Book Eq 6.13 http://www.r-5.org/files/books/computers/algo-list/image-processing/vision/Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision-EN.pdf
    M = proj_mat[:,0:3]
    p4 = proj_mat[:,3]
    camloc = - np.linalg.inv(M) @ p4
    return camloc


        