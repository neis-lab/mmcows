import numpy as np
# from mpl_toolkits.mplot3d import Axes3D

def cal_line_equation(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    # Calculate the direction vector
    direction = np.asarray([x2 - x1, y2 - y1, z2 - z1])

    # Choose one of the points as the origin
    origin = point1

    return np.concatenate((origin, direction), axis=0)


def cal_dist_point_to_line(line_eq, Q):
    P0, v = line_eq[0:3], line_eq[3:6]
    # P0: Line origin
    # v: Line direction
    # Q: Point outside the line

    # Compute vector P0Q
    P0Q = Q - P0

    # Compute the projected vector t0 of vector P0Q on the line R0v
    # t0 = np.dot(P0Q, v) / np.dot(v, v)
    t0 = np.dot(P0Q, v) / (np.linalg.norm(v)**2)

    # Compute the closest point on the line to Q (location of the projected point)
    closest_point_on_line = P0 + t0 * v

    # Compute the distance between Q and the closest point (projected point) on the line
    distance = np.linalg.norm(Q - closest_point_on_line)

    return distance

def cal_dist_point_to_point(point_1, point_2):
    distance = np.linalg.norm(point_1 - point_2)
    return distance

def cal_dist_line_to_line(line1_eq, line2_eq):
    line1_point, line1_direction = line1_eq[0:3], line1_eq[3:6]
    line2_point, line2_direction = line2_eq[0:3], line2_eq[3:6]

    # Find a point on each line (line parameter t = 0)
    point1 = line1_point
    point2 = line2_point

    # Vector connecting the two points
    v = point2 - point1

    # Orthogonal vector to both lines (cross product of direction vectors)
    orthogonal_vector = np.cross(line1_direction, line2_direction)

    # Magnitude of the projection of v onto the orthogonal vector
    distance = np.abs(np.dot(v, orthogonal_vector) / np.linalg.norm(orthogonal_vector))

    return distance

def middle_point_between_two_lines(line1_eq, line2_eq):
    line1_point, line1_direction = line1_eq[0:3], line1_eq[3:6]
    line2_point, line2_direction = line2_eq[0:3], line2_eq[3:6]

    v1 = line1_direction
    v2 = line2_direction
    
    # Vectors connecting a point on each line to the other line's points
    p1_to_p2 = np.array(line2_point) - np.array(line1_point)
    
    # Calculate the parameters for the closest point
    v1_dot_v1 = np.dot(v1, v1)
    v2_dot_v2 = np.dot(v2, v2)
    v1_dot_v2 = np.dot(v1, v2)
    p1_to_p2_dot_v1 = np.dot(p1_to_p2, v1)
    p1_to_p2_dot_v2 = np.dot(p1_to_p2, v2)
    
    # Calculate parameters for the closest points on each line
    t = (v2_dot_v2 * p1_to_p2_dot_v1 - v1_dot_v2 * p1_to_p2_dot_v2) / (v1_dot_v1 * v2_dot_v2 - v1_dot_v2 ** 2)
    s = (v1_dot_v2 * p1_to_p2_dot_v1 - v1_dot_v1 * p1_to_p2_dot_v2) / (v1_dot_v1 * v2_dot_v2 - v1_dot_v2 ** 2)
    
    # Calculate the closest points on each line
    closest_point_line1 = np.array(line1_point) + t * np.array(line1_direction)
    closest_point_line2 = np.array(line2_point) + s * np.array(line2_direction)
    
    # Calculate the midpoint between the closest points
    midpoint = (closest_point_line1 + closest_point_line2) / 2.0
    
    return np.asarray(midpoint)

def cal_line_distance_n_point(line1_eq, line2_eq):
    distance = cal_dist_line_to_line(line1_eq, line2_eq)
    middle_point = middle_point_between_two_lines(line1_eq, line2_eq)
    return distance.astype(int), middle_point.astype(int)

def cal_point_on_line_given_z(line1_eq, z_given):
    """
    Calculate the point on a 3D line given the z-value.
    
    Arguments:
    r0 : tuple or list - Coordinates of a point on the line (x0, y0, z0).
    direction : tuple or list - Direction vector of the line (a, b, c).
    z_given : float - The z-value of the desired point on the line.
    
    Returns:
    tuple - Coordinates of the point on the line with the given z-value.
    """
    r0, direction = line1_eq[0:3], line1_eq[3:6]

    # Unpack coordinates
    x0, y0, z0 = r0
    a, b, c = direction
    
    # Calculate parameter t
    t = (z_given - z0) / c
    
    # Calculate point coordinates
    x = x0 + t * a
    y = y0 + t * b
    
    return np.asarray([x, y, z_given])

def plot_line_3d(ax, origin, direction, label, alpha=1):
    # origin, direction = line_eq[0:3]/100, line_eq[3:6]/100

    ax.plot3D([origin[0], origin[0] + direction[0]], 
            [origin[1], origin[1] + direction[1]], 
            [origin[2], origin[2] + direction[2]], label=label, alpha=alpha)
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Test the function
    # line1_origin = [0, 2, 0]
    # line1_direction = [4, -4, -4]

    # line2_origin = [0, 0, 0]
    # line2_direction = [4, 0, 0]

    # line3_origin = [1, 1, 0]
    # line3_direction = [2, 2, 1]

    ## On the same surface
    line1_origin = [0, 0, 0]
    line1_direction = [3, 0, 0]

    line2_origin = [3, 0, 0]
    line2_direction = [-1.5, 3, 0]

    line3_origin = [1.5, 3, 0]
    line3_direction = [-1.5, -3, 0]

    line1_eq = np.concatenate((line1_origin, line1_direction), axis=0)
    line2_eq = np.concatenate((line2_origin, line2_direction), axis=0)
    line3_eq = np.concatenate((line3_origin, line3_direction), axis=0)

    # distance = cal_dist_line_to_line(line1_eq, line2_eq)
    # print("Distance between the lines:", distance)

    nearest_point = middle_point_between_two_lines(line1_eq, line2_eq)
    # nearest_point = [ 1.19429258e-001,  4.97476007e-003, -1.57845382e-251]
    nearest_point = [ 1.5,  0, 0]
    print("Nearest point:", nearest_point)

    dist1 = cal_dist_point_to_line(line1_eq, nearest_point)
    dist2 = cal_dist_point_to_line(line2_eq, nearest_point)
    dist3 = cal_dist_point_to_line(line3_eq, nearest_point)

    print(f"Dist: {dist1 + dist2 + dist3}")
    

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the lines
    t = np.linspace(-10, 10, 100)
    line1_points = np.array(line1_origin) + np.outer(t, np.array(line1_direction))
    line2_points = np.array(line2_origin) + np.outer(t, np.array(line2_direction))

    plot_line_3d(ax, line1_origin, line1_direction, 'line 1')
    plot_line_3d(ax, line2_origin, line2_direction, 'line 2')
    plot_line_3d(ax, line3_origin, line3_direction, 'line 3')

    # Plot the axes
    ax.plot([-1, 1], [0, 0], [0, 0], color='r')  # X-axis
    ax.plot([0, 0], [-1, 1], [0, 0], color='g')  # Y-axis
    ax.plot([0, 0], [0, 0], [-1, 1], color='b')  # Z-axis

    # Plot the nearest point
    ax.scatter(nearest_point[0], nearest_point[1], nearest_point[2], color='red', label='Nearest Point')

    # Set the same scale for all axes
    ax.set_aspect('equal')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()
