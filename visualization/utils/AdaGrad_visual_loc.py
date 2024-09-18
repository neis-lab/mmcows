import numpy as np

try:
    from utils.line_geometry import *
    from utils.point_geometry import *
except:
    from line_geometry import *
    from point_geometry import *

def visual_localization(line_eqs, 
                            initial_point=np.array([-100, -100, -100]), 
                            learning_rate=50, 
                            gradient_threshold = 0.0001, 
                            iterations=5000
                            ):
    current_point = initial_point.copy().astype(float)

    iter = 0
    sq_grad_sums = np.zeros((3))
    
    for i in range(iterations):
        # Compute gradients for each line
        gradients = np.zeros_like(current_point).astype(float)
        total_distance = 0
        for line in line_eqs:
            # print('line')
            P0, v = line[0:3], line[3:6]
            P0Q = current_point - P0
            t0 = np.dot(P0Q, v) / np.dot(v, v)
            closest_point_on_line = P0 + t0 * v
            direction = (current_point - closest_point_on_line) / np.linalg.norm(current_point - closest_point_on_line)
            
            gradients += direction
            total_distance += (cal_dist_point_to_line(line, current_point) / 1)

        # update the sum of the squared partial derivatives
        for i in range(gradients.shape[0]):
            sq_grad_sums[i] += gradients[i]**2.0
        
        for i in range(3):
            # calculate the step size for this variable
            alpha = learning_rate / (1e-9 + np.sqrt(sq_grad_sums[i]))
            
            # calculate the new position in this variable
            current_point[i] = current_point[i] - alpha * gradients[i]

        gradient = np.linalg.norm(gradients)
        if gradient < gradient_threshold:
            break

        iter += 1
    
    total_distance = 0
    for line in line_eqs:
        total_distance += cal_dist_point_to_line(line, current_point)

    return current_point, total_distance, iter, gradient

# Example usage:
if __name__ == "__main__":

    line1_origin = np.array([-1000, 600, 400])
    line1_point2 = np.array([100, -500, 0])
    
    line2_origin = np.array([1000, 600, 400])
    line2_point2 = np.array([-300, -600, 0])
    
    line3_origin = np.array([1000, -600, 400])
    # line3_point2 = np.array([100, 650, 0])
    line3_point2 = np.array([-200, -450, 0])
    
    line4_origin = np.array([-1000, -600, 400])
    line4_point2 = np.array([200, -350, 0])

    line1_eq = cal_line_equation(line1_origin, line1_point2)
    line2_eq = cal_line_equation(line2_origin, line2_point2)
    line3_eq = cal_line_equation(line3_origin, line3_point2)
    line4_eq = cal_line_equation(line4_origin, line4_point2)

    line1_direction = line1_eq[3:6]
    line2_direction = line2_eq[3:6]
    line3_direction = line3_eq[3:6]
    line4_direction = line4_eq[3:6]

    line_eqs = [
        line1_eq, line2_eq, line3_eq, line4_eq
    ]

    ## Remove lines that do not converge
    proj_point_list = []
    for curr_line in line_eqs:
        # Calculate the projected point on the common plane
        proj_point = cal_point_on_line_given_z(curr_line, z_given=130)
        # print(proj_point.astype(int))
        proj_point_list.append(proj_point)
    
    outliers_indices, cluster_center = find_outliers(proj_point_list, threshold=300)
    if len(outliers_indices) > 0:
        print("Indices of outliers:", outliers_indices)
        line_eqs = np.delete(line_eqs, outliers_indices, axis=0) # delete rows corresponding to the indices from the 2D array

    print("Cluster center:", cluster_center)

    ## Initial point for gradient descent
    initial_point = np.array([-100, -100, -100])  # Choose a point not directly on any line

    # Perform gradient descent to find the nearest 3D point
    nearest_point, total_distance, iter, gradient = visual_localization(line_eqs, 
                                                                        #    initial_point=np.array([-100, -100, -100]), 
                                                                        #    learning_rate=0.05, 
                                                                        #    gradient_threshold = 0.1, 
                                                                        #    iterations=5000
                                                                           )

    # gradient 0.01 -> [  96 -599   91] d = 1.7
    print(f'total d: {total_distance/100:.2f} m')
    print(f'gradient: {gradient:.5f}')
    print(f'# iter: {iter}')

    print("Nearest 3D point:", nearest_point.astype(int))

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
    plot_line_3d(ax, line4_origin, line4_direction, 'line 4')

    # Plot the axes
    ax.plot([-1, 1], [0, 0], [0, 0], color='r')  # X-axis
    ax.plot([0, 0], [-1, 1], [0, 0], color='g')  # Y-axis
    ax.plot([0, 0], [0, 0], [-1, 1], color='b')  # Z-axis

    # Plot the nearest point
    ax.scatter(nearest_point[0], nearest_point[1], nearest_point[2], color='red', label='Nearest Point')

    # Plot cluster center
    ax.scatter(cluster_center[0], cluster_center[1], cluster_center[2], c='b', marker='s', s=20, label='Cluster Center')

    # Set the same scale for all axes
    ax.set_aspect('equal')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()
