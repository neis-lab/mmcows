import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


Anchors = np.asarray([
                    [0,0,0],                # 0
                    [-6.1, 5.13, 3.88],      # 1
                    [0, 5.13, 4.04],         # 2
                    [6.1, 5.13, 3.95],       # 3
                    [-0.36, 0, 5.17],       # 4
                    [0.36, 0, 5.17],        # 5
                    [-6.1, -6.26, 5.47],    # 6
                    [0, -6.26, 5.36],       # 7
                    [6.1, -6.26, 5.49]])    # 8


stall_y = 4.75
stall_x = 10.27
stall_z = 0.2
stall_y_off = -stall_y/2
stall_x_off = -4
stall_z_off = 0
stall_face_color = 'purple'
stall_face_opacity = 0
stall_edge_color = 'purple'

# stall_corner_1 = np.asarray([-stall_x/2,stall_y/2,stall_z])
# stall_corner_2 = np.asarray([stall_x/2,stall_y/2,stall_z])
# stall_corner_3 = np.asarray([stall_x/2,-stall_y/2,stall_z])
# stall_corner_4 = np.asarray([-stall_x/2,-stall_y/2,stall_z])

# feed_lock_x = 20
# feed_lock_x_off = -6
# feed_lock_z = 1.55

# feed_lock_1 = np.asarray([-feed_lock_x/2,feed_lock_x_off,feed_lock_z])
# feed_lock_2 = np.asarray([feed_lock_x/2,feed_lock_x_off,feed_lock_z])

pen_min_x, pen_max_x = -8.79, 10.42
pen_min_y, pen_max_y = -6.46, 5.33
pen_z = 0

pen_corner_1 = np.asarray([pen_min_x,pen_max_y,pen_z])
pen_corner_2 = np.asarray([pen_max_x,pen_max_y,pen_z])
pen_corner_3 = np.asarray([pen_max_x,pen_min_y,pen_z])
pen_corner_4 = np.asarray([pen_min_x,pen_min_y,pen_z])


feed_lock_x = pen_max_x - pen_min_x
feed_lock_y = 0
feed_lock_x_off = pen_min_x
feed_lock_y_off = pen_min_y
feed_lock_z_off = 0.14
feed_lock_z = 1.47 - feed_lock_z_off
feed_lock_face_color = 'purple'
feed_lock_face_opacity = 0
feed_lock_edge_color = 'purple'

feeding_area_x = feed_lock_x
feeding_area_y = 1.8
feeding_area_z = 0
feeding_area_x_off = pen_min_x
feeding_area_y_off = -feeding_area_y + feed_lock_y_off
feeding_area_z_off = 0.14
feeding_area_face_color = 'purple'
feeding_area_face_opacity = 0 #.05
feeding_area_edge_color = 'purple'


trough_x = 0.6
trough_y = 1.8
trough_z = 0.6
trough_right_x_off = (pen_max_x - trough_x) 
trough_left_x_off = (pen_min_x - trough_x/2) 
trough_y_off = -trough_y/2
trough_z_off = 0.15
trough_face_color = 'gold'
trough_edge_color = 'y'
trough_face_opacity = 0.3

# ele_area_y = 5
# ele_area_x = 5
# ele_area_z = 0.15
# ele_area_y_off = -ele_area_y/2
# ele_area_left_x_off = -(ele_area_x + stall_x/2)
# ele_area_right_x_off = (stall_x/2)
# ele_area_z_off = 0
# ele_area_face_color = 'gray'
# ele_area_face_opacity = 0
# ele_area_edge_color = 'slategrey'


# RGB values for colors from brown to red
brown = (139/255, 69/255, 19/255)
light_brown = (205/255, 133/255, 63/255)
medium_brown =  (255/255, 129/255, 89/255)
medium_red = (255/255, 80/255, 42/255)
red = (255/255, 0/255, 0/255)

def draw_cube(ax, x, y, z, x_off, y_off, z_off, face_color, face_opacity, edge_color):
    # Define cube vertices with offset
    vertices = [
        [0 + x_off, 0 + y_off, 0 + z_off],
        [x + x_off, 0 + y_off, 0 + z_off],
        [x + x_off, y + y_off, 0 + z_off],
        [0 + x_off, y + y_off, 0 + z_off],
        [0 + x_off, 0 + y_off, z + z_off],
        [x + x_off, 0 + y_off, z + z_off],
        [x + x_off, y + y_off, z + z_off],
        [0 + x_off, y + y_off, z + z_off]
    ]

    # Define cube faces
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[7], vertices[6], vertices[2], vertices[3]],
        [vertices[0], vertices[4], vertices[7], vertices[3]],
        [vertices[1], vertices[5], vertices[6], vertices[2]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[2], vertices[3]]
    ]

    # Draw transparent cube
    ax.add_collection3d(Poly3DCollection(faces, facecolors=face_color, linewidths=1, edgecolors=edge_color, alpha=face_opacity))

def draw_pen(ax, cam_coord, anchor=True, structure=True, legend=True, hide_ticks=False):
    
    if hide_ticks == True:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # Set the axis limits to -5 to 5 for all three axes
    min_x, max_x = -12, 12
    ax.set_xlim(min_x, max_x+1)
    ax.set_ylim(-8, 6)
    ax.set_zlim(0, 5)

    ax.set_xticks(np.arange(min_x, max_x+1, 2))
    ax.set_yticks(np.arange(-6, 6, 2))
    # ax.set_zticks(np.arange(0, z_range, 2))

    # Format the grid: Not working
    # ax.grid(True)
    # ax.xaxis.grid(color='gray', linestyle='dashed', linewidth=0.5)
    # ax.grid(linewidth=5)
    # ax.grid(b=True, which='major', color='gray', linestyle='-')

    # Plot the unit axes 
    # ax.plot([-x_range, x_range], [0, 0], [0, 0], color='green', alpha=1)  # X-axis
    # ax.plot([0, 0], [-y_range, y_range], [0, 0], color='green', alpha=1)  # Y-axis
    # ax.plot([0, 0], [0, 0], [0, z_range], color='green', alpha=1)  # Z-axis

    if structure==True:
        # Stall boundaries
        draw_cube(ax, stall_x, stall_y, stall_z, stall_x_off, stall_y_off, stall_z_off, stall_face_color, stall_face_opacity, stall_edge_color) 

        # Elevated areas
        # draw_cube(ax, ele_area_x, ele_area_y, ele_area_z, ele_area_left_x_off, ele_area_y_off, ele_area_z_off, ele_area_face_color, ele_area_face_opacity, ele_area_edge_color) # left
        # draw_cube(ax, ele_area_x, ele_area_y, ele_area_z, ele_area_right_x_off, ele_area_y_off, ele_area_z_off, ele_area_face_color, ele_area_face_opacity, ele_area_edge_color) # right
        draw_cube(ax, feeding_area_x, feeding_area_y, feeding_area_z, feeding_area_x_off, feeding_area_y_off, feeding_area_z_off, feeding_area_face_color, feeding_area_face_opacity, feeding_area_edge_color) 

        # Feed lock
        draw_cube(ax, feed_lock_x, feed_lock_y, feed_lock_z, feed_lock_x_off, feed_lock_y_off, feed_lock_z_off, feed_lock_face_color, feed_lock_face_opacity, feed_lock_edge_color) 

        # Pen boundaries
        ax.plot([pen_corner_1[0], pen_corner_2[0]], [pen_corner_1[1], pen_corner_2[1]], [pen_corner_1[2], pen_corner_2[2]], color='purple', alpha=1)  # corner 1-2
        ax.plot([pen_corner_2[0], pen_corner_3[0]], [pen_corner_2[1], pen_corner_3[1]], [pen_corner_2[2], pen_corner_3[2]], color='purple', alpha=1)  # corner 2-3
        ax.plot([pen_corner_3[0], pen_corner_4[0]], [pen_corner_3[1], pen_corner_4[1]], [pen_corner_3[2], pen_corner_4[2]], color='purple', alpha=1)  # corner 3-4
        ax.plot([pen_corner_1[0], pen_corner_4[0]], [pen_corner_1[1], pen_corner_4[1]], [pen_corner_1[2], pen_corner_4[2]], color='purple', alpha=1)  # corner 1-4

        # Trough 1 boundaries
        draw_cube(ax, trough_x, trough_y, trough_z, trough_left_x_off, trough_y_off, trough_z_off, trough_face_color, trough_face_opacity, trough_edge_color) # left
        draw_cube(ax, trough_x, trough_y, trough_z, trough_right_x_off, trough_y_off, trough_z_off, trough_face_color, trough_face_opacity, trough_edge_color) # right

    # Show all anchors
    if anchor == True:
        ax.scatter(Anchors[1:, 0], Anchors[1:, 1], Anchors[1:, 2], c='navy', marker='P', s=60, alpha=1, label='UWB anchor') # All anchors

    # Show cameras
    cam_coord = cam_coord / 100
    ax.scatter(cam_coord[:, 0], cam_coord[:, 1], cam_coord[:, 2], c='darkred', marker='s', s=60, alpha=1, label='Camera')
    ax.text(cam_coord[0, 0], cam_coord[0, 1], cam_coord[0, 2] + 0.3, f'{1}', fontsize=16, color='black', ha='center', va='bottom')
    ax.text(cam_coord[1, 0], cam_coord[1, 1], cam_coord[1, 2] + 0.3, f'{2}', fontsize=16, color='black', ha='center', va='bottom')
    ax.text(cam_coord[2, 0], cam_coord[2, 1], cam_coord[2, 2] + 0.3, f'{3}', fontsize=16, color='black', ha='center', va='bottom')
    ax.text(cam_coord[3, 0], cam_coord[3, 1], cam_coord[3, 2] + 0.3, f'{4}', fontsize=16, color='black', ha='center', va='bottom')

    if legend==True:
        # Annotating the legend
        ax.scatter([], [], marker='o', c='blue', s=50, label='Non-lying cow')
        ax.scatter([], [], marker='o', c='green', s=50, label='Lying cow')
        ax.scatter([], [], marker='s', c='gold', s=50, label='Water trough')
        ax.plot([], [], color=rgb2hex(red), label='CBT > 39.00')[0]
        ax.plot([], [], color=rgb2hex(medium_red), label='38.75—39.00')[0]
        ax.plot([], [], color=rgb2hex(medium_brown), label='38.50—38.75')[0]
        ax.plot([], [], color=rgb2hex(light_brown), label='38.25—38.50')[0]
        ax.plot([], [], color=rgb2hex(brown), label='CBT < 38.25')[0]

        ax.legend(loc="lower left", bbox_to_anchor=(0,0), ncol=3)
    

    # Hide X and Y axes tick marks
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])

    # Set equal aspect mode for all three axes
    ax.set_box_aspect([np.ptp([min_x,max_x]), np.ptp([-8, 6]), np.ptp([0,5])])
    # ax.set_box_aspect([np.ptp([-13,13]), np.ptp([-8, 6]), np.ptp([0,5])])

if __name__ == '__main__':
    cam_coord = np.asarray([[-1189,   541,   383],     # cam 1
                      [1191,  584,  356],     # cam 2
                      [1179, -647,  379],     # cam 3
                      [-1186,  -656,   383]])    # cam 4
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    draw_pen(ax1, cam_coord)
    plt.show()