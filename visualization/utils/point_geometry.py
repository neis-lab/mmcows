import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def find_outliers(points, threshold=200):
    """
    Find the indices of the points that are outliers based on the distance from the cluster center.
    
    Arguments:
    points : list of tuples - List of 3D points [(x1, y1, z1), (x2, y2, z2), ...].
    center : tuple - Coordinates of the cluster center (x, y, z).
    threshold : float - Threshold distance for considering outliers.
    
    Returns:
    list - Indices of the outlier points.
    """

    # find cluster center
    cluster_center = np.median(points, axis=0)

    outliers_indices = []
    for i, point in enumerate(points):
        distance = np.linalg.norm(point - cluster_center)
        # print(distance)
        if distance > threshold:
            outliers_indices.append(i)
    return outliers_indices, cluster_center

# from sklearn.cluster import KMeans
# def find_outliers(points, threshold=200):
#     # KMeans clustering
#     kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(points)

#     # Compute the threshold
#     centroids = kmeans.cluster_centers_
#     threshold = np.mean(centroids)

#     return outliers_indices, cluster_center

if __name__ == '__main__':
    points = [np.array([4, 5, 6]), np.array([7, 8, 9]), np.array([10, 11, 0]), np.array([400, 100, 0])]
    outliers_indices, cluster_center = find_outliers(points, threshold=200)
    print("Cluster center:", cluster_center)
    print("Indices of outliers:", outliers_indices)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points = np.array(points)

    # Plot points
    ax.scatter(points[:,0], points[:,1], points[:,2], c='b', marker='o', label='Points')

    # Plot cluster center
    ax.scatter(cluster_center[0], cluster_center[1], cluster_center[2], c='r', marker='s', s=20, label='Cluster Center')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot')

    # Set the same scale for all axes
    ax.set_aspect('equal')

    # Add legend
    ax.legend()
    plt.show()

