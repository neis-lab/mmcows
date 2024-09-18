import numpy as np
from sympy.solvers import solve
from sympy import *

# New number 2024.01.23
n_y = 5.33 - 0.2
s_y = 6.46 - 0.2

# # Previously used numbers
# n_y = 5.7
# s_y = 5.67

anchors_3D_position = np.asarray([
                    [0,0,0],            #
                    [-6.1,n_y , 3.88],  # 1
                    [0, n_y, 4.04],     # 2
                    [6.1, n_y, 3.95],   # 3
                    [-0.36, 0, 5.17],   # 4
                    [0.36, 0, 5.17],    # 5
                    [-6.1, -s_y, 5.47],# 6
                    [0, -s_y, 5.36],   # 7
                    [6.1, -s_y, 5.49]])# 8

class triagulation():
    def __init__(self):
        self.dummy = 0

    def check_numbers_exist(self, arr1, arr2):
        return np.all(np.isin(arr1, arr2))

    def get_smallest_samples(self, array, index, n_data):
        feature_to_sort = array[:,index]
        shortest_indices = np.argsort(feature_to_sort)[:n_data]
        smallest_data_points = array[shortest_indices]
        return smallest_data_points

    def get_largest_samples(self, array, index, n_data):
        feature_to_sort = array[:,index]
        lastest_indices = np.argsort(feature_to_sort)[-n_data:]
        largest_data_points = array[lastest_indices]
        return largest_data_points

    # def mannual_data_points(self, data_points):
    #     anchor_1 = data_points[0,:]
    #     anchor_2 = data_points[1,:]
    #     anchor_3 = data_points[2,:]
    #     anchor_4 = data_points[3,:]
    #     anchor_5 = data_points[4,:]
    #     anchor_6 = data_points[5,:]
    #     anchor_7 = data_points[6,:]
    #     anchor_8 = data_points[7,:]

    #     selected_samples = np.vstack((anchor_3, anchor_5, anchor_8))
    #     # selected_samples = np.vstack((anchor_1, anchor_4, anchor_6))
    #     # selected_samples = np.vstack((anchor_4, anchor_6, anchor_7))

    #     return selected_samples

    def cal_group_score(self, data_dict):
        data_array = data_dict["group_data"]
        # invalid_anchor_id = 0

        anchor_ids = data_array[:,1]
        proba_LOS = data_array[:,4]
        if np.any(anchor_ids == -1) or np.any(proba_LOS < 30): # invalid ID 
            data_dict["score"] = -1.0
        else:
            distance = -1/10 * np.mean(data_array[:,2]) + 13/10
            n_samples = np.abs(np.mean(data_array[:,3])/5)  # Normalization
            LOS_proba = np.abs(np.mean(data_array[:,4])/100) # Normalization

            # data_dict["score"] = distance * 8 + n_samples * 3 + LOS_proba * 5 # Combining different factos
            data_dict["score"] = (-1)*np.sum(data_array[:,2]) + 50 # Based only on the shortest distance

        min_LOS = 0 # Filtering out based on LOS
        for data_point in data_array:
            if data_point[4] < min_LOS: 
                data_dict["score"] = -1.0
        
    def find_best_group(self, data_points):

        anchor_1 = data_points[0,:] # Takes in a row corresponding to data from one anchor
        anchor_2 = data_points[1,:]
        anchor_3 = data_points[2,:]
        anchor_4 = data_points[3,:]
        anchor_5 = data_points[4,:]
        anchor_6 = data_points[5,:]
        anchor_7 = data_points[6,:]
        anchor_8 = data_points[7,:]

        # Small triangles
        anchor_group_1 = np.vstack((anchor_1, anchor_2, anchor_4)) # Group containing data of three anchors
        anchor_group_2 = np.vstack((anchor_2, anchor_3, anchor_5))
        anchor_group_3 = np.vstack((anchor_1, anchor_4, anchor_6))
        anchor_group_4 = np.vstack((anchor_3, anchor_5, anchor_8))
        anchor_group_5 = np.vstack((anchor_4, anchor_6, anchor_7))
        anchor_group_6 = np.vstack((anchor_5, anchor_7, anchor_8))

        # Medium triangles 1
        anchor_group_7 = np.vstack((anchor_1, anchor_6, anchor_7))
        anchor_group_8 = np.vstack((anchor_2, anchor_6, anchor_7))
        anchor_group_9 = np.vstack((anchor_1, anchor_2, anchor_6))
        anchor_group_10 = np.vstack((anchor_1, anchor_2, anchor_7))

        # Medium triangles 2
        anchor_group_11 = np.vstack((anchor_2, anchor_7, anchor_8))
        anchor_group_12 = np.vstack((anchor_3, anchor_7, anchor_8))
        anchor_group_13 = np.vstack((anchor_2, anchor_3, anchor_7))
        anchor_group_14 = np.vstack((anchor_2, anchor_3, anchor_8))

        # Big triangles
        anchor_group_15 = np.vstack((anchor_1, anchor_7, anchor_3))
        anchor_group_16 = np.vstack((anchor_6, anchor_2, anchor_8))

        anchor_group_17 = np.vstack((anchor_1, anchor_6, anchor_3))
        anchor_group_18 = np.vstack((anchor_1, anchor_6, anchor_8))
        anchor_group_19 = np.vstack((anchor_3, anchor_8, anchor_1))
        anchor_group_20 = np.vstack((anchor_3, anchor_8, anchor_6))

        anchor_group_list = [ anchor_group_1,
                            anchor_group_2,
                            anchor_group_3,
                            anchor_group_4,
                            anchor_group_5,
                            anchor_group_6,
                            anchor_group_7,
                            anchor_group_8,
                            anchor_group_9,
                            anchor_group_10,
                            anchor_group_11,
                            anchor_group_12,
                            anchor_group_13,
                            anchor_group_14,
                            anchor_group_15,
                            anchor_group_16,
                            anchor_group_17,
                            anchor_group_18,
                            anchor_group_19,
                            anchor_group_20]
        
        anchor_group_dict_list = []
        for anchor_group in anchor_group_list:
            anchor_group_dict = {"group_data": anchor_group, "score": 0}
            anchor_group_dict_list.append(anchor_group_dict)

        combined_list = []
        for anchor_group_dict in anchor_group_dict_list:
            self.cal_group_score(anchor_group_dict)
            combined_list.append(anchor_group_dict)

        sorted_list = sorted(combined_list, key=lambda x: x["score"], reverse=True) # sort in decreasing order by score
        first_dict = sorted_list[0]
        data_point = first_dict['group_data']
        # print(data_point[:,1].astype('int'))

        if first_dict['score'] == -1:
            return (-1) * np.ones((3, 6))
        else:
            for element in sorted_list:
                print(int(element['score']), end=' ')
            print('\t', end=' ')
            return data_point

    def location_cal(self, anchors_3D_position, an_id, d): # an_id starts from 1
        x, y, z = symbols('x, y, z')
        A1 = anchors_3D_position[an_id[0]][0]
        A2 = anchors_3D_position[an_id[0]][1]
        A3 = anchors_3D_position[an_id[0]][2]
        B1 = anchors_3D_position[an_id[1]][0]
        B2 = anchors_3D_position[an_id[1]][1]
        B3 = anchors_3D_position[an_id[1]][2]
        C1 = anchors_3D_position[an_id[2]][0]
        C2 = anchors_3D_position[an_id[2]][1]
        C3 = anchors_3D_position[an_id[2]][2]

        Eq1 = Eq(x**2 + y**2 + z**2 - 2*(A1*x + A2*y + A3*z), d[0]**2 - (A1**2 + A2**2 + A3**2))
        Eq2 = Eq(x**2 + y**2 + z**2 - 2*(B1*x + B2*y + B3*z), d[1]**2 - (B1**2 + B2**2 + B3**2))
        Eq3 = Eq(x**2 + y**2 + z**2 - 2*(C1*x + C2*y + C3*z), d[2]**2 - (C1**2 + C2**2 + C3**2))
        # try:
        x1, x2 = solve([Eq1, Eq2, Eq3], [x, y, z])
        # except:
        #     x1 = solve([Eq1, Eq2, Eq3], [x, y, z])
        #     x2 = np.array([0, 0, 10])

        x1 = [complex(item) for item in x1]
        x2 = [complex(item) for item in x2]
        # print(x1, end=" ")
        if np.iscomplex(x1[0]) or np.iscomplex(x1[1]) or np.iscomplex(x1[2]):

            # print(' ')
            return np.round(x1,3)
        else:

            if float(np.real(x2[2])) < float(np.real(x1[2])):
                x1[2] = x2[2]
            # loc = np.round(np.array([float(np.real(x1[0])), float(np.real(x1[1])), float(np.real(x1[2])), float(np.real(x2[2]))]),3)
            loc = np.round(np.array([float(np.real(x1[0])), float(np.real(x1[1])), float(np.real(x1[2]))]),3)
            # print(str('[%.2f\t%.2f\t%.2f]' %(loc[0], loc[1], loc[2])))
            return loc
        # return 1

# ================================================
class gd_localization():
    def __init__(self):
        self.dummy = 0

    def random_location(self):
        # Specify the ranges for each element
        range_first_element = (-8, 8)
        range_second_element = (-5, 5)
        range_third_element = (0, 2)

        # Create a random 1 by 3 array
        random_array = np.array([
            np.random.uniform(*range_first_element),
            np.random.uniform(*range_second_element),
            np.random.uniform(*range_third_element)
        ])
        return random_array

    # Function to calculate the Euclidean distance between two points in 3D space
    def distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2)**2))

    # Objective function to minimize
    def objective_function(self, target_location, distances, weights, anchors_location):
        total_error = 0
        for i, point in enumerate(anchors_location):
            dist = self.distance(target_location, point)
            total_error += weights[i] * (distances[i] - dist)**2
        return total_error

    # Gradient of the objective function with respect to coordinates (x, y, z) of point P
    def gradient(self, target_location, distances, weights, anchors_location):
        grad = np.zeros(3)
        for i, point in enumerate(anchors_location):
            diff = target_location - point
            dist = self.distance(target_location, point)
            grad += 2 * weights[i] * (dist - distances[i]) * diff / dist
        return grad

    # Gradient Descent algorithm
    def gradient_descent(self, prev_location, anchors_location, distances, weights, learning_rate=0.01, num_iterations=10000):
        # Initialize the coordinates of the unknown point
        target_location = prev_location
        iteration = 0
        # loss_value = 1000

        gradient_threshold=0.004 * (np.linalg.norm(weights)/np.linalg.norm(np.ones(len(weights))))  # norm of weights ranging from 0 to > 1, thus the norm needs to be normalized w.r.t max length

        while iteration < num_iterations:
            # Calculate the gradient
            grad = self.gradient(target_location, distances, weights, anchors_location)
            
            # Update the coordinates of P using the gradient descent update rule
            target_location -= learning_rate * grad

            if target_location[2] > 5:
                target_location[2] -= 5
                target_location[2] *= (-1)
                target_location[2] += 5

            # Check the magnitude of the gradient
            if np.linalg.norm(grad) < gradient_threshold:
                break

            iteration += 1
        
        loss_value = self.objective_function(target_location=target_location, distances=distances, weights=weights, anchors_location=anchors_location)

        return target_location, loss_value, iteration


    def gd_location (self, prev_location, anchors_location, distances, weights): # anchor_id starts from 1

        n_tries = 0
        loss_value = 10
        loc = np.zeros(3)
        n_iterations = 0
        loss_threshold = 3 # min acceptable loss value
        max_n_tries = 5

        loss_list = []
        loc_list = []

        while n_tries < max_n_tries and loss_value > loss_threshold:
            if n_tries == 0:
                prev_location = prev_location
            if n_tries == 1:
                print("n_tries: " + str(n_tries+1), end=" ")
                prev_location = np.zeros(3)
            if n_tries > 1:
                print(" " + str(n_tries+1), end=",")
                prev_location = self.random_location()

            loc, loss_value, n_iterations = self.gradient_descent(prev_location, anchors_location, distances, weights)

            loss_list.append(loss_value)
            loc_list.append(loc)

            n_tries += 1

        # Find the index of the smallest value
        if n_tries > 1:
            loss_value = min(loss_list)
            min_index = loss_list.index(loss_value)
            loc = loc_list[min_index] 
            print(" ")

        return np.round(loc,3), loss_value, n_iterations
    


    def gd_localization(self, prev_location, all_anchor_data):
        anchor_1 = all_anchor_data[0,:]
        anchor_2 = all_anchor_data[1,:]
        anchor_3 = all_anchor_data[2,:]
        anchor_4 = all_anchor_data[3,:]
        anchor_5 = all_anchor_data[4,:]
        anchor_6 = all_anchor_data[5,:]
        anchor_7 = all_anchor_data[6,:]
        anchor_8 = all_anchor_data[7,:]
        anchor_group_1 = np.array([anchor_1, anchor_2, anchor_4, anchor_6, anchor_7])
        anchor_group_2 = np.array([anchor_2, anchor_3, anchor_5, anchor_7, anchor_8])

        # print(np.shape(anchor_group_1))
        # print(np.shape(anchor_group_2))
        
        data_list_1 = []
        for row in anchor_group_1:
            anchor_id = row[1]
            n_samples = row[3]
            if anchor_id != -1 and n_samples != -1:
                data_list_1.append(row)
        anchor_group_1 = np.asarray(data_list_1).reshape((-1,6))

        data_list_2 = []
        for row in anchor_group_2:
            anchor_id = row[1]
            n_samples = row[3]
            if anchor_id != -1 and n_samples != -1:
                data_list_2.append(row)
        anchor_group_2 = np.asarray(data_list_2).reshape((-1,6))

        data_list_3 = []
        for row in all_anchor_data:
            anchor_id = row[1]
            n_samples = row[3]
            if anchor_id != -1 and n_samples != -1:
                data_list_3.append(row)
        anchor_group_all = np.asarray(data_list_3).reshape((-1,6))

        ## Choosing between two groups
        group_1_distances = anchor_group_1[:, 2].astype(float)
        group_1_score = - np.mean(group_1_distances) # Based on shortest distance

        group_2_distances = anchor_group_2[:, 2].astype(float)
        group_2_score = - np.mean(group_2_distances) # Based on shortest distance

        if group_1_score > group_2_score:
            selected_anchor_group = anchor_group_1
        else:
            selected_anchor_group = anchor_group_2

        valid_datapoint = True
        if selected_anchor_group.shape[0] < 3:
            valid_datapoint = False
        if selected_anchor_group.shape[0] == 3:
            anchor_ids = selected_anchor_group[:, 1].astype(int)
            if np.any(anchor_ids == 2) and np.any(anchor_ids == 7):
                if np.any(anchor_ids == 4) or np.any(anchor_ids == 5):
                    valid_datapoint = False

        if valid_datapoint == False:
            selected_anchor_group = anchor_group_all

        data_point = selected_anchor_group

        if len(data_point[:,0]) > 2:
            # Extract data from the selected anchors
            anchor_ids  = data_point[:, 1].astype(int)
            distances   = data_point[:, 2].astype(float)
            n_samples   = data_point[:, 3].astype(float)
            LOS_proba   = data_point[:, 4].astype(float)

            anchors_location = []
            for i in anchor_ids:
                anchors_location.append(anchors_3D_position[i])
            anchors_location = np.asarray(anchors_location).reshape((-1, 3))

            # weights = LOS_proba/100
            # weights = (LOS_proba/1.5 + 33)/100
            # weights = np.ones(len(LOS_proba)).astype(float)
            weights = np.multiply((LOS_proba/1.5 + 33)/100, n_samples/5)
            curr_location, loss_value, n_iterations = self.gd_location(prev_location, anchors_location, distances, weights)

            # ## Triangulation when gd failed
            # x_loc = curr_location[0]
            # y_loc = curr_location[1]
            # z_loc = curr_location[2]
            # if np.abs(x_loc) > 11 or np.abs(y_loc) > 7 or z_loc > 2 or z_loc < 0:

            #     tri = triagulation()
                
            #     data_point = tri.find_best_group(all_anchor_data)
            #     # Extract data from the selected anchors
            #     anchor_ids  = data_point[:, 1].astype(int)
            #     distances   = data_point[:, 2].astype(float)
            #     n_samples   = data_point[:, 3].astype(int)
            #     LOS_proba   = data_point[:, 4].astype(int)

            #     anchors_location = []
            #     for i in anchor_ids:
            #         anchors_location.append(anchors_3D_position[i])
            #     anchors_location = np.asarray(anchors_location).reshape((-1, 3))

            #     weights = np.ones(len(LOS_proba)).astype(float)
            #     curr_location, loss_value, n_iterations = self.gd_location(prev_location, anchors_location, distances, weights)
        
        else:
            curr_location = np.full(3, np.nan).astype(float)
            loss_value = np.nan
            n_iterations = np.nan
            anchor_ids = np.nan
        
        return curr_location, loss_value, n_iterations, anchor_ids


# Example usage:
# ===============================================
if __name__ == '__main__':

    localizer = gd_localization()

    data_point = np.array(
        [[1690174841,	1,	11.6,	5,	0,	-99.7],
        [1690174841,	2,	7.01,	5,	76,	-93.7],
        [1690174841,	3,	6.28,	5,	6,	-93.7],
        [1690174841,	4,	5.97,	5,	97,	-97.4],
        [1690174841,	5,	5.48,	5,	0,	-93.4],
        [1690174841,	6,	12.6,	5,	0,	-95.9],
        [1690174841,	7,	8.95,	5,	80,	-92.7],
        [1690174841,	8,	8.26,	5,	0,	-94.1]]
    )

    for i in range(8):
        print(f'{int(data_point[i,0])}\t{int(data_point[i,1])}\t{data_point[i,2]:.2f}\t{int(data_point[i,3])}\t{int(data_point[i,4])}\t{int(data_point[i,5])}')

    ## Localization
    prev_location = np.array([0, 0, 0]).astype(float) # first initial location
    curr_location, loss_value, n_iterations, selected_anchor_ids = localizer.gd_localization(prev_location, data_point)

    print(f"curr_location {curr_location}")
    print(f"loss_value {loss_value}")
    print(f"n_iterations {n_iterations}")
    print(f"selected_anchor_ids {selected_anchor_ids}")

    print("\nDone\n")
            

