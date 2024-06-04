import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import label

DOORS_THRESHOLD = 0.49
P_INIT = 0.5
# MAPS_FOLDER = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/lab15_simulation"
# MAPS_FOLDER = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/lab15_sim_rotated"
MAPS_FOLDER = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/lab15_real"

# PATH_TO_SAVE = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/simulation_doors.png"
# PATH_TO_SAVE = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/simulation_doors_rotated.png"
PATH_TO_SAVE = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/real_doors.png"
TOLERANCE = 1e-6

def load_maps_from_folder():
    maps = []
    for file in os.listdir(MAPS_FOLDER):
        if file.endswith(".npy"):
            data = np.load(os.path.join(MAPS_FOLDER, file))
            maps.append(data)
    return  maps


def init_result_map(maps):
    max_rows = 0
    max_columns = 0

    for map in maps:
        rows, columns = map.shape
        max_rows = max(max_rows, rows)
        max_columns = max(max_columns, columns)
    
    # map = np.full((max_rows, max_columns), P_INIT, dtype=np.float32)
    map = np.zeros([max_rows,max_columns])
    return map 


def display_maps(maps):
    i = 1
    plt.figure(figsize=(15, 5))
    for map in maps:
        plt.subplot(2, 3, i)
        plt.scatter(50, 50, color="red", label='Lidar', s=10)
        plt.imshow(map, cmap='gray_r')
        plt.title(f'map {i}')
        plt.colorbar()
        i+=1
    
    plt.show()

def sanitize_map(map):
    map[np.isinf(map)] = 0
    map = np.nan_to_num(map)
    return map

def segmentation(binary_mask):
    pattern = [[1,1,1],
               [1,1,1],
               [1,1,1]]
    labeled_array, num_features = label(binary_mask, pattern)
    segments = []
    
    for i in range(1, num_features + 1):
        segment = np.argwhere(labeled_array == i)
        if len(segment) >= 3:  # Filter out small segments
            segments.append(segment)
    
    return segments, labeled_array

def remove_duplicates(list_of_arrays):
    seen = set()
    result = []
    for array in list_of_arrays:
        tup = tuple(map(tuple, array))
        if tup not in seen:
            seen.add(tup)
            result.append(array)
    return result

def main():
    maps = load_maps_from_folder()
    result_map = init_result_map(maps)
    background =np.copy(result_map)
    display_maps(maps)

    # plt.imshow(maps[0] - maps[1], cmap='gray_r')
    # plt.title('??? map')
    # plt.colorbar()
    # plt.show()
    # value = 1

    segments_list = []
    for map in maps:
        reflections = np.where(map > 0.9, 1, 0)
        for point in np.argwhere(reflections == 1):
            background[point[0], point[1]] = 1 

        for map2 in maps:
            if (map == map2).all():
                continue
            else:
                difference_map = np.absolute(map - map2)
                difference_map = sanitize_map(difference_map)
                binary_map = np.where(difference_map - DOORS_THRESHOLD > TOLERANCE, 1, 0)
                binary_map = sanitize_map(binary_map)

                # plt.figure(figsize=(15, 5))
                # plt.subplot(1, 4, 1)
                # plt.scatter(50, 50, color="red", label='Lidar', s=10)
                # plt.imshow(map, cmap='gray_r')
                # plt.title('Map 1')
                # plt.colorbar()

                # plt.subplot(1, 4, 2)
                # plt.scatter(50, 50, color="red", label='Lidar', s=10)
                # plt.imshow(map2, cmap='gray_r')
                # plt.title('Map 2')
                # plt.colorbar()

                # plt.subplot(1, 4, 3)
                # plt.scatter(50, 50, color="red", label='Lidar', s=10)
                # plt.imshow(difference_map, cmap='gray_r')
                # plt.title('Difference map')
                # plt.colorbar()

                # plt.subplot(1, 4, 4)
                # plt.scatter(50, 50, color="red", label='Lidar', s=10)
                # plt.imshow(binary_map, cmap='gray_r')
                # plt.title('Binary map')
                # plt.colorbar()
                # plt.show()
        
                segments, segment_labels = segmentation(binary_map)

                cmap = plt.get_cmap('summer', len(segments))
                cmap.set_under('white')
                colors = [cmap(i) for i in range(len(segments))]

                # plt.figure(figsize=(15, 5))
                # plt.subplot(1, 4, 1)
                # plt.imshow(difference_map, cmap='gray_r')
                # plt.scatter(50, 50, color="red", label='Lidar', s=10)
                # for i, segment in enumerate(segments):
                #     x_coords, y_coords = zip(*segment)
                #     plt.scatter(y_coords, x_coords, color=colors[i], label=f'Segment {i+1}', s=10)
                # plt.title('Map with segments')


                # plt.subplot(1, 4, 2)
                # plt.scatter(50, 50, color="red", label='Lidar', s=10)
                # plt.imshow(segment_labels, cmap='gray_r')
                # plt.title('segment_labels')

                for segment in segments:
                    for point in segment:
                        if result_map[point[0], point[1]] == 0:
                            result_map[point[0], point[1]] = 1

                    flag = True
                    for noted_segment in segments_list:
                        if np.array_equal(segment, noted_segment, equal_nan=False):
                            flag = False
                    
                    if flag is True:
                        segments_list.append(segment)

                # plt.subplot(1, 4, 3)
                # plt.scatter(50, 50, color="red", label='Lidar', s=10)
                # plt.imshow(result_map, cmap='gray_r')
                # plt.title('result_map')
                # # plt.show()

                # plt.subplot(1, 4, 4)
                # plt.scatter(50, 50, color="red", label='Lidar', s=10)

                segments, segment_labels = segmentation(result_map)

                cmap = plt.get_cmap('summer', len(segments))
                cmap.set_under('white')
                colors = [cmap(i) for i in range(len(segments))]

                # plt.imshow(difference_map, cmap='gray_r')
                # plt.scatter(50, 50, color="red", label='Lidar', s=10)
                # for i, segment in enumerate(segments):
                #     x_coords, y_coords = zip(*segment)
                #     plt.scatter(y_coords, x_coords, color=colors[i], label=f'Segment {i+1}', s=10)
                # plt.title('Map segments_list')
                # plt.show()

    print(len(segments_list))
    segments_list = remove_duplicates(segments_list)
    print(len(segments_list))

    plt.scatter(50, 50, color="red", label='Lidar', s=10)

    segments, segment_labels = segmentation(result_map)

    cmap = plt.get_cmap('summer', len(segments))
    cmap.set_under('white')
    colors = [cmap(i) for i in range(len(segments))]

    plt.imshow(background, cmap='gray_r')
    plt.scatter(50, 50, color="red", label='Lidar', s=10)
    for i, segment in enumerate(segments):
        x_coords, y_coords = zip(*segment)
        plt.scatter(y_coords, x_coords, color=colors[i], label=f'Segment {i+1}', s=10)
    plt.title('DETECTED AREAS')
    plt.savefig(PATH_TO_SAVE)
    plt.show()         
if __name__ == "__main__":
    main()