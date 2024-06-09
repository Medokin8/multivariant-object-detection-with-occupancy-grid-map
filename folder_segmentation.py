import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import label

DOORS_THRESHOLD = 0.49
P_INIT = 0.5
# MAPS_FOLDER = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/lab15_simulation"
MAPS_FOLDER = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/lab15_sim_rotated"
# MAPS_FOLDER = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/glass_doors"
# MAPS_FOLDER = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/lab15_real"

# PATH_TO_SAVE = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/simulation_doors.png"
PATH_TO_SAVE = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/simulation_rotated_doors.png"
# PATH_TO_SAVE = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/glass_doors.png"
# PATH_TO_SAVE = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/real_doors.png"

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


def plot_segments(segment_list, plot_label):
    plt.figure(figsize=(15, 5))
    cmap = plt.get_cmap('tab20', len(segment_list))
    cmap.set_under('white')
    colors = [cmap(i) for i in range(len(segment_list))]
    back = np.zeros([100,100])
    plt.imshow(back, cmap='gray_r')
    plt.scatter(50, 50, color="red", label='Lidar', s=10)
    for i, segment in enumerate(segment_list):
        x_coords, y_coords = zip(*segment)
        plt.scatter(y_coords, x_coords, color=colors[i], label=f'{plot_label} {i+1}', s=10)
    plt.legend()
    

# def divide_blobs(segment_list):
#     # Removal of segments witohout common points
#     segments_without_common_points =[]
#     segments_with_common_points = []
#     for i, segment in enumerate(segment_list):
#         common_point_flag = False
#         for j, segment2 in enumerate(segment_list):
#             if i != j :
#                 for point in segment:
#                     for point2 in segment2:
#                         if (point == point2).all():
#                             common_point_flag = True

#         if common_point_flag is False:
#             segments_without_common_points.append(segment)
#         else:
#             segments_with_common_points.append(segment)

#     print(segments_without_common_points)
#     print(segments_with_common_points)
#     plot_segments(segments_without_common_points, "A")
#     plt.show()

#     # Removal of segments that are inside other segmen
#     lost_segments = []
#     for i, segment in enumerate(segments_with_common_points):
#         for j, segment2 in enumerate(segments_with_common_points):
#             if i != j :
#                 num_of_common = 0
#                 for point in segment:
#                     for point2 in segment2:
#                         if (point == point2).all():
#                             num_of_common += 1

#                 if num_of_common == len(segment):
#                     local = np.copy(segment)
#                     segments_without_common_points.append(local)
    
#     print(segments_without_common_points)
#     plot_segments(segments_without_common_points, "A")
#     plt.show()

#     print(lost_segments)
#     plot_segments(lost_segments, "L")
#     plt.show()

#     return segments_without_common_points

def find_most_similar_segment(unique_point, source_segment, segments):
    # Compute similarity based on the number of common points
    source_points_set = set(tuple(point) for point in source_segment)
    max_similarity = 0
    most_similar_segment = None
    for segment in segments:
        common_points = source_points_set.intersection(set(tuple(point) for point in segment))
        similarity = len(common_points)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_segment = segment
    return most_similar_segment

def divide_blobs(segment_list):
    included_points = set()
    output_segments = []
    single_point_segments = []

    for segment in segment_list:
        segment_points = [tuple(point) for point in segment]
        unique_points = [point for point in segment_points if point not in included_points]
        unique_segment = np.array(unique_points)
        if len(unique_segment) > 1:
            output_segments.append(unique_segment)
            included_points.update(unique_points)
        elif len(unique_segment) == 1:
            single_point_segments.append((unique_segment[0], segment))

    for unique_point, source_segment in single_point_segments:
        most_similar_segment = find_most_similar_segment(unique_point, source_segment, output_segments)
        if most_similar_segment is not None:
            most_similar_segment = np.vstack([most_similar_segment, unique_point])
        else:
            output_segments.append(np.array([unique_point]))
    
    return output_segments



def main():
    maps = load_maps_from_folder()
    result_map = init_result_map(maps)
    background =np.copy(result_map)
    display_maps(maps)

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

                # cmap = plt.get_cmap('tab20', len(segments))
                # cmap.set_under('white')
                # colors = [cmap(i) for i in range(len(segments))]
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

                segments, segment_labels = segmentation(result_map)

                # plt.subplot(1, 4, 4)
                # plt.scatter(50, 50, color="red", label='Lidar', s=10)
                # cmap = plt.get_cmap('tab20', len(segments))
                # cmap.set_under('white')
                # colors = [cmap(i) for i in range(len(segments))]
                # plt.imshow(difference_map, cmap='gray_r')
                # plt.scatter(50, 50, color="red", label='Lidar', s=10)
                # for i, segment in enumerate(segments):
                #     x_coords, y_coords = zip(*segment)
                #     plt.scatter(y_coords, x_coords, color=colors[i], label=f'Segment {i+1}', s=10)
                # plt.title('Map segments_list')
                # plt.show()

    segments_list = remove_duplicates(segments_list)
    print(len(segments_list))
    plot_segments(segments_list, "Segment")
    plt.show()

    blobs = divide_blobs(segments_list)
    plot_segments(blobs, "Segment")
    plt.show()

if __name__ == "__main__":
    main()