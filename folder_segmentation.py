import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import label

P_THRESHOLD = 0.49
TOLERANCE = 1e-6
MIN_NUMBER_OF_CELLS = 3

# MAPS_FOLDER = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/data/glass_doors"
# MAPS_FOLDER = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/data/lab15_real"
# MAPS_FOLDER = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/data/lab15_sim_rotated"
MAPS_FOLDER = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/data/lab15_simulation"

# PATH_TO_SAVE = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/outputs/glass_doors/"
# PATH_TO_SAVE = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/outputs/lab15_real/"
# PATH_TO_SAVE = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/outputs/lab15_sim_rotated/"
PATH_TO_SAVE = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/outputs/lab15_sim/"


def load_maps_from_folder():
    maps = []
    for file in os.listdir(MAPS_FOLDER):
        if file.endswith(".npy"):
            data = np.load(os.path.join(MAPS_FOLDER, file))
            maps.append(data)
    return maps


def init_result_map(maps):
    max_rows = 0
    max_columns = 0

    for map in maps:
        rows, columns = map.shape
        max_rows = max(max_rows, rows)
        max_columns = max(max_columns, columns)

    map = np.zeros([max_rows, max_columns])
    return map


def display_maps(maps):
    i = 1

    if not os.path.exists((PATH_TO_SAVE +f"maps/")):
        os.makedirs((PATH_TO_SAVE +f"maps/"))

    for map in maps:
        plt.clf()
        plt.title(f"map {i}")
        plt.scatter(50, 50, color="red", label="Lidar", s=10)
        global_min = map.min()
        global_max = map.max()

        im = plt.imshow(map, cmap="gray_r", vmin=global_min, vmax=global_max)

        num_ticks = 5
        ticks = np.linspace(global_min, global_max, num_ticks)
        tick_labels = [f'{tick:.2f}' for tick in ticks]
        cbar = plt.colorbar(im)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)

        plt.savefig(PATH_TO_SAVE + "maps/"+ "map_" + str(i) + ".png")
        plt.close()
        i += 1
    
    # i = 1
    # # plt.figure(figsize=(15, 5))
    # for map in maps:
    #     plt.subplot(2, 3, i)
    #     plt.scatter(50, 50, color="red", label="Lidar", s=10)
    #     plt.imshow(map, cmap="gray_r")
    #     plt.title(f"map {i}")
    #     plt.colorbar()
    #     i += 1

    # plt.show()
    plt.clf()

def sanitize_map(map):
    map[np.isinf(map)] = 0
    map = np.nan_to_num(map)
    return map


def segmentation(binary_mask):
    pattern = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    labeled_array, num_features = label(binary_mask, pattern)
    segments = []

    for i in range(1, num_features + 1):
        segment = np.argwhere(labeled_array == i)
        if len(segment) >= MIN_NUMBER_OF_CELLS:
            segments.append(segment)

    return segments, labeled_array


def plot_segments(segment_list, plot_label):
    plt.figure()
    cmap = plt.get_cmap("tab20", len(segment_list))
    cmap.set_under("white")
    colors = [cmap(i) for i in range(len(segment_list))]
    back = np.zeros([100, 100])
    plt.imshow(back, cmap="gray_r")
    plt.scatter(50, 50, color="red", label="Lidar", s=10)
    for i, segment in enumerate(segment_list):
        x_coords, y_coords = zip(*segment)
        plt.scatter(y_coords, x_coords, color=colors[i], label=f"{plot_label} {i+1}", s=10)
    plt.legend(loc='upper right')


def find_most_similar_segment(source_segment, segments):
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
        most_similar_segment = find_most_similar_segment(source_segment, output_segments)
        if most_similar_segment is not None:
            most_similar_segment = np.vstack([most_similar_segment, unique_point])
        else:
            output_segments.append(np.array([unique_point]))

    return output_segments


def main():
    maps = load_maps_from_folder()
    result_map = init_result_map(maps)
    background = np.copy(result_map)
    display_maps(maps)

    segments_list = []
    for int_map1, map in enumerate(maps):
        reflections = np.where(map > 0.9, 1, 0)
        for point in np.argwhere(reflections == 1):
            background[point[0], point[1]] = 1

        for int_map2, map2 in enumerate(maps):
            if (map == map2).all():
                continue
            else:
                difference_map = np.absolute(map - map2)
                difference_map = sanitize_map(difference_map)
                binary_map = np.where(difference_map - P_THRESHOLD > TOLERANCE, 1, 0)
                binary_map = sanitize_map(binary_map)

                plt.figure()
                plt.subplots_adjust(wspace=0.5, hspace=0.5)
                plt.subplot(2, 2, 1)
                plt.scatter(50, 50, color="red", label='Lidar', s=10)
                im = plt.imshow(map, cmap='gray_r')
                plt.title(f'Map {int_map1+1}')

                global_min = map.min()
                global_max = map.max()
                num_ticks = 5
                ticks = np.linspace(global_min, global_max, num_ticks)
                tick_labels = [f'{tick:.2f}' for tick in ticks]
                cbar = plt.colorbar(im)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(tick_labels)


                plt.subplot(2, 2, 2)
                plt.scatter(50, 50, color="red", label='Lidar', s=10)
                im = plt.imshow(map2, cmap='gray_r')
                plt.title(f'Map {int_map2+1}')
                global_min = map2.min()
                global_max = map2.max()
                num_ticks = 5
                ticks = np.linspace(global_min, global_max, num_ticks)
                tick_labels = [f'{tick:.2f}' for tick in ticks]
                cbar = plt.colorbar(im)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(tick_labels)

                plt.subplot(2, 2, 3)
                plt.scatter(50, 50, color="red", label='Lidar', s=10)
                im = plt.imshow(difference_map, cmap='gray_r')
                plt.title('Difference map')
                global_min = difference_map.min()
                global_max = difference_map.max()
                num_ticks = 5
                ticks = np.linspace(global_min, global_max, num_ticks)
                tick_labels = [f'{tick:.2f}' for tick in ticks]
                cbar = plt.colorbar(im)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(tick_labels)

                plt.subplot(2, 2, 4)
                plt.scatter(50, 50, color="red", label='Lidar', s=10)
                im = plt.imshow(binary_map, cmap='gray_r')
                plt.title('Binary map')
                global_min = binary_map.min()
                global_max = binary_map.max()

                cbar = plt.colorbar(im)
                cbar.set_ticks([global_min, global_max])
                cbar.set_ticklabels([f'{global_min:.2f}', f'{global_max:.2f}'])

                plt.savefig(PATH_TO_SAVE + "comparison_maps/"+ "comparison_" + str(int_map1+1) + "_" + str(int_map2+1) + ".png")
                # plt.show()
                plt.close()
                plt.clf()


                if not os.path.exists((PATH_TO_SAVE +f"segment_maps/segemnts_{int_map1+1}_{int_map2+1}/")):
                    os.makedirs((PATH_TO_SAVE +f"segment_maps/segemnts_{int_map1+1}_{int_map2+1}/"))

                segments, segment_labels = segmentation(binary_map)
                cmap = plt.get_cmap('tab20', len(segments))
                cmap.set_under('white')
                colors = [cmap(i) for i in range(len(segments))]
                
                plt.figure()
                plt.subplots_adjust(wspace=0.5)
                plt.imshow(difference_map, cmap='gray_r')
                plt.scatter(50, 50, color="red", label='Lidar', s=10)
                for i, segment in enumerate(segments):
                    x_coords, y_coords = zip(*segment)
                    plt.scatter(y_coords, x_coords, color=colors[i], label=f'Segment {i+1}', s=10)
                plt.title(f'Detected segments on difference map \n (Map {int_map1+1} - Map {int_map2+1})')
                plt.legend(loc='upper right')
                plt.savefig(PATH_TO_SAVE +f"segment_maps/segemnts_{int_map1+1}_{int_map2+1}/" + "detected_segments" + str(int_map1+1) + "_" + str(int_map2+1) + ".png")
                plt.close()
                plt.clf()

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

                plt.figure()
                plt.scatter(50, 50, color="red", label='Lidar', s=10)
                plt.imshow(result_map, cmap='gray_r')
                plt.title('Output map')
                plt.legend(loc='upper right')
                plt.savefig(PATH_TO_SAVE +f"segment_maps/segemnts_{int_map1+1}_{int_map2+1}/" + "result_map" + str(int_map1+1) + "_" + str(int_map2+1) + ".png")
                plt.close()
                plt.clf()

                segments, segment_labels = segmentation(result_map)

                plt.figure()
                cmap = plt.get_cmap('tab20', len(segments))
                cmap.set_under('white')
                colors = [cmap(i) for i in range(len(segments))]
                plt.imshow(difference_map, cmap='gray_r')
                plt.scatter(50, 50, color="red", label='Lidar', s=10)
                for i, segment in enumerate(segments):
                    x_coords, y_coords = zip(*segment)
                    plt.scatter(y_coords, x_coords, color=colors[i], label=f'Segment {i+1}', s=10)
                plt.title(f'Detected segments on result map')
                plt.legend(loc='upper right')
                plt.savefig(PATH_TO_SAVE +f"segment_maps/segemnts_{int_map1+1}_{int_map2+1}/" + "segment_map" + str(int_map1+1) + "_" + str(int_map2+1) + ".png")
                plt.close()
                plt.clf()


    if not os.path.exists((PATH_TO_SAVE +f"object_detection/")):
        os.makedirs((PATH_TO_SAVE +f"object_detection/"))

    plot_segments(segments_list, "Segment")
    plt.title(f"All detedcted segments based on maps 1:{len(maps)}")
    plt.savefig(PATH_TO_SAVE +"object_detection/segment_list_map.png")
    plt.close()
    plt.clf()

    blobs = divide_blobs(segments_list)
    plot_segments(blobs, "Segment")
    plt.title(f"Filtered segments based on maps 1:{len(maps)}")
    plt.savefig(PATH_TO_SAVE +"object_detection/blobs.png")
    plt.close()
    plt.clf()

    cmap = plt.get_cmap("tab20", len(blobs))
    cmap.set_under("white")
    colors = [cmap(i) for i in range(len(blobs))]

    plt.imshow(background, cmap="gray_r")
    plt.scatter(50, 50, color="red", label="Lidar", s=10)
    for i, segment in enumerate(blobs):
        x_coords, y_coords = zip(*segment)
        plt.scatter(y_coords, x_coords, color=colors[i], label=f"Segment {i+1}", s=10)
    plt.title(f"Detected Objects based on maps 1:{len(maps)}")
    plt.legend(loc='upper right')
    plt.savefig(PATH_TO_SAVE +"object_detection/objects_detected.png")
    plt.close()
    plt.clf()


if __name__ == "__main__":
    main()
