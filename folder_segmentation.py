import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import label

DOORS_THRESHOLD = 0.49
P_INIT = 0.5
MAPS_FOLDER = "/home/nikodem/Documents/door-detection-with-ogm/folder_segmentation/lab15_simulation"
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
    map = np.empty([max_rows,max_columns])
    return map 


def display_maps(maps):
    i = 1
    for map in maps:
        plt.subplot(2, 3, i)
        plt.imshow(map, cmap='gray_r')
        plt.title(f'map {i}')
        plt.colorbar()
        i+=1
    
    plt.show()


def segmentation(binary_mask):
    labeled_array, num_features = label(binary_mask)
    segments = []
    
    for i in range(1, num_features + 1):
        segment = np.argwhere(labeled_array == i)
        if len(segment) >= 5:  # Filter out small segments
            segments.append(segment)
    
    return segments, labeled_array


def main():
    maps = load_maps_from_folder()
    result_map = init_result_map(maps)
    # display_maps(maps)

    # plt.imshow(maps[0] - maps[1], cmap='gray_r')
    # plt.title('??? map')
    # plt.colorbar()
    # plt.show()
    index = 1
    for map in maps: 
        for map2 in maps:
            if (map == map2).all():
                continue
            else:
                difference_map = np.absolute(map - map2)
                binary_map = np.where(difference_map - DOORS_THRESHOLD > TOLERANCE, 1, 0)

                # plt.subplot(2, 2, 1)
                # plt.imshow(map, cmap='gray_r')
                # plt.title('Map 1')
                # plt.colorbar()

                # plt.subplot(2, 2, 2)
                # plt.imshow(map2, cmap='gray_r')
                # plt.title('Map 2')
                # plt.colorbar()

                # plt.subplot(2, 2, 3)
                # plt.imshow(difference_map, cmap='gray_r')
                # plt.title('Difference map')
                # plt.colorbar()

                # plt.subplot(2, 2, 4)
                # plt.imshow(binary_map, cmap='gray_r')
                # plt.title('Binary map')
                # plt.colorbar()
                # plt.show()
        
                segments, segment_labels = segmentation(binary_map)

                cmap = plt.get_cmap('summer', len(segments))
                cmap.set_under('white')
                colors = [cmap(i) for i in range(len(segments))]

                plt.subplot(1, 3, 1)
                plt.imshow(difference_map, cmap='gray_r')
                plt.scatter(50, 50, color="red", label='Lidar', s=10)
                for i, segment in enumerate(segments):
                    x_coords, y_coords = zip(*segment)
                    plt.scatter(y_coords, x_coords, color=colors[i], label=f'Segment {i+1}', s=10)
                plt.title('Map with segments')


                plt.subplot(1, 3, 2)
                plt.imshow(segment_labels, cmap='gray_r')
                plt.title('segment_labels')

                for segment in segments:
                    index_flag = False
                    for point in segment:
                        if result_map[point[0], point[1]] == 0:
                            result_map[point[0], point[1]] = 1#index
                    #         index_flag = True
                    # if index_flag is True:
                    #     index += 1
                    


                plt.subplot(1, 3, 3)
                plt.imshow(result_map, cmap='gray_r')
                plt.title('result_map')
                plt.colorbar()
                plt.show()

if __name__ == "__main__":
    main()