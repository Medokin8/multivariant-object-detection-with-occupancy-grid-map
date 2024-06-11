import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

DOORS_THRESHOLD = 0.49
# filename = 'output_results/results_data.txt'
# name1 = 'lab15_main_opened'
# name2 = 'lab15_side_opened'
# folder = 'simulations/'

# file1 = 'simulated_maps/' + name1 + '.npy'
# file2 = 'simulated_maps/' + name2 + '.npy'
# differenece_file_name = 'output_diff_maps_images/' + folder + name1+ '_' + name2 + '.png'
# door_detected_file_name = 'output_doors_detected_images/' + folder + name1+ '_' + name2 + '.png'

filename = 'output_results/results_data_3.txt'
name1 = 'lab15_main_opened'
name2 = 'lab15_side_opened'
folder = 'tmp/'

file1 = 'real_maps/' + name1 + '.npy'
file2 = 'real_maps/' + name2 + '.npy'
differenece_file_name = 'output_diff_maps_images/' + folder + name1+ '_' + name2 + '.png'
door_detected_file_name = 'output_doors_detected_images/' + folder + name1+ '_' + name2 + '.png'


# Load .npy files
data1 = np.load(file1)
data2 = np.load(file2)

# Visualize the content of the .npy files
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.scatter(50, 50, color="red", label='Lidar', s=10)
plt.imshow(data1, cmap='gray_r')
plt.title('Occupancy grid map from first file')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.scatter(50, 50, color="red", label='Lidar', s=10)
plt.imshow(data2, cmap='gray_r')
plt.title('Occupancy grid map from second file')
plt.colorbar()

# Compute and visualize the difference
difference = data1 - data2
rows, columns = data1.shape
binary_mask2 = np.empty([rows, columns])
for row in range(rows):
    for column in range(columns):
        # row=76
        # column=80
        d1 = data1[row,column]
        d2 = data2[row,column]
        difference_value = abs(difference[row, column])
        tolerance = 1e-6
        if difference_value <= 0.5:
            if abs(data1[row, column] - 0.01) < tolerance and abs(data2[row, column] - 0.5) < tolerance:
                binary_mask2[row, column] = 0
            elif abs(data1[row, column] - 0.5) < tolerance and abs(data2[row, column] - 0.01) < tolerance:
                binary_mask2[row, column] = 0   
            elif abs(data1[row, column] - 0.5) < tolerance and abs(data2[row, column] - 0.99) < tolerance:
                binary_mask2[row, column] = 1
            elif abs(data1[row, column] - 0.99) < tolerance and abs(data2[row, column] - 0.5) < tolerance:
                binary_mask2[row, column] = 1
            else:
                binary_mask2[row, column] = 0
        else:
            binary_mask2[row, column] = 1

plt.subplot(2, 2, 3)
plt.scatter(50, 50, color="red", label='Lidar', s=10)
plt.imshow(binary_mask2, cmap='gray_r')
plt.title('Difference between occupancy grid maps')
plt.colorbar()

# Apply threshold to the difference array
binary_mask = np.where(abs(difference) > DOORS_THRESHOLD, 1, 0)
plt.subplot(2, 2, 4)
plt.scatter(50, 50, color="red", label='Lidar', s=10)
plt.imshow(binary_mask, cmap='gray_r')
plt.title(f"Binary mask (threshold {DOORS_THRESHOLD})")
plt.colorbar()
plt.tight_layout()
plt.savefig(differenece_file_name)
plt.show()

# Define the segmentation function
def segmentation(binary_mask):
    labeled_array, num_features = label(binary_mask)
    segments = []
    
    for i in range(1, num_features + 1):
        segment = np.argwhere(labeled_array == i)
        if len(segment) >= 5:  # Filter out small segments
            segments.append(segment)
    
    return segments, labeled_array

# Apply the function to the binary_mask
segments, segment_labels = segmentation(binary_mask)

# Display the found segments
print(f"Found {len(segments)} segments of at least 5 consecutive cells.")
for i, segment in enumerate(segments):
    segment_list = [list(cords) for cords in segment]
    print(f"Segment {i+1}: {segment_list}")

# Visualization with distinct colors for each segment
plt.imshow(data1+data2, cmap='gray_r')
# Define a colormap
cmap = plt.get_cmap('summer', len(segments))
cmap.set_under('white')
colors = [cmap(i) for i in range(len(segments))]
plt.scatter(50, 50, color="red", label='Lidar', s=10)

# Plot each segment in a different color
for i, segment in enumerate(segments):
    x_coords, y_coords = zip(*segment)
    plt.scatter(y_coords, x_coords, color=colors[i], label=f'Segment {i+1}', s=10)

with open(filename, 'a') as file:
    file.write(f"{folder}\n")
    file.write(f"File 1: {name1}\n")
    file.write(f"File 2: {name2}\n")
    file.write(f"Found {len(segments)} segments of at least 5 consecutive cells.\n")
    for i, segment in enumerate(segments):
        segment_list = [list(cords) for cords in segment]
        file.write(f"Segment {i+1}: {segment_list}\n")
    file.write("\n")

# Add a legend
plt.legend()
plt.title('Detected segments in the resulting occupancy grid map')
plt.savefig(door_detected_file_name)
plt.show()