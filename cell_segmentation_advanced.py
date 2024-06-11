import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

DOORS_THRESHOLD = 0.49

filename = 'output_results/results_data.txt'
name1 = 'lab15_all_opened'
name2 = 'lab15_side_opened'
name3 = 'lab15_main_opened'
folder = 'simulations/'

# real_maps
# simulated_maps
file1 = 'real_maps/' + name1 + '.npy'
file2 = 'real_maps/' + name2 + '.npy'
file3 = 'real_maps/' + name3 + '.npy'

differenece_file_name = 'output_diff_maps_images/translation.png'
door_detected_file_name = 'output_doors_detected_images/translation.png'

data1 = np.load(file1)
data2 = np.load(file2)
data3 = np.load(file3)

plt.subplot(2, 3, 1)
plt.scatter(50, 50, color="red", label='Lidar', s=10)
plt.imshow(data1, cmap='gray_r')
plt.title('Occupancy grid map from first file')
plt.colorbar()

plt.subplot(2, 3, 2)
plt.scatter(50, 50, color="red", label='Lidar', s=10)
plt.imshow(data2, cmap='gray_r')
plt.title('Occupancy grid map from second file')
plt.colorbar()

plt.subplot(2, 3, 3)
plt.scatter(50, 50, color="red", label='Lidar', s=10)
plt.imshow(data3, cmap='gray_r')
plt.title('Occupancy grid map from third file')
plt.colorbar()

dif12 = data1 - data2
dif13 = data1 - data3
dif23 = data2 - data3

plt.subplot(2, 3, 4)
plt.scatter(50, 50, color="red", label='Lidar', s=10)
plt.imshow(dif12, cmap='gray_r')
plt.title('Difference between OGMs 1 2')
plt.colorbar()

plt.subplot(2, 3, 5)
plt.scatter(50, 50, color="red", label='Lidar', s=10)
plt.imshow(dif13, cmap='gray_r')
plt.title('Difference between OGMs 1 3')
plt.colorbar()

plt.subplot(2, 3, 6)
plt.scatter(50, 50, color="red", label='Lidar', s=10)
plt.imshow(dif23, cmap='gray_r')
plt.title('Difference between OGMs 2 3')
plt.colorbar()
plt.show()

bm12 = np.where(abs(dif12) > DOORS_THRESHOLD, 1, 0)
bm13 = np.where(abs(dif13) > DOORS_THRESHOLD, 1, 0)
bm23 = np.where(abs(dif23) > DOORS_THRESHOLD, 1, 0)

plt.subplot(1, 4, 1)
plt.scatter(50, 50, color="red", label='Lidar', s=10)
plt.imshow(bm12, cmap='gray_r')
plt.title(f"Binary mask (threshold {DOORS_THRESHOLD})")
plt.colorbar()

plt.subplot(1, 4, 2)
plt.scatter(50, 50, color="red", label='Lidar', s=10)
plt.imshow(bm13, cmap='gray_r')
plt.title(f"Binary mask (threshold {DOORS_THRESHOLD})")
plt.colorbar()

plt.subplot(1, 4, 3)
plt.scatter(50, 50, color="red", label='Lidar', s=10)
plt.imshow(bm23, cmap='gray_r')
plt.title(f"Binary mask (threshold {DOORS_THRESHOLD})")
plt.colorbar()



def spread_val(omega_binary):
    local = omega_binary.copy()
    rows, cols = local.shape
    for row in range(rows):
        for col in range(cols):
            if omega_binary[row, col] == 1:
                if row > 0 and col > 0:
                    local[row - 1, col - 1] = 1
                if row > 0:
                    local[row - 1, col] = 1
                if row > 0 and col < cols - 1:
                    local[row - 1, col + 1] = 1
                if col > 0:
                    local[row, col - 1] = 1
                if col < cols - 1:
                    local[row, col + 1] = 1
                if row < rows - 1 and col > 0:
                    local[row + 1, col - 1] = 1
                if row < rows - 1:
                    local[row + 1, col] = 1
                if row < rows - 1 and col < cols - 1:
                    local[row + 1, col + 1] = 1
    return local
                


plt.subplot(1, 4, 4)
omega_binary = np.maximum(np.maximum(bm12, bm13), bm23)
omega_binary = spread_val(omega_binary)
plt.scatter(50, 50, color="red", label='Lidar', s=10)
plt.imshow(omega_binary, cmap='gray_r')
plt.title(f"omega_binary")
plt.colorbar()
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
segments, segment_labels = segmentation(omega_binary)

# Display the found segments
print(f"Found {len(segments)} segments of at least 5 consecutive cells.")
for i, segment in enumerate(segments):
    segment_list = [list(cords) for cords in segment]
    print(f"Segment {i+1}: {segment_list}")

# Visualization with distinct colors for each segment
plt.imshow(data1+data2+data3, cmap='gray_r')
# Define a colormap
cmap = plt.get_cmap('summer', len(segments))
cmap.set_under('white')
colors = [cmap(i) for i in range(len(segments))]
plt.scatter(50, 50, color="red", label='Lidar', s=10)

# Plot each segment in a different color
for i, segment in enumerate(segments):
    x_coords, y_coords = zip(*segment)
    plt.scatter(y_coords, x_coords, color=colors[i], label=f'Segment {i+1}', s=10)

# Add a legend
plt.legend()
plt.title('Detected segments in the resulting occupancy grid map')
plt.show()

