import numpy as np
import matplotlib.pyplot as plt

DOORS_THRESHOLD = -0.2
file1 = 'simulated_maps/lab15_main_and_side_opened.npy'
file2 = 'simulated_maps/lab15_all_closed.npy'
differenece_file_name = 'output_diff_maps_images/' + 'lab15_main_and_side' + '_difference.png'
door_detected_file_name = 'output_doors_detected_images' + 'lab15_main_and_side' + '_difference.png'

def find_lines(binary_mask, min_length=5):
    used_cells = set()
    lines = []
    labeled_mask = np.zeros_like(binary_mask)
    current_label = 1

    def mark_used(cells, label):
        for cell in cells:
            used_cells.add(cell)
            labeled_mask[cell] = label

    def is_used(cell):
        return cell in used_cells

    def search_direction(start, direction):
        x, y = start
        dx, dy = direction
        line = []
        while 0 <= x < binary_mask.shape[0] and 0 <= y < binary_mask.shape[1] and binary_mask[x, y] == 0 and not is_used((x, y)):
            line.append((x, y))
            x += dx
            y += dy
        return line

    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # right, down, down-right, down-left

    for x, y in zip(*interesting_cells):
        if is_used((x, y)):
            continue

        for direction in directions:
            line = search_direction((x, y), direction)
            if len(line) >= min_length:
                lines.append(line)
                mark_used(line, current_label)
                current_label += 1

    return lines, labeled_mask


# Load .npy files
data1 = np.load(file1)
data2 = np.load(file2)

# Visualize the content of the .npy files
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(data1, cmap='gray')
plt.title('Occupancy grid map from first file')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(data2, cmap='gray')
plt.title('Occupancy grid map from  second file')
plt.colorbar()

# Compute and visualize the difference
difference = data1 - data2
plt.subplot(2, 2, 3)
plt.imshow(difference, cmap='gray')
plt.title('Difference between occupancy grid maps')
plt.colorbar()

# Apply threshold to the difference array
threshold = DOORS_THRESHOLD
binary_mask = np.where(difference > threshold, 1, 0)

plt.subplot(2, 2, 4)
plt.imshow(binary_mask, cmap='gray')
plt.title('Binary mask (threshold 0.2)')
plt.colorbar()
plt.tight_layout()
plt.savefig(differenece_file_name)
plt.show()

interesting_cells = np.where(binary_mask == 0)
coordinates = list(zip(*interesting_cells))

# Apply the function to the binary_mask
lines, labeled_mask = find_lines(binary_mask)

# Display the found lines
print(f"Found {len(lines)} lines of at least 5 consecutive cells.")
for i, line in enumerate(lines):
    print(f"Line {i+1}: {line}")

# Visualization with distinct colors for each line
plt.imshow(difference, cmap='gray')
# Define a colormap
cmap = plt.get_cmap('summer', len(lines))
cmap.set_under('white')
colors = [cmap(i) for i in range(len(lines))]

# Plot each line in a different color
for i, line in enumerate(lines):
    x_coords, y_coords = zip(*line)
    plt.scatter(y_coords, x_coords, color=colors[i], label=f'Door {i+1}', s=10)

# Add a legend
plt.legend()
plt.title('Detected doors in the resulting occupancy grid map')
plt.show()