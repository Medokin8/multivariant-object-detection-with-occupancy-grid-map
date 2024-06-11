import numpy as np
import matplotlib.pyplot as plt

DOORS_THRESHOLD = 0.49
name1 = 'lab15_side_opened'
name2 = 'lab15_main_opened'
folder = 'two_states/'

file1 = 'tmp/' + name1 + '.npy'
file2 = 'tmp/' + name2 + '.npy'
differenece_file_name = 'output_diff_maps_images/' + folder + name1+ '_' + name2 + '.png.png'
door_detected_file_name = 'output_doors_detected_images/' + folder + name1+ '_' + name2 + '.png.png'

def find_lines(binary_mask, min_length=5):
    used_cells = set()
    lines = []
    labeled_mask = np.zeros_like(binary_mask)
    current_label = 0
    
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
binary_mask = np.where(abs(difference) > DOORS_THRESHOLD, 1, 0)

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
plt.imshow(binary_mask, cmap='gray')
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
plt.savefig(door_detected_file_name)
plt.show()


#       simulated_maps

# name1 = 'lab15_main_and_side_opened'
# name2 = 'lab15_all_closed'
# Found 2 lines of at least 5 consecutive cells.
# Line 1: [(34, 42), (34, 43), (34, 44), (34, 45), (34, 46), (34, 47), (34, 48), (34, 49)]
# Line 2: [(44, 37), (45, 37), (46, 37), (47, 37), (48, 37), (49, 37), (50, 37)]

# name1 = 'lab15_all_opened'
# name2 = 'lab15_all_closed'
# Found 2 lines of at least 5 consecutive cells.
# Line 1: [(34, 42), (34, 43), (34, 44), (34, 45), (34, 46), (34, 47), (34, 48), (34, 49), (34, 50), (34, 51), (34, 52), (34, 53), (34, 54)]
# Line 2: [(45, 37), (46, 37), (47, 37), (48, 37), (49, 37), (50, 37)]

# file1 = 'simulated_maps/' + name1 + '.npy'
# file2 = 'simulated_maps/' + name2 + '.npy'
# Found 1 lines of at least 5 consecutive cells.
# Line 1: [(34, 42), (34, 43), (34, 44), (34, 45), (34, 46), (34, 47), (34, 48), (34, 49), (34, 50), (34, 51), (34, 52), (34, 53), (34, 54)]

# name1 = 'lab15_side_opened'
# name2 = 'lab15_all_closed'
# Found 1 lines of at least 5 consecutive cells.
# Line 1: [(45, 37), (46, 37), (47, 37), (48, 37), (49, 37), (50, 37), (51, 37)]

# name1 = 'lab15_main_opened'
# name2 = 'lab15_all_closed'
# Found 1 lines of at least 5 consecutive cells.
# Line 1: [(34, 42), (34, 43), (34, 44), (34, 45), (34, 46), (34, 47), (34, 48), (34, 49)]

# name1 = 'room_opened'
# name2 = 'room_closed'
# Found 1 lines of at least 5 consecutive cells.
# Line 1: [(47, 3), (48, 3), (49, 3), (50, 3), (51, 3), (52, 3), (53, 3), (54, 3), (55, 3), (56, 3), (57, 3), (58, 3), (59, 3), (60, 3), (61, 3), (62, 3), (63, 3), (64, 3), (65, 3), (66, 3), (67, 3)]

# name1 = 'custom_room_opened'
# name2 = 'custom_room_closed'
# Found 1 lines of at least 5 consecutive cells.
# Line 1: [(85, 110), (85, 111), (85, 112), (85, 113), (85, 114), (85, 115), (85, 116), (85, 117), (85, 118), (85, 119), (85, 120), (85, 121), (85, 122), (85, 123), (85, 124), (85, 125), (85, 126), (85, 127), (85, 128), (85, 129), (85, 130), (85, 131), (85, 132), (85, 133), (85, 134), (85, 135), (85, 136), (85, 137), (85, 138)]

# name1 = 'lab15_side_opened'
# name2 = 'lab15_main_opened'
# Found 3 lines of at least 5 consecutive cells.
# Line 1: [(35, 41), (36, 41), (37, 41), (38, 41), (39, 41), (40, 41), (41, 41), (42, 41), (43, 41), (44, 41)]
# Line 2: [(38, 40), (39, 40), (40, 40), (41, 40), (42, 40), (43, 40), (44, 40)]
# Line 3: [(45, 37), (46, 37), (47, 37), (48, 37), (49, 37), (50, 37), (51, 37)]



#       real_maps
# name1 = 'lab15_main_and_side_opened'
# name2 = 'lab15_all_closed'
# Found 2 lines of at least 5 consecutive cells.
# Line 1: [(55, 34), (56, 35), (57, 36), (58, 37), (59, 38), (60, 39), (61, 40), (62, 41)]
# Line 2: [(58, 36), (59, 37), (60, 38), (61, 39), (62, 40), (63, 41)]

# name1 = 'lab15_main_and_side_opened'
# name2 = 'lab15_all_closed'
# Found 2 lines of at least 5 consecutive cells.
# Line 1: [(58, 60), (59, 59), (60, 58), (61, 57), (62, 56), (63, 55)]
# Line 2: [(59, 58), (60, 57), (61, 56), (62, 55), (63, 54)]

# name1 = 'lab15_main_opened'
# name2 = 'lab15_all_closed'
# Found 0 lines of at least 5 consecutive cells.

# name1 = 'lab15_side_opened'
# name2 = 'lab15_all_closed'
# Found 2 lines of at least 5 consecutive cells.
# Line 1: [(59, 58), (60, 57), (61, 56), (62, 55), (63, 54)]
# Line 2: [(59, 59), (60, 58), (61, 57), (62, 56), (63, 55)]

# name1 = 'lab15_glass_opened'
# name2 = 'lab15_glass_closed'
# Found 0 lines of at least 5 consecutive cells.

# name1 = 'lab15_glass_opened'
# name2 = 'lab15_glass_covered'
# Found 0 lines of at least 5 consecutive cells.