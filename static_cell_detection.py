import numpy as np
import matplotlib.pyplot as plt
DOORS_THRESHOLD = -0.2

# Load .npy files
file2 = 'simulated_maps/opened.npy'
file1 = 'simulated_maps/closed.npy'
data1 = np.load(file1)
data2 = np.load(file2)

# Visualize the content of the .npy files
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(data1, cmap='gray')
plt.title('Content of file1.npy')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(data2, cmap='gray')
plt.title('Content of file2.npy')
plt.colorbar()

# Compute and visualize the difference
difference = data1 - data2
plt.subplot(2, 2, 3)
plt.imshow(difference, cmap='gray')
plt.title('Difference between file1.npy and file2.npy')
plt.colorbar()

# Apply threshold to the difference array
threshold = DOORS_THRESHOLD
binary_mask = np.where(difference > threshold, 1, 0)

plt.subplot(2, 2, 4)
img_mask = plt.imshow(binary_mask, cmap='gray')
plt.title('Binary mask (threshold 0.2)')
plt.colorbar()

plt.show()
