import numpy as np
import csv
from pathlib import Path
import copy
import matplotlib.pyplot as plt

#L,W,R : 150,150,1
MAX_LENGTH = 250 
MAX_WIDTH = 250
CENTER_X = int(MAX_WIDTH/2)
CENTER_Y = int(MAX_LENGTH/2)
SOURCE_CORD_X = 50
SOURCE_CORD_Y = 45

#Ze względu na wielkość point clouda muszę dać większą rozdzieczość
RESOLUTION = 1

memory_ogm = np.zeros((MAX_WIDTH, MAX_LENGTH))
new_ogm = copy.deepcopy(memory_ogm)

def load_csv(path_to_file: Path) -> np.array:
    data_list = []
    with open(path_to_file, newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            row = [float(i) for i in row]

            #Magical geometry math
            row[0] = SOURCE_CORD_X - row[0]
            row[1] = row[1] - SOURCE_CORD_Y
            # Z is irrelevant
            # row = [a - b for a,b in zip([50, 45, 50], row)] #Camera Coords
            data_list.append(np.array(row))
    return np.array(data_list)     

def fill_map_with_data(map, data):
    for point in data:
        x = int(point[0]/RESOLUTION) + CENTER_X
        y = int(point[1]/RESOLUTION) + CENTER_Y
        map[x,y] = 1
    return map

scans = load_csv(Path("csvs/custom_room_opened/0.csv"))
memory_ogm = fill_map_with_data(memory_ogm, scans)
plt.xlabel('Y')
plt.ylabel('X')
plt.imshow(memory_ogm, interpolation="nearest",cmap='Blues')
plt.show()

scans_new = load_csv(Path("csvs/custom_room_closed/0.csv"))
new_ogm = fill_map_with_data(new_ogm, scans_new)
plt.xlabel('Y')
plt.ylabel('X')
plt.imshow(new_ogm, interpolation="nearest",cmap='Blues')
plt.show()

print("Case Open->Close (Find 1)")
differece_map = []
for new_row, memory_row in zip(new_ogm, memory_ogm):
    differece_row = [a - b for a,b in zip(new_row, memory_row)]
    differece_map.append(differece_row)

for i in range(MAX_WIDTH):
    for j in range(MAX_LENGTH):
        if differece_map[i][j] == 1:
            print("Detected doors in place: " + str(i+1) + " x " + str(j+1))

plt.xlabel('Y')
plt.ylabel('X')
plt.imshow(differece_map, interpolation="nearest",cmap='grey')
plt.colorbar()
plt.imshow(differece_map)
plt.show()



print("Case Close->Open (Find -1)")
differece_map = []
for new_row, memory_row in zip(new_ogm, memory_ogm):
    differece_row = [b - a for a,b in zip(new_row, memory_row)]
    differece_map.append(differece_row)

for i in range(MAX_WIDTH):
    for j in range(MAX_LENGTH):
        if differece_map[i][j] == -1:
            print("Detected doors in place: " + str(i+1) + " x " + str(j+1))

plt.xlabel('Y')
plt.ylabel('X')
plt.imshow(differece_map, interpolation="nearest",cmap='grey')
plt.colorbar()
plt.imshow(differece_map)
plt.show()
