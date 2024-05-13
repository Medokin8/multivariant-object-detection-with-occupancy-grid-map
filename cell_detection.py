import numpy as np
import csv
from pathlib import Path
import copy
import matplotlib.pyplot as plt

MAX_LENGTH = 250
MAX_WIDTH = 250
CENTER_X = int(MAX_WIDTH/2)
CENTER_Y = int(MAX_LENGTH/2)

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
            row[0] = 50 - row[0]
            row[1] = row[1] - 45
            # Z is irrelevant
            # row = [a - b for a,b in zip([50, 45, 50], row)] #Camera Coords
            data_list.append(np.array(row))
    return np.array(data_list)     


def fill_map_with_data(map, data):
    for point in data:
        x = int(point[0]/RESOLUTION) + CENTER_X
        y = int(point[1]/RESOLUTION) + CENTER_Y
        map[x,y] = 100
    return map

scans = load_csv(Path("csvs/custom_room_opened/0.csv"))
memory_ogm = fill_map_with_data(memory_ogm, scans)
plt.xlabel('X')
plt.ylabel('Y')
plt.imshow(memory_ogm, interpolation="nearest",cmap='Blues')
plt.show()

scans_new = load_csv(Path("csvs/custom_room_closed/0.csv"))
new_ogm = fill_map_with_data(new_ogm, scans_new)
plt.xlabel('X')
plt.ylabel('Y')
plt.imshow(new_ogm, interpolation="nearest",cmap='Greens')
plt.show()

differece_map = []
for new_row, memory_row in zip(new_ogm, memory_ogm):
    differece_row = [a - b for a,b in zip(new_row, memory_row)]
    differece_map.append(differece_row)
plt.xlabel('X')
plt.ylabel('Y')
plt.imshow(differece_map, interpolation="nearest",cmap='grey')
plt.show()