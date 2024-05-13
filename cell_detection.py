import numpy as np
import csv
from pathlib import Path
import copy

MAX_LENGTH = 140
MAX_WIDTH = 140
CENTER = 70

RESOLUTION = 0.1

memory_ogm = np.zeros((MAX_WIDTH, MAX_LENGTH))
new_ogm = copy.deepcopy(memory_ogm)


def load_csv(path_to_file: Path) -> np.array:
    data_list = []
    with open(path_to_file, newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            row = [float(i) for i in row]
            data_list.append(np.array(row))

    return np.array(data_list)     

scans = load_csv(Path("csvs/custom_room_closed/0.csv"))
