import os
from pathlib import Path

import numpy as np


def count_shapes(data_paths):
    """
    Counts the shapes of the data in the three datasets and returns a dictionary with the count of each shape.
    Result:
    RestingState: 99.6% (1080) have shape (1200,12)
    Memory: 99.9% (1085) have shape (405, 12)
    Language: 99.8% (1049) have shape (316, 12)
    """
    shapes_count = {task: dict() for task in data_paths.keys()}
    for task, data_path in data_paths.items():
        for filename in os.listdir(data_path):
            path = Path(data_path) / filename
            data = np.loadtxt(path)
            shapes_count[task][data.shape] = shapes_count[task].get(data.shape, 0) + 1
    return shapes_count


def load_data(data_paths, shapes):
    """
    Loads data from the three datasets, discards the last 6 columns (derivatives),
    transforms degrees to radians for the last 3 columns and returns a dictionary
    with the data for each task.
    """
    data_dict = {task: np.zeros(shape=shapes[task]) for task in data_paths.keys()}
    for task, data_path in data_paths.items():
        i = 0
        for filename in os.listdir(data_path):
            path = Path(data_path) / filename
            data = np.loadtxt(path, usecols=range(6))
            # Aggiunge i dati al dizionario solo se hanno la shape corretta, altrimenti li scarta
            if data.shape == shapes[task][1:]:
                # Transform degrees to radians for the last 3 columns
                data[:, 3:] = np.deg2rad(data[:, 3:])
                data_dict[task][i] = data
                i += 1
    return data_dict


# dictionary with the paths to the three datasets
data_paths = {
    "Resting": "../RECENTRE-main/HCP/RestingStateLR_dataset",
    "Memory": "../RECENTRE-main/HCP/MemoryTaskLR_dataset",
    "Language": "../RECENTRE-main/HCP/LanguageTaskLR_dataset",
}

# shapes holds the results of count_shapes(data_paths) (kept only the >99% shapes)
shapes = {
    "Resting": (1080, 1200, 6),
    "Memory": (1085, 405, 6),
    "Language": (1049, 316, 6),
}

data = load_data(data_paths, shapes)
