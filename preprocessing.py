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
    where each patient is associated to his three tasks.

    {"patient1": {"Resting": ..., "Memory": ..., "Language":...}, "patient2": {...}, ...}

    Also filter so that patients that do not have all three tasks are discarded.

    """
    patient_dict = {}
    # Count the number of time series in the raw dataset
    initial_runs = 0
    for task, data_path in data_paths.items():
        # Loop over files in the dataset directory
        for filename in os.listdir(data_path):
            initial_runs += 1
            path = Path(data_path) / filename
            # Load only the first 6 columns
            data = np.loadtxt(path, usecols=range(6))
            # Removes time series with shapes different from the ones specified in <shapes>
            if data.shape == shapes[task]:
                # Transform degrees to radians for the last 3 columns
                data[:, 3:] = np.deg2rad(data[:, 3:])
                # removes ".txt"
                patient_id = filename[:-4]
                # builds dictionary entry or updates it if already created
                patient_dict[patient_id] = patient_dict.get(patient_id, dict())
                patient_dict[patient_id][task] = data

    # Removes patients that miss one or more of the three tasks
    patients_to_remove = [
        patient_id
        for patient_id, tasks in patient_dict.items()
        if len(tasks) < len(data_paths)
    ]
    for patient_id in patients_to_remove:
        del patient_dict[patient_id]

    print(
        f"Filtered patients dataset. Removed {initial_runs - 3 * len(patient_dict)} time series due to shape mismatch or incomplete patients. {3 * len(patient_dict)} left"
    )
    return patient_dict


def get_task_dict(patient_dict, task):
    """Returns a dictionary with {patient_id: data} where data is the time series of the specified task"""
    assert task in ["Resting", "Memory", "Language"], (
        f"Valid tasks: Resting, Memory or Language, got {task}"
    )
    task_dict = {}
    for id, task in patient_dict.items():
        task_dict[id] = patient_dict[id][task]

    return task_dict


# dictionary with the paths to the three datasets
data_paths = {
    "Resting": "../RECENTRE-main/HCP/RestingStateLR_dataset",
    "Memory": "../RECENTRE-main/HCP/MemoryTaskLR_dataset",
    "Language": "../RECENTRE-main/HCP/LanguageTaskLR_dataset",
}

# shapes holds the results of count_shapes(data_paths) (kept only the >99% shapes)
shapes = {
    "Resting": (1200, 6),
    "Memory": (405, 6),
    "Language": (316, 6),
}

if __name__ == "__main__":
    # patient_dict holds complete patients (patients that have all the three tasks recorded)
    patient_dict = load_data(data_paths, shapes)
    # task_dicts one dictionary for each task, with {patient_id: data}
    task_dicts = {task: get_task_dict(patient_dict, task) for task in data_paths.keys()}
