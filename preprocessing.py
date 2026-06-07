import os
from pathlib import Path

import numpy as np


def load_data(data_paths):
    """
    Loads data from the three datasets, discards the last 6 columns (derivatives),
    transforms degrees to radians for the last 3 columns and returns a dictionary
    where each patient is associated to his three tasks.

    {"patient1": {"R": ..., "M": ..., "L":...}, "patient2": {...}, ...}

    Also filter so that patients that do not have all three tasks are discarded.

    """

    # the per-task frame shapes shared by >99% of the raw runs (after dropping
    # the 6 derivative columns); runs with any other shape are discarded below
    shapes = {
        "R": (1200, 6),
        "M": (405, 6),
        "L": (316, 6),
    }

    patient_dict = {}
    # Count the number of time series in the raw dataset
    initial_runs = 0
    for task, data_path in data_paths.items():
        # Loop over files in the dataset directory
        for filename in os.listdir(data_path):
            if filename.startswith("."):
                continue
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
        patient_id for patient_id, tasks in patient_dict.items() if len(tasks) < 3
    ]
    for patient_id in patients_to_remove:
        del patient_dict[patient_id]

    print(
        f"Filtered patients dataset. Removed {initial_runs - 3 * len(patient_dict)} time series due to shape mismatch or incomplete patients. {3 * len(patient_dict)} left"
    )
    return patient_dict


def get_task_dict(patient_dict, task):
    """Returns {patient_id: data} where data is the time series of the specified task"""
    assert task in ["R", "M", "L"], f"Valid tasks: R, M or L, got {task}"
    return {patient_id: tasks[task] for patient_id, tasks in patient_dict.items()}


if __name__ == "__main__":
    # dictionary with the paths to the three datasets
    data_paths = {
        "R": "../datasets/HCP/RestingStateLR_dataset",
        "M": "../datasets/HCP/MemoryTaskLR_dataset",
        "L": "../datasets/HCP/LanguageTaskLR_dataset",
    }

    # # patient_dict holds complete patients (patients that have all the three tasks recorded)
    patient_dict = load_data(data_paths)
    # # task_dicts has one dictionary for each task, with {patient_id: data}
    task_dicts = {task: get_task_dict(patient_dict, task) for task in data_paths.keys()}
    # save task_dicts to disk
    os.makedirs("datasets", exist_ok=True)
    for task, task_dict in task_dicts.items():
        np.save(f"datasets/{task}_dict.npy", task_dict)
