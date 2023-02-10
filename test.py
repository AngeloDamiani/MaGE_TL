import os
import json
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict

folder = "logs"
prefix = "NEW_200_"
models = ["T", "T_ext", "T_hyb", "T_synth"]
black_list = ["T_ext"]


def read_error(subfolder):
    print(f"Reading {subfolder}")
    errors_path = os.path.join(folder, subfolder, "errors.json")
    with open(errors_path, "r") as f:
        folder_errors = json.load(f)
    return folder_errors["errors"]


def read_errors(folder, prefix):
    pattern = f"^{prefix}*"
    errors = defaultdict(list)
    for subfolder in os.listdir(folder):
        if re.match(pattern, subfolder):
            folder_errors = read_error(subfolder)
            for i, name in enumerate(models):
                errors[name].append(folder_errors[i])
    return errors


def plot_errors(errors):
    fig, ax = plt.subplots()
    for model, error in errors.items():
        if not model in black_list:
            model_error = np.array(error)
            errors_mean = np.mean(model_error, axis=0)
            errors_min = np.min(model_error, axis=0)
            errors_max = np.max(model_error, axis=0)
            ax.plot(sorted(errors_mean,reverse=True), label=model)
            #ax.fill_between(range(len(errors_mean)), errors_min, errors_max, alpha=0.5)
            ax.legend()
            ax.grid()
    
    fig.set_size_inches(10.5, 6.5)
    plt.show()


if __name__ == "__main__":
    errors = read_errors(folder, prefix)
    plot_errors(errors)
