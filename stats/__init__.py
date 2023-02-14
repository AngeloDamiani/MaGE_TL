import os
import json
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict

model_names = ["T", "T_ext", "T_hyb", "T_synth"]


def read_error(folder, subfolder):
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
            folder_errors = read_error(folder, subfolder)
            for i, name in enumerate(model_names):
                errors[name].append(folder_errors[i])
    return errors


def plot_errors_vs_H(description, folder, skip_models=[], average=False):
    fig, ax = plt.subplots()

    label_it = True
    for h, runs in description.items():
        h_errors = defaultdict(list)
        for subfolder in runs:
            errors = read_error(folder, subfolder)
            for i, model in enumerate(model_names):
                h_errors[model].append(sum(errors[i]) / len(errors[i]))        

        cmap = plt.get_cmap("tab10")
        for i, model in enumerate(model_names):
            if not model in skip_models:
                if average:
                    h_errors[model] = [sum(h_errors[model]) / len(h_errors[model])]

                label = model if label_it else None
                x = [h] * len(h_errors[model])
                y = h_errors[model]
                ax.scatter(x, y, label=label, color=cmap(i))
                ax.legend()
        
        label_it = False

    fig.set_size_inches(10.5, 6.5)
    plt.grid(True)
    plt.show()

def plot_errors(errors, skip_models=[]):
    fig, ax = plt.subplots()
    for model, error in errors.items():
        if not model in skip_models:
            model_error = np.array(error)
            errors_mean = np.mean(model_error, axis=0)
            error_std = np.std(model_error, axis=0)

            errors_mean, error_std = sort_together(errors_mean, error_std)

            error_min = [mean - error_std[i] for i, mean in enumerate(errors_mean)]
            error_max = [mean + error_std[i] for i, mean in enumerate(errors_mean)]

            ax.plot(errors_mean, label=model)
            ax.fill_between(range(len(errors_mean)), error_min, error_max, alpha=0.5)
            ax.legend()
            ax.grid()

    fig.set_size_inches(10.5, 6.5)
    plt.show()


def sort_together(errors_mean, error_std):
    return (
        list(i)
        for i in zip(
            *sorted(
                zip(errors_mean, error_std),
                reverse=True,
                key=lambda dual: dual[0],
            )
        )
    )
