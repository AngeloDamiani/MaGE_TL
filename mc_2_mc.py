import json
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from config_gen import get_config
from mapping.datasets import RLDatasetFormatter
from mapping.models import Discriminator, TransitionModel, LitAutoEncoder

# Read params from the terminal
parser = argparse.ArgumentParser(
    description="Training a Mapping with random hyper params."
)
parser.add_argument("id", type=int, default=0, help="Id of the current run")
parser.add_argument("prefix", type=str, default="A", help="Prefix for current run")
parser.add_argument("mode", type=str, default="random", help="Config mode")
args = parser.parse_args()

# Generate a random hyperparameter configuration
hyper_params = get_config(args.id, args.prefix, mode='default')
hyp = argparse.Namespace(**hyper_params)

# Ensure the logging dir exists
log_path = Path(hyp.log_dir)
log_path.mkdir(parents=True, exist_ok=True)

# Store the randomly generated config
with open(log_path / "config.json", "w") as f:
    json.dump(hyper_params, f)

# Define some QoL functions
load_csv = lambda x: RLDatasetFormatter().from_csv(x)
log_dir = lambda x: f"{hyp.log_dir}/{x}"

# Define dataset-specific metadata
s_s_max = torch.tensor([1.0, 1.0, 8.0])
s_s_min = torch.tensor([-1.0, -1.0, -8.0])
s_a_max = torch.tensor([2.0])
s_a_min = torch.tensor([-2.0])

t_s_max = torch.tensor([0.6, 0.07])
t_s_min = torch.tensor([-1.2, -0.07])
t_a_max = torch.tensor([1.0])
t_a_min = torch.tensor([-1.0])

# Load dataset providers
data_formatter_s = load_csv("data/UntrainedPendDataset5000.csv").normalize_data(
    s_s_max, s_s_min, s_a_max, s_a_min
)
data_formatter_s_val = load_csv("data/UntrainedPendDataset5000_2.csv").normalize_data(
    s_s_max, s_s_min, s_a_max, s_a_min
)

data_formatter_t = load_csv("data/UntrainedMCDataset500.csv").normalize_data(
    t_s_max, t_s_min, t_a_max, t_a_min
)
# H = 100
# data_formatter_t.s = data_formatter_t.s[:H]
# data_formatter_t.a = data_formatter_t.a[:H]
# data_formatter_t.r = data_formatter_t.r[:H]
# data_formatter_t.s1 = data_formatter_t.s1[:H]


data_formatter_t_val = load_csv("data/UntrainedMCDataset5000.csv").normalize_data(
    t_s_max, t_s_min, t_a_max, t_a_min
)
data_formatter_t_test = load_csv("data/UntrainedMCDataset50000.csv").normalize_data(
    t_s_max, t_s_min, t_a_max, t_a_min
)


s_s_size = data_formatter_s.state_size
s_a_size = data_formatter_s.action_size
t_s_size = data_formatter_t.state_size
t_a_size = data_formatter_t.action_size

# Train the transition model
dataset_t = data_formatter_t.as_transitions()
T = TransitionModel(s_dim=t_s_size, a_dim=t_a_size, lr=hyp.T_lr)
T.train_model(
    dataset_t,
    logs_dir=log_dir("T"),
    epochs=hyp.T_epochs,
    batch_size=hyp.T_batch_size,
)

# Pre-train Discriminator
disc_dataset = data_formatter_t.transition_as_valid()
D = Discriminator(lr=hyp.D_lr, s_dim=t_s_size, a_dim=t_a_size)
D.train_model(
    disc_dataset,
    logs_dir=log_dir("Discriminator"),
    epochs=hyp.D_epochs_pretrain,
    batch_size=hyp.D_batch_size_pretrain,
)

# Create Auto-encoder model
AE = LitAutoEncoder(
    D.as_dict()["func"],
    T.as_dict()["func"],
    (hyp.lmd_R, hyp.lmd_T, hyp.lmd_D),
    lr=hyp.AE_lr,
    s_s_size=s_s_size,
    s_a_size=s_a_size,
    t_s_size=t_s_size,
    t_a_size=t_a_size,
)

###################### ADVERSARIAL TRAINING ###########################

for i in range(hyp.iterations):

    # Train Auto-encoder
    AE.D = D.as_dict()["func"]
    AE.train_model(
        dataset=data_formatter_s.transition_identity(),
        batch_size=hyp.AE_batch_size,
        epochs=hyp.AE_epochs,
        logs_dir=log_dir("AE"),
    )

    # Map using the learned mapping
    M = AE.as_dict()["M"]
    test_dataset_s = data_formatter_s_val.transition_identity()
    test_dataset_s.shuffle()
    sas, _ = test_dataset_s[: len(data_formatter_t.as_transitions())]
    with torch.no_grad():
        synthetic_data = M(sas)

    # Create a dataset to train the discriminator
    r_synth = torch.zeros((synthetic_data.shape[0], 1))
    s_synth, a_synth, s1_synth = torch.split(
        synthetic_data, [t_s_size, t_a_size, t_s_size], 1
    )
    data_formatter_synth = RLDatasetFormatter([s_synth, a_synth, r_synth, s1_synth])
    disc_dataset = data_formatter_t.transition_as_valid()
    fake_samples = data_formatter_synth.transition_as_fake()
    disc_dataset = disc_dataset.merge(fake_samples)

    # Train the discriminator
    D.train_model(
        disc_dataset,
        logs_dir=log_dir("Discriminator"),
        epochs=hyp.D_epochs,
        batch_size=hyp.D_batch_size,
    )

##################### TRAINING END #########################

# SYNTHESIZE DATA
dataset_s_2 = data_formatter_s_val.transition_identity()
AE_dict = AE.as_dict()
sas, _ = dataset_s_2[:]
with torch.no_grad():
    reconstructed_data = AE_dict["func"](sas)
    encoded_data = AE_dict["M"](sas)

# RECONSTRUCTED DATA
s_rec, a_rec, s1_rec = torch.split(
    reconstructed_data, [s_s_size, s_a_size, s_s_size], 1
)
# ENCODED DATA
s_t_code, a_t_code, s1_t_code = torch.split(
    encoded_data, [t_s_size, t_a_size, t_s_size], 1
)
r_code = torch.zeros(len(s1_t_code), 1)
synth_data_formatter = RLDatasetFormatter([s_t_code, a_t_code, r_code, s1_t_code])
# ORIGINAL DATA
s, a, s1 = torch.split(sas, [s_s_size, s_a_size, s_s_size], 1)

################### ADDITIONAL MODELS #####################################

# TRAIN SYNTHETIC MODEL
synth_dataset_t = synth_data_formatter.as_transitions()
T_synth = TransitionModel(lr=hyp.T_lr, s_dim=t_s_size, a_dim=t_a_size)
T_synth.train_model(
    dataset=synth_dataset_t,
    batch_size=hyp.T_batch_size,
    epochs=hyp.T_epochs,
    logs_dir=log_dir("T_synth"),
)

# TRAIN HYBRID MODEL
# Create a hybrid dataset
dataset_t = data_formatter_t.as_transitions()
hybrid_dataset = synth_dataset_t.merge(dataset_t)
# Train a model on the hybrid dataset
T_hyb = TransitionModel(lr=hyp.T_lr, s_dim=t_s_size, a_dim=t_a_size)
T_hyb.train_model(
    dataset=hybrid_dataset,
    batch_size=hyp.T_batch_size,
    epochs=hyp.T_epochs,
    logs_dir=log_dir("T_hyb"),
)

# TRAIN EXTENDED MODEL
# Create an extended version of the target dataset and train the model on it
ext_dataset_t = data_formatter_t_val.as_transitions()

# Train a model on the hybrid dataset
T_ext = TransitionModel(lr=hyp.T_lr, s_dim=t_s_size, a_dim=t_a_size)
T_ext.train_model(
    dataset=ext_dataset_t,
    batch_size=hyp.T_batch_size,
    epochs=hyp.T_epochs,
    logs_dir=log_dir("T_ext"),
)

###########################  PLOTS #######################################

# Plot the samples vs Reconstructed samples
fig, axs = plt.subplots(2 * s_s_size + s_a_size, 2)
fig.suptitle("Source Samples vs Reconstructed Samples", fontsize=16)
for i in range(s_s_size):
    axs[i, 0].plot(s[:, i].detach().numpy())
    axs[i, 0].set_title(f"s[{i}]")
    axs[i, 0].grid()
    axs[i, 1].plot(s_rec[:, i].detach().numpy())
    axs[i, 1].set_title(f"s_rec[{i}]")
    axs[i, 1].grid()
for i in range(s_s_size, s_a_size + s_s_size):
    index_data = i - s_s_size
    axs[i, 0].plot(a[:, index_data].detach().numpy())
    axs[i, 0].set_title(f"a[{index_data}]")
    axs[i, 0].grid()
    axs[i, 1].plot(a_rec[:, index_data].detach().numpy())
    axs[i, 1].set_title(f"a_rec[{index_data}]")
    axs[i, 1].grid
for i in range(s_a_size + s_s_size, s_a_size + 2 * s_s_size):
    index_data = i - s_s_size - s_a_size
    axs[i, 0].plot(s1[:, index_data].detach().numpy())
    axs[i, 0].set_title(f"s1[{index_data}]")
    axs[i, 0].grid()
    axs[i, 1].plot(s1_rec[:, index_data].detach().numpy())
    axs[i, 1].set_title(f"s1_rec[{index_data}]")
    axs[i, 1].grid()
fig.set_size_inches(18.5, 10.5)
plt.savefig(log_path / f"reconstruction.png")
plt.show(block=False)


# Plot the reconstruction error per dimension
fig, axs = plt.subplots(2 * s_s_size + s_a_size, 1)
err_s = s - s_rec
err_a = a - a_rec
err_s1 = s1 - s1_rec
fig.suptitle("Reconstruction Error per state/action dimension", fontsize=16)
for i in range(s_s_size):
    axs[i].hist(err_s[:, i].detach().numpy(), bins=1000)
    axs[i].set_title(f"s[{i}]-s_rec[{i}]")
    axs[i].grid()
for i in range(s_s_size, s_a_size + s_s_size):
    index_data = i - s_s_size
    axs[i].hist(err_a[:, index_data].detach().numpy(), bins=1000)
    axs[i].set_title((f"a[{index_data}]-a_rec[{index_data}]"))
    axs[i].grid()
for i in range(s_a_size + s_s_size, s_a_size + 2 * s_s_size):
    index_data = i - s_s_size - s_a_size
    axs[i].hist(err_s1[:, index_data].detach().numpy(), bins=1000)
    axs[i].set_title(f"s1[{index_data}]-s1_rec[{index_data}]")
    axs[i].grid()
fig.set_size_inches(18.5, 18.5)
plt.savefig(log_path / f"reconstruction_error.png")
plt.show(block=False)


# Plot the marginal distributions of the samples generated
fig, axs = plt.subplots(2 * t_s_size + t_a_size, 1)
fig.suptitle("Normalized Generated Samples Distribution", fontsize=16)
for i in range(t_s_size):
    axs[i].hist(s_t_code[:, i].detach().numpy(), bins=100)
    axs[i].set_title(f"s_t_code[{i}]")
    axs[i].grid()
for i in range(t_s_size, t_a_size + t_s_size):
    index_data = i - t_s_size
    axs[i].hist(a_t_code[:, index_data].detach().numpy(), bins=100)
    axs[i].set_title(f"a_t_code[{index_data}]")
    axs[i].grid()
for i in range(t_a_size + t_s_size, t_a_size + 2 * t_s_size):
    index_data = i - t_s_size - t_a_size
    axs[i].hist(s1_t_code[:, index_data].detach().numpy(), bins=100)
    axs[i].set_title(f"s1_t_code[{index_data}]")
    axs[i].grid()
fig.set_size_inches(18.5, 10.5)
plt.savefig(log_path / f"distributions.png")
plt.show(block=False)


# Create the test dataset for evaluating all models
super_ext_dataset_t = data_formatter_t_test.as_transitions()
sa, s1 = super_ext_dataset_t[:]

# Plot the error on the models
models = [T, T_ext, T_hyb, T_synth]
model_names = [
    "Trained on 500 target",
    "Trained on 5000 target",
    "Trained on 500 target + 5000 synthetic",
    "Trained on 5000 synthetic",
]
errors = []
fig = plt.figure(figsize=(18.5, 10.5))
for m in models:
    m = m.as_dict()["func"]
    with torch.no_grad():
        error = torch.norm(m(sa) - s1, dim=1).detach().numpy()
        error = [float(i) for i in error]
        errors.append(error)
        plt.plot(sorted(error, reverse=True))
plt.legend(model_names)
plt.title("Transition model trained using different datasets")
plt.grid()
plt.savefig(log_path / f"transitions.png")
plt.show(block=False)


# Plot error distribution per dimension
fig, axs = plt.subplots(t_s_size, 1)
fig.suptitle("Normalized Prediction Error per state dimension", fontsize=16)
for j, m in enumerate(models):
    m = m.as_dict()["func"]
    with torch.no_grad():
        error = m(sa) - s1
        for i in range(t_s_size):
            axs[i].hist(error[:, i].detach().numpy(), bins=1000, label=model_names[j])
            axs[i].set_title(f"s1[{i}]-s1_T[{i}]")
            axs[i].grid(True)
            axs[i].legend()
fig.set_size_inches(18.5, 14.5)
plt.savefig(log_path / f"error_per_dimension.png")
plt.show(block=False)


# Violin plots
fig, ax = plt.subplots()
plt.title("Error distribution of transition models trained on various datasets")
ax.violinplot(errors, showmedians=True, showextrema=False)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
bodies = ax.violinplot(errors, showmedians=True, showextrema=False)["bodies"]
for i, pc in enumerate(bodies):
    pc.set_facecolor(colors[i % len(colors)])
    pc.set_edgecolor("black")
    pc.set_alpha(0.7)
labels = [
    "500 target",
    "5000 target",
    "500 target \n+ 5000 synthetic",
    "5000 synthetic",
]
plt.xticks(list(range(1, len(errors) + 1)), labels)
plt.savefig(log_path / f"violins.png")
plt.show(block=False)


# Export errors
export = {'errors': errors, 'labels': labels}
with open(log_path / "errors.json", "w") as f:
    json.dump(export, f)
