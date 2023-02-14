import torch
import matplotlib.pyplot as plt

from mapping.datasets import RLDatasetFormatter
from mapping.models import TransitionModel, LitAutoEncoder
from mapping.problem_factory import problem_factory
import json


SOURCE = 'Pend' 
TARGET = 'MC'

# WARNING, PUT THE 'H' IN DESCENDING ORDER
ae_ckpt_path = [
    #"logs/G4_500_5/AE/lightning_logs/version_19/checkpoints/epoch=67-step=1292.ckpt",
    "logs/G4_500_2/AE/lightning_logs/version_19/checkpoints/epoch=67-step=1292.ckpt",
    "logs/ANG_200_3/AE/lightning_logs/version_19/checkpoints/epoch=67-step=1292.ckpt",
    "logs/NEW_100_13/AE/lightning_logs/version_19/checkpoints/epoch=67-step=1292.ckpt"
]
T_LOG_DIR = "logs/T_graph1"
H = [500, 200, 100]

source_data, target_data = problem_factory(source_name=SOURCE,target_name=TARGET)
data_formatter_s, data_formatter_s_val = source_data['dfs']
data_formatter_t, data_formatter_t_val, data_formatter_t_test = target_data['dfs']
s_s_max, s_s_min, s_a_max, s_a_min = source_data['max_mins']
t_s_max, t_s_min, t_a_max, t_a_min = target_data['max_mins']
s_s_size, s_a_size = source_data['sizes']
t_s_size, t_a_size = target_data['sizes']
fig = plt.figure(figsize=(10.5, 7.5))

error_data = {}

for i in range(len(ae_ckpt_path)):
    L = H[i]
    ############### DATASET LOADING ############################################

    data_formatter_t.s = data_formatter_t.s[:L]
    data_formatter_t.a = data_formatter_t.a[:L]
    data_formatter_t.r = data_formatter_t.r[:L]
    data_formatter_t.s1 = data_formatter_t.s1[:L]

    AE = LitAutoEncoder.load_from_checkpoint(ae_ckpt_path[i], D=None, T=None, lambdas=(0,0,0), 
        lr=0,
        s_s_size=s_s_size,
        s_a_size=s_a_size,
        t_s_size=t_s_size,
        t_a_size=t_a_size
    )
    AE_dict = AE.as_dict()
    M = AE_dict["M"]

    # SYNTHESIZE DATA

    dataset_s_2 = data_formatter_s_val.transition_identity()
    dataset_s_2.shuffle()
    sas, _ = dataset_s_2[:]
    with torch.no_grad():
        encoded_data = M(sas)

    s_t_code, a_t_code, s1_t_code = torch.split(
        encoded_data, [t_s_size, t_a_size, t_s_size], 1
    )
    r_code = torch.zeros(len(s1_t_code), 1)
    synth_data_formatter = RLDatasetFormatter([s_t_code, a_t_code, r_code, s1_t_code])
    synth_dataset_t = synth_data_formatter.as_transitions()

    # TRAIN T

    errors = []
    samples = [i for i in range(0, len(dataset_s_2), 100)]


    # Create the test dataset for evaluating models
    super_ext_dataset_t = data_formatter_t_test.as_transitions()
    sa, s1 = super_ext_dataset_t[:]
    for n_samples in samples:
        dataset_t = data_formatter_t.as_transitions()
        if n_samples != 0:
            train_data_formatter = RLDatasetFormatter()
            train_data_formatter.s = synth_data_formatter.s[:n_samples]
            train_data_formatter.a = synth_data_formatter.a[:n_samples]
            train_data_formatter.r = synth_data_formatter.r[:n_samples]
            train_data_formatter.s1 = synth_data_formatter.s1[:n_samples]
            train_dataset = train_data_formatter.as_transitions()
            train_dataset = dataset_t.merge(train_dataset)
            train_dataset.shuffle()
        else:
            train_dataset = dataset_t

        T_hat = TransitionModel(s_dim=t_s_size, a_dim=t_a_size, lr=0.0001)
        T_hat.train_model(
            train_dataset,
            logs_dir=T_LOG_DIR,
            epochs=5,
            batch_size=100,
        )

        T_hat = T_hat.as_dict()["func"]
        with torch.no_grad():
            #error = torch.norm(T_hat(sa) - s1, dim=1).detach().numpy()
            error = float(torch.mean(torch.norm(T_hat(sa) - s1, dim=1)))

            errors.append(error)

    
    #plt.boxplot(errors, showfliers=False)
    p = plt.plot(samples, errors, linestyle=':', label='L = '+str(L))
    plt.scatter(samples[0], errors[0], marker='x')
    plt.scatter(samples[1:], errors[1:], marker='o', c=p[0].get_color())
    plt.title("Transition Model Average Prediction Error")
    plt.xlabel("Synthetic Samples\n + 'L' Target Samples\n used to train the model")
    plt.ylabel('Average Prediction Error')
    error_data[L] = errors
    
    
plt.legend()
plt.grid()
plt.show()


with open("errors_L.json", "w") as f:
    json.dump(error_data, f)




