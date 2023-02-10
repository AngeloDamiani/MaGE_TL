from mapping.datasets import RLDatasetFormatter
from mapping.models import TransitionModel, Discriminator

data_formatter = RLDatasetFormatter().from_csv('data/UntrainedMCDataset5000.csv')
T_dataset = data_formatter.transition_as_valid()
sas, label = T_dataset[:]

T = Discriminator(data_formatter.state_size, data_formatter.action_size)
T.train_model(T_dataset, logs_dir='/D')


