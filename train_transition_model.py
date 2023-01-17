from mapping.datasets import RLDatasetFormatter
from mapping.models import TransitionModel, Discriminator

data_formatter = RLDatasetFormatter().from_csv('data/UntrainedMCDataset500.csv')
dataset_t = data_formatter.as_transitions()

T = TransitionModel(data_formatter.state_size, data_formatter.action_size)
T.train_model(dataset_t, model_dir='/T')


