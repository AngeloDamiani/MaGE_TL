from mapping.datasets import RLDatasetFormatter
from mapping.models import TransitionModel, Discriminator
import torch


def generate_synthetic_dataset(n, s_bound, a_bound):

    fake_s, fake_s1, fake_a = [], [], []

    for r0, r1 in s_bound:
        fake_s.append((r1-r0)*torch.rand((n, 1)) + r0)
    fake_s = torch.cat(fake_s, 1)

    for r0, r1 in a_bound:
        fake_a.append((r1-r0)*torch.rand((n, 1)) + r0)
    fake_a = torch.cat(fake_a, 1)

    for r0, r1 in s_bound:
        fake_s1.append((r1-r0)*torch.rand((n, 1)) + r0)
    fake_s1 = torch.cat(fake_s1, 1)
    fake_r = torch.zeros((n, 1))

    return RLDatasetFormatter([fake_s, fake_a, fake_r, fake_s1]).transition_as_fake()


data_formatter = RLDatasetFormatter().from_csv('data/UntrainedMCDataset500.csv')
disc_dataset = data_formatter.transition_as_valid()

fake_samples = [
    {
        'count': len(disc_dataset)//2,
        's_bound': [(-1.2, 0.6), (-0.07, 0.07)],
        'a_bound': [(-1, 1)]
    },
    {
        'count': len(disc_dataset)//2,
        's_bound': [(-12, 6), (-0.7, 0.7)],
        'a_bound': [(-10, 10)]
    }]

for desc in fake_samples:
    sd = generate_synthetic_dataset(desc['count'], desc['s_bound'], desc['a_bound'])
    disc_dataset = disc_dataset.merge(sd)


D = Discriminator(data_formatter.state_size, data_formatter.action_size)
D.train_model(disc_dataset, logs_dir='/Discriminator', batch_size=20)
