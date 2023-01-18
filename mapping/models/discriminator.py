import torch
import torch.nn as nn
import torch.optim as optim

from mapping.models import ModelInterface

class Discriminator(ModelInterface):
    def __init__(self, s_dim, a_dim):
        super(Discriminator, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.sfc = nn.Sequential(
            nn.Linear(s_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.actionfc = nn.Sequential(
            nn.Linear(a_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.s1fc = nn.Sequential(
            nn.Linear(s_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.model = nn.Sequential(
            nn.Linear(128 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.loss_criterion = nn.BCELoss()

    def forward(self, x):
        x = torch.atleast_2d(x)
        s, a, s1 = torch.split(x, [self.s_dim, self.a_dim, self.s_dim], dim=1)
        s_feature = self.sfc(s)
        a_feature = self.actionfc(a)
        s1_feature = self.s1fc(s1)
        feature = torch.cat((s_feature, a_feature, s1_feature),1)
        return self.model(feature)

    def _get_loss_batch(self, batch):
        sas, label = batch
        res = self.forward(sas)
        return self.loss_criterion(res, label)

    def training_step(self, batch, batch_idx):
        loss = self._get_loss_batch(batch)
        # Include extra logging here
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss_batch(batch)
        self.log("val_loss", loss)

        sas, label = batch
        res = self.forward(sas)
        self.log("val_accuracy", torch.mean(torch.abs(res-label)))

        return loss

    def test_step(self, batch, batch_idx):
        loss = self._get_loss_batch(batch)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001)
        return optimizer

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward(retain_graph=True)
