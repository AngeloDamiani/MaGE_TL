import torch

import torch.nn as nn
import torch.optim as optim

from mapping.models import ModelInterface

class TransitionModel(ModelInterface):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.loss_criterion = nn.L1Loss()
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.statefc = nn.Sequential(
            nn.Linear(s_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.actionfc = nn.Sequential(
            nn.Linear(a_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.predfc = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, s_dim))

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        # Include extra logging here
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("val_loss", loss)

        sa, s1 = batch
        s1_hat = self.forward(sa)
        self.log("val_accuracy",nn.MSELoss()(s1, s1_hat))
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("test_loss", loss)
        return loss

    def _get_loss(self, batch):
        sa, s1 = batch
        s1_hat = self.forward(sa)
        return self.loss_criterion(s1_hat, s1)

    def forward(self, X):
        X = torch.atleast_2d(X)
        s, a = X.split([self.s_dim,self.a_dim],1)
        state_feature = self.statefc(s.float())
        action_feature = self.actionfc(a.float())
        feature = torch.cat((state_feature, action_feature), 1)
        return self.predfc(feature)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward(retain_graph=True)
    
    def as_dict(self):
        T_dict = {}
        T_dict['func'] = lambda sa: self(sa)
        T_dict['s_dim'] = self.s_dim
        T_dict['a_dim'] = self.a_dim
        return T_dict
