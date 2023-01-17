import torch

import torch.nn as nn
import torch.optim as optim

from mapping.models.model_interface import ModelInterface


class LitAutoEncoder(ModelInterface):
    def __init__(self, encoder, decoder, transition_model, discriminator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.D = discriminator
        self.T = transition_model

        self.s_size = self.T.s_dim
        self.a_size = self.T.a_dim

    def _get_loss(self, batch):
        """
        Given a batch of inputs, returns all the computed losses
        batch should be a tuple X,X (i.e. an identity)
        """
        x, _= batch
        
        x_hat = self.forward(x)
        x_enc = self.encoder(x)

        s_t, a_t, s1_t = torch.split(x_enc, [self.s_size, self.a_size, self.s_size], dim=1)
        valid = torch.zeros((batch.shape[0], 1)).fill_(1.0)

        # Compute all independent losses
        D_loss = nn.BCELoss()(self.D(x_enc), valid)

        sa = torch.cat([s_t,a_t],1)
        T_loss = nn.L1Loss()(self.T(sa), s1_t)

        AE_loss = nn.MSELoss()(x, x_hat)

        return AE_loss, T_loss, D_loss

    def compute_loss(self, batch, log_mode=None):
        AE_loss, T_loss, D_loss = self._get_loss(batch)
        loss = AE_loss + T_loss + D_loss
        if log_mode:
            self.log(f"{log_mode}_loss", loss)
            self.log(f"{log_mode}_AE_loss", AE_loss)
            self.log(f"{log_mode}_T_loss", T_loss)
            self.log(f"{log_mode}_D_loss", D_loss)
        return loss

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward(retain_graph=True)

    def forward(self, x):
        """The forward function takes an input and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, log_mode='train')

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, log_mode='val')

    def test_step(self, batch, batch_idx):
        return self.compute_loss(batch, log_mode='test')

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001)
        return optimizer
