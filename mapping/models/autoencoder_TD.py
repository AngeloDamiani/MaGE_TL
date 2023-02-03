import torch
import torch.nn as nn
import torch.optim as optim

from mapping.models import ModelInterface

class LitAutoEncoder(ModelInterface):
    def __init__(self, lr=0.0001, **kwargs):
        super(LitAutoEncoder, self).__init__(lr)

        self.s_s_size = kwargs['s_s_size']
        self.s_a_size = kwargs['s_a_size']
        self.t_s_size = kwargs['t_s_size']
        self.t_a_size = kwargs['t_a_size']

        self.D = kwargs['D']
        self.T = kwargs['T']
        
        if 'lambdas' in kwargs:
            self.lamb_AE, self.lamb_T, self.lamb_D = kwargs["lambdas"]
        else:
            self.lamb_AE, self.lamb_T, self.lamb_D = self._default_lambdas()
        

        self.sfc_s = nn.Sequential(
            nn.Linear(self.s_s_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.actionfc_s = nn.Sequential(
            nn.Linear(self.s_a_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.s1fc_s = nn.Sequential(
            nn.Linear(self.s_s_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.model_enc = nn.Sequential(
            nn.Linear(128 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 2*self.t_s_size+self.t_a_size)
        )

        self.sfc_t = nn.Sequential(
            nn.Linear(self.t_s_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.actionfc_t = nn.Sequential(
            nn.Linear(self.t_a_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.s1fc_t = nn.Sequential(
            nn.Linear(self.t_s_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.model_dec = nn.Sequential(
            nn.Linear(128 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 2*self.s_s_size+self.s_a_size)
        )
    def _default_lambdas(self):
        return (1,1,1)

    def encoder(self, x):
        s_s, a_s, s1_s = torch.split(x, [self.s_s_size, self.s_a_size, self.s_s_size], dim=1)
        s_feature = self.sfc_s(s_s)
        a_feature = self.actionfc_s(a_s)
        s1_feature = self.s1fc_s(s1_s)
        feature = torch.cat((s_feature, a_feature, s1_feature),1)
        return self.model_enc(feature)

    def decoder(self, x):
        s_t, a_t, s1_t = torch.split(x, [self.t_s_size, self.t_a_size, self.t_s_size], dim=1)
        s_feature = self.sfc_t(s_t)
        a_feature = self.actionfc_t(a_t)
        s1_feature = self.s1fc_t(s1_t)
        feature = torch.cat((s_feature, a_feature, s1_feature),1)
        return self.model_dec(feature)

    def _get_loss(self, batch):
        """
        Given a batch of inputs, returns all the computed losses
        batch should be a tuple X,X (i.e. an identity)
        """
        x, _= batch
        
        x_hat = self.forward(x)
        
        x_enc = self.encoder(x)

        s_t, a_t, s1_t = torch.split(x_enc, [self.t_s_size, self.t_a_size, self.t_s_size], dim=1)
        valid = torch.zeros((x.shape[0], 1)).fill_(1.0)

        # Compute all independent losses
        sa = torch.cat([s_t,a_t],1)
        T_loss = nn.L1Loss()(self.T(sa), s1_t)
        D_loss = nn.BCELoss()(self.D(x_enc), valid)
        AE_loss = nn.MSELoss()(x, x_hat)

        return AE_loss, D_loss, T_loss

    def compute_loss(self, batch, log_mode=None):
        AE_loss, T_loss, D_loss = self._get_loss(batch)
        loss = self.lamb_AE*AE_loss + self.lamb_T*T_loss + self.lamb_D*D_loss
        
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
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def as_dict(self):
        ae_dict = {}
        ae_dict['func'] = lambda sas: self(sas)
        ae_dict['M'] = lambda sas: self.encoder(sas)
        ae_dict['inv_M'] = lambda sas: self.decoder(sas)
        return ae_dict
        


