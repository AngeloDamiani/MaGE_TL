import torch
import torch.nn as nn
import torch.optim as optim

from mapping.models import ModelInterface

class LitAutoEncoder(ModelInterface):
    def __init__(self, D, T, encoder=None, decoder=None, dim_s=None, lambdas=None):
        super().__init__()

        D_dict = D.as_dict()
        T_dict = T.as_dict()
        self._D = D_dict['func']
        self._T = T_dict['func']

        self.s_size = T_dict['s_dim']
        self.a_size = T_dict['a_dim']

        if lambdas is None:
            self.lamb_AE, self.lamb_T, self.lamb_D = self._default_lambdas()
        else:
            self.lamb_AE, self.lamb_T, self.lamb_D = lambdas
        
        if encoder is None or decoder is None:
            assert dim_s is not None, "Parameter dim_s is required when omiting encoder or decoder"
        
        self.encoder = self._default_encoder(dim_s) if encoder is None else encoder
        self.decoder = self._default_decoder(dim_s) if decoder is None else decoder

    def _default_lambdas(self):
        return (1,1,1)

    def _default_network(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Linear(
                64, output_size)
        )
    
    def _default_decoder(self, dim_s):
        return self._default_network(self.s_size*2+self.a_size, dim_s)
        
    def _default_encoder(self, dim_s):
        return self._default_network(dim_s, self.s_size*2+self.a_size)

    def _get_loss(self, batch):
        """
        Given a batch of inputs, returns all the computed losses
        batch should be a tuple X,X (i.e. an identity)
        """
        x, _= batch
        
        x_hat = self.forward(x)
        
        x_enc = self.encoder(x)

        s_t, a_t, s1_t = torch.split(x_enc, [self.s_size, self.a_size, self.s_size], dim=1)
        valid = torch.zeros((x.shape[0], 1)).fill_(1.0)

        # Compute all independent losses
        sa = torch.cat([s_t,a_t],1)

        with torch.no_grad():
            T_loss = nn.L1Loss()(self._T(sa), s1_t)
            D_loss = nn.BCELoss()(self._D(x_enc), valid)

        AE_loss = nn.MSELoss()(x, x_hat)

        return AE_loss, T_loss, D_loss

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
        optimizer = optim.AdamW(self.parameters(), lr=0.05)
        return optimizer

    def as_dict(self):
        ae_dict = {}
        ae_dict['func'] = lambda sas: self(sas)
        ae_dict['M'] = lambda sas: self.encoder(sas)
        ae_dict['inv_M'] = lambda sas: self.decoder(sas)
        return ae_dict
        


