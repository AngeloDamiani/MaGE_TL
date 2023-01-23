import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from mapping.config import LOGGING_DIR

ACC = "gpu" if torch.cuda.is_available() else None


class ModelInterface(pl.LightningModule):

    def training_step(self, batch, batch_idx):
        ...

    def validation_step(self, batch, batch_idx):
        ...

    def test_step(self, batch, batch_idx):
        ...

    def forward(self, X):
        ...

    def configure_optimizers(self):
        ...

    def backward(self, loss, optimizer, optimizer_idx):
        ...

    def train_model(self, dataset, batch_size=20, epochs=100, splits=[0.9, 0.1], model_dir='/int_model'):
        datasets = random_split(dataset, splits)
        train_loader, validation_loader = (
            DataLoader(i, batch_size=batch_size) for i in datasets
        )

        # Configure the training
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=LOGGING_DIR+model_dir)
        trainer = pl.Trainer(
            logger=tb_logger, max_epochs=epochs, accelerator=ACC)
        trainer.logger._log_graph = True  # Plot the computation graph in tensorboard
        # Train the autoencoder
        trainer.fit(
            model=self, train_dataloaders=train_loader, val_dataloaders=validation_loader
        )
    
    def as_dict(self):
        m_dict = {}
        m_dict['func'] = lambda X: self(X)
        return m_dict

from mapping.models.autoencoder import LitAutoEncoder
from mapping.models.discriminator import Discriminator
from mapping.models.transition_model import TransitionModel

__all__ = [
    'LitAutoEncoder',
    'Discriminator',
    'TransitionModel'
]