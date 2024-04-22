import os

import hydra
from config_typings import Config

import torch

from SMT import SMT

from data import GraphicCLDataModule
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

torch.set_float32_matmul_precision('high')

@hydra.main(version_base=None, config_path="config")
def main(config:Config):
    data_module = GraphicCLDataModule(config.data, config.cl, fold=config.data.fold)

    model = SMT.load_from_checkpoint(config.experiment.pretrain_weights, config=config.model_setup)
    
    trainer = Trainer(max_epochs=config.experiment.max_epochs, 
                      check_val_every_n_epoch=config.experiment.val_after, precision='16-mixed')

    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()