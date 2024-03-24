import os
import torch
import hydra
from SMT import SMT
from config_typings import Config

from data import PretrainingLinesDataset
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

torch.set_float32_matmul_precision('high')

@hydra.main(version_base=None, config_path="config")
def main(config:Config):

    data_module = PretrainingLinesDataset(config=config.data)
    
    train_dataset = data_module.train_dataset
    w2i, i2w = train_dataset.get_dictionaries()

    model = SMT(config=config.model_setup, w2i=w2i, i2w=i2w)

    wandb_logger = WandbLogger(project='FP_SMT', group=f"{config.metadata.corpus_name}", name=f"{config.metadata.model_name}", log_model=False)

    early_stopping = EarlyStopping(monitor=config.experiment.metric_to_watch, min_delta=0.01, patience=5, mode="min", verbose=True)
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/{config.metadata.corpus_name}/", filename=f"{config.metadata.model_name}_pretraining", 
                                   monitor=config.experiment.metric_to_watch, mode='min',
                                   save_top_k=1, verbose=True)

    trainer = Trainer(max_epochs=config.experiment.max_epochs, 
                      check_val_every_n_epoch=config.experiment.val_after, 
                      logger=wandb_logger, callbacks=[checkpointer, early_stopping], precision='16-mixed')

    trainer.fit(model, datamodule=data_module)

    model = SMT.load_from_checkpoint(checkpointer.best_model_path)

    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()