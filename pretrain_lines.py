import os
import gin
import fire
import torch

from config.config_utils import ExperimentConfig
from data import PretrainingLinesDataset
from torch.utils.data import DataLoader
from ModelManager import get_SMT_network, SMT
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

torch.set_float32_matmul_precision('high')

def main(config_vars, corpus, model_name):
    #logger.info("-----------------------")
    #logger.info(f"Training with the {model_name} model")
    #logger.info("-----------------------")

    data_path = config_vars.data.data_path
    vocab_name = config_vars.data.vocab_name
    out_dir = config_vars.data.out_dir

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/hyp", exist_ok=True)
    os.makedirs(f"{out_dir}/gt", exist_ok=True)

    data_module = PretrainingLinesDataset(data_path=data_path, vocab_name=vocab_name)
    
    train_dataset = data_module.train_dataset
    w2i, i2w = train_dataset.get_dictionaries()

    model = get_SMT_network(in_channels=1,
                            max_height=2376, max_width=1680, 
                            max_len=4512, 
                            out_categories=len(train_dataset.get_i2w()),
                            w2i=w2i, i2w=i2w, 
                            model_name="SMTCNN", out_dir=out_dir)


    wandb_logger = WandbLogger(project='FP_SMT', group=f"{corpus}", name=f"{model_name}", log_model=False)

    early_stopping = EarlyStopping(monitor=config_vars.training.metric_to_watch, min_delta=0.01, patience=5, mode="min", verbose=True)
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/{corpus}/", filename=f"{model_name}_pretraining", 
                                   monitor=config_vars.training.metric_to_watch, mode='min',
                                   save_top_k=1, verbose=True)

    trainer = Trainer(max_epochs=config_vars.training.max_epochs, 
                      check_val_every_n_epoch=config_vars.training.val_after, 
                      logger=wandb_logger, callbacks=[checkpointer, early_stopping])

    trainer.fit(model, datamodule=data_module)

    model = SMT.load_from_checkpoint(checkpointer.best_model_path)

    trainer.test(model, datamodule=data_module)

def launch(config, corpus, model_name):
    config_vars = ExperimentConfig(config)
    gin.parse_config_file(f"{config_vars.model.config_path}{model_name}.gin")
    main(config_vars, corpus, model_name)

if __name__ == "__main__":
    fire.Fire(launch)