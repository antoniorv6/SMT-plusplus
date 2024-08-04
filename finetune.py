import json
import torch
from loguru import logger
from fire import Fire
from data import FinetuningDataset
from smt_trainer import SMTPP_Trainer
from ExperimentConfig import ExperimentConfig, experiment_config_from_dict

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

torch.set_float32_matmul_precision('high')

def main(config:ExperimentConfig, fold:int, weights_path:str=None):
    data = FinetuningDataset(config=config, fold=fold)

    if weights_path is None:
        logger.warning("No weights path provided, starting from scratch")
        model_wrapper = SMTPP_Trainer(maxh=2512, maxw=2512, maxlen=5512, out_categories=len(data.train_dataset.w2i), 
                                  padding_token=data.train_dataset.w2i['<pad>'], in_channels=1, w2i=data.train_dataset.w2i, i2w=data.train_dataset.i2w, 
                                  d_model=256, dim_ff=256, num_dec_layers=8)
    else:
        logger.info(f"Loading weights from {weights_path}")
        model_wrapper = SMTPP_Trainer.load_from_checkpoint(weights_path)
    
    wandb_logger = WandbLogger(project='SMTPP', group=f"Polish_Scores", name=f"SMTPP_Polish_Scores_Bekern_f{fold}", log_model=False)

    early_stopping = EarlyStopping(monitor="val_SER", min_delta=0.01, patience=5, mode="min", verbose=True)
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/finetuning/", filename=f"SMTPP_Polish_Scores_Bekern_f{fold}", 
                                   monitor="val_SER", mode='min',
                                   save_top_k=1, verbose=True)

    trainer = Trainer(max_epochs=100000, 
                      check_val_every_n_epoch=3500, 
                      logger=wandb_logger, callbacks=[checkpointer, early_stopping], precision='16-mixed')
    
    data.train_dataset.set_trainer_data(trainer)

    trainer.fit(model_wrapper, datamodule=data)
    
    model_wrapper = SMTPP_Trainer.load_from_checkpoint(checkpointer.best_model_path)
    
    model_wrapper.model.save_pretrained("SMTPP", variant="Mozarteum_BeKern_fold0")
    

def launch(config_path:str, fold:int, weights_path:str=None):
    with open(config_path, 'r') as file:
        config_dict = json.load(file)
        config = experiment_config_from_dict(config_dict)

    main(config=config, fold=fold, weights_path=weights_path)

if __name__ == "__main__":
    Fire(launch)