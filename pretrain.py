import json
import torch
from fire import Fire
from data import PretrainingDataset
from smt_trainer import SMTPP_Trainer
from ExperimentConfig import ExperimentConfig, experiment_config_from_dict

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

torch.set_float32_matmul_precision('high')

def main(config:ExperimentConfig):

    data = PretrainingDataset(config=config)
    model_wrapper = SMTPP_Trainer(maxh=2512, maxw=2512, maxlen=5512, out_categories=len(data.train_dataset.w2i), 
                                  padding_token=data.train_dataset.w2i['<pad>'], in_channels=1, w2i=data.train_dataset.w2i, i2w=data.train_dataset.i2w, 
                                  d_model=256, dim_ff=256, num_dec_layers=8)
    
    wandb_logger = WandbLogger(project='SMTPP', group=f"Polish_Scores", name=f"SMTPP_Pretraining_Polish_Scores_Bekern", log_model=False)

    early_stopping = EarlyStopping(monitor="val_SER", min_delta=0.01, patience=5, mode="min", verbose=True)
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/pretraining/", filename=f"SMTPP_Polish_Scores_BeKern_pretraining", 
                                   monitor="val_SER", mode='min',
                                   save_top_k=1, verbose=True)

    trainer = Trainer(max_epochs=10000, 
                      check_val_every_n_epoch=5, 
                      logger=wandb_logger, callbacks=[checkpointer, early_stopping], precision='16-mixed')

    trainer.fit(model_wrapper, datamodule=data)
    
    model_wrapper = SMTPP_Trainer.load_from_checkpoint(checkpointer.best_model_path)
    
    model_wrapper.model.save_pretrained("SMTPP", variant="Mozarteum_BeKern_pretrain")
    

def launch(config_path:str):
    with open(config_path, 'r') as file:
        config_dict = json.load(file)
        config = experiment_config_from_dict(config_dict)

    main(config=config)

if __name__ == "__main__":
    Fire(launch)