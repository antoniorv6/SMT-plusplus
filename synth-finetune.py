import json
import torch
from loguru import logger
from fire import Fire
from data import SynthFinetuningDataset
from smt_trainer import SMTPP_Trainer
from ExperimentConfig import ExperimentConfig, experiment_config_from_dict

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision('high')

def main(config:ExperimentConfig, fold:int, weights_path:str=None):
    data = SynthFinetuningDataset(config=config, fold=fold)

    if weights_path is None:
        logger.warning("No weights path provided, starting from scratch")
        model_wrapper = SMTPP_Trainer(maxh=2512, maxw=2512, maxlen=5512, out_categories=len(data.train_dataset.w2i), 
                                  padding_token=data.train_dataset.w2i['<pad>'], in_channels=1, w2i=data.train_dataset.w2i, i2w=data.train_dataset.i2w, 
                                  d_model=256, dim_ff=256, num_dec_layers=8)
    else:
        logger.info(f"Loading weights from {weights_path}")
        model_wrapper = SMTPP_Trainer.load_from_checkpoint(weights_path)
    
    wandb_logger = WandbLogger(project='SMTPP', group="Polish_Scores", name="SMTPP_Mozarteum_Synthetic", log_model=False)
    
    checkpointer = ModelCheckpoint(dirpath="weights/finetuning/", filename="SMTPP_Mozarteum_Synthetic", save_on_train_epoch_end=True)

    trainer = Trainer(max_epochs=5, 
                      logger=wandb_logger, callbacks=[checkpointer], precision='16-mixed')
    
    data.train_dataset.set_trainer_data(trainer)

    trainer.fit(model_wrapper, datamodule=data)
    

def launch(config_path:str, fold:int, weights_path:str=None):
    with open(config_path, 'r') as file:
        config_dict = json.load(file)
        config = experiment_config_from_dict(config_dict)

    main(config=config, fold=fold, weights_path=weights_path)

if __name__ == "__main__":
    Fire(launch)