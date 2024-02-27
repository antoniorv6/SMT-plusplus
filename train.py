import os
import gin
import fire
import torch

from config.config_utils import ExperimentConfig
from data import GraphicCLDataModule
from ModelManager import get_SMT_network, SMT
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

torch.set_float32_matmul_precision('high')

def main(config_vars, corpus, model_name, weights_path, fold):
    #logger.info("-----------------------")
    #logger.info(f"Training with the {model_name} model")
    #logger.info("-----------------------")

    data_path = f"{config_vars.data.data_path}{fold}"
    synth_path = config_vars.data.synth_path
    vocab_name = config_vars.data.vocab_name
    out_dir = config_vars.data.out_dir

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/hyp", exist_ok=True)
    os.makedirs(f"{out_dir}/gt", exist_ok=True)

    data_module = GraphicCLDataModule(data_path=data_path, synth_data_path=synth_path, vocab_name=vocab_name,
                                      num_cl_steps=config_vars.cl.cl_steps, max_synth_prob=config_vars.cl.cl_steps, increase_steps=config_vars.cl.stage_steps, finetune_samples=config_vars.cl.finetune_steps,
                                      batch_size=1, num_workers=24) 
    
    train_dataset = data_module.train_dataset

    model = SMT.load_from_checkpoint(weights_path)

    wandb_logger = WandbLogger(project='FP_SMT', group=f"{corpus}", name=f"{model_name}", log_model=False)

    early_stopping = EarlyStopping(monitor=config_vars.training.metric_to_watch, min_delta=0.01, patience=5, mode="min", verbose=True)
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/{corpus}/", filename=f"{model_name}_fold_{fold}", 
                                   monitor=config_vars.training.metric_to_watch, mode='min',
                                   save_top_k=1, verbose=True)
    
    checkpointer2 = ModelCheckpoint(dirpath=f"weights/{corpus}/", filename=f"{model_name}_synthetic", 
                                   every_n_train_steps=119000, verbose=True)

    trainer = Trainer(max_epochs=config_vars.training.max_epochs, 
                      check_val_every_n_epoch=config_vars.training.val_after, 
                      logger=wandb_logger, callbacks=[checkpointer, checkpointer2, early_stopping], precision='16-mixed')
    
    train_dataset.set_logger(wandb_logger)
    train_dataset.set_trainer_data(trainer)
    
    trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)

def launch(config, corpus, model_name, weights_path, fold):
    config_vars = ExperimentConfig(config)
    gin.parse_config_file(f"{config_vars.model.config_path}{model_name}.gin")
    main(config_vars, corpus, model_name, weights_path, fold)

if __name__ == "__main__":
    fire.Fire(launch)