import torch
from data import FinetuningDataset
from smt_trainer import SMTPP_Trainer

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

torch.set_float32_matmul_precision('high')

def main():
    data = FinetuningDataset(vocab_name='Mozarteum_BeKern', tokenization_mode='bekern')
    model_wrapper = SMTPP_Trainer.load_from_checkpoint('weights/pretraining/SMTPP_Mozarteum_BeKern_pretraining.ckpt')
    
    wandb_logger = WandbLogger(project='SMTPP', group=f"Mozarteum", name=f"SMTPP_Mozarteum_Bekern_f0", log_model=False)

    early_stopping = EarlyStopping(monitor="val_SER", min_delta=0.01, patience=5, mode="min", verbose=True)
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/finetuning/", filename=f"SMTPP_Mozarteum_Bekern_f0", 
                                   monitor="val_SER", mode='min',
                                   save_top_k=1, verbose=True)

    trainer = Trainer(max_epochs=100000, 
                      check_val_every_n_epoch=3500, 
                      logger=wandb_logger, callbacks=[checkpointer, early_stopping], precision='16-mixed')
    
    data.train_dataset.set_trainer_data(trainer)

    trainer.fit(model_wrapper, datamodule=data)
    
    model_wrapper = SMTPP_Trainer.load_from_checkpoint(checkpointer.best_model_path)
    
    model_wrapper.model.save_pretrained("SMTPP", variant="Mozarteum_BeKern_fold0")
    

if __name__ == "__main__":
    main()