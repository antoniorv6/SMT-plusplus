import torch
from data import FinetuningDataset
from smt_trainer import SMTPP_Trainer

from lightning.pytorch import Trainer

torch.set_float32_matmul_precision('high')

def main():
    data = FinetuningDataset(vocab_name='Mozarteum_BeKern', tokenization_mode='bekern')
    model_wrapper = SMTPP_Trainer.load_from_checkpoint('weights/finetuning/SMTPP_Mozarteum_Bekern_f0.ckpt')

    trainer = Trainer(max_epochs=100000, 
                      check_val_every_n_epoch=3500, precision='16-mixed')
    
    data.train_dataset.set_trainer_data(trainer)

    trainer.test(model_wrapper, datamodule=data)
    
    model_wrapper.model.save_pretrained("SMTPP", variant="Mozarteum_BeKern_fold0")
    

if __name__ == "__main__":
    main()