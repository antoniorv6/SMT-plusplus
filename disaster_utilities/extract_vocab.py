from smt_trainer import SMTPP_Trainer

model = SMTPP_Trainer.load_from_checkpoint('weights/finetuning/SMTPP_Mozarteum_Bekern_f0.ckpt').model
print(model.i2w)
