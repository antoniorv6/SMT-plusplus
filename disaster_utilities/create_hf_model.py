from smt_trainer import SMTPP_Trainer
from fire import Fire
import os

def main(weights_path, save_path):
    model = SMTPP_Trainer.load_from_checkpoint(weights_path).model
    model.push_to_hub(save_path, commit_message="Weights uploaded")

if __name__ == "__main__":
    Fire(main)