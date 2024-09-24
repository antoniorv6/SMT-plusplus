import re
import cv2
import wandb
import torch
import random
import numpy as np
from rich import progress
from ExperimentConfig import ExperimentConfig
from Generator.SynthGenerator import VerovioGenerator
from data_augmentation.data_augmentation import augment, convert_img_to_tensor
from utils.vocab_utils import check_and_retrieveVocabulary

from datasets import load_dataset
from torch.utils.data import Dataset
from lightning.pytorch import LightningDataModule

def clean_kern(krn, avoid_tokens=['*tremolo','*staff2', '*staff1','*Xped', '*tremolo', '*ped', '*Xtuplet', '*tuplet', "*Xtremolo", '*cue', '*Xcue', '*rscale:1/2', '*rscale:1', '*kcancel', '*below']):
    krn = krn.split('\n')
    newkrn = []
    # Remove the lines that contain the avoid tokens
    for idx, line in enumerate(krn):
        if not any([token in line.split('\t') for token in avoid_tokens]):
            #If all the tokens of the line are not '*'
            if not all([token == '*' for token in line.split('\t')]):
                newkrn.append(line.replace("\n", ""))
                
    return "\n".join(newkrn)

def parse_kern_file(krn: str, tokenization_mode='bekern') -> str:
    krn = clean_kern(krn)
    krn = krn.replace(" ", " <s> ")
    krn = krn.replace("\t", " <t> ")
    krn = krn.replace("\n", " <b> ")
    krn = krn.replace(" /", "")
    krn = krn.replace(" \\", "")
    krn = krn.replace("·/", "")
    krn = krn.replace("·\\", "")
    
    if tokenization_mode == "kern":
        krn = krn.replace("·", "").replace('@', '')
    
    if tokenization_mode == "ekern":
        krn = krn.replace("·", " ").replace('@', '')
    
    if tokenization_mode == "bekern":
        krn = krn.replace("·", " ").replace("@", " ")
        
    krn = krn.split(" ")[4:-1]
    krn = [re.sub(r'(?<=\=)\d+', '', token) for token in krn]
    
    return krn

def load_from_files_list(file_ref:str, split:str="train", tokenization_mode='bekern', reduce_ratio=0.5) -> list:
    dataset = load_dataset(file_ref, split=split)
    x = []
    y = []
    for sample in dataset:
        y.append(['<bos>'] + parse_kern_file(sample["transcription"], tokenization_mode=tokenization_mode) + ['<eos>'])
        img = img = np.array(sample['image'])
        width = int(np.ceil(img.shape[1] * reduce_ratio))
        height = int(np.ceil(img.shape[0] * reduce_ratio))
        img = cv2.resize(img, (width, height))
        x.append(img)
    return x, y

def batch_preparation_img2seq(data):
    images = [sample[0] for sample in data]
    dec_in = [sample[1] for sample in data]
    gt = [sample[2] for sample in data]

    max_image_width = max([img.shape[2] for img in images])
    max_image_height = max([img.shape[1] for img in images])

    X_train = torch.ones(size=[len(images), 1, max_image_height, max_image_width], dtype=torch.float32)

    for i, img in enumerate(images):
        _, h, w = img.size()
        X_train[i, :, :h, :w] = img
    
    max_length_seq = max([len(w) for w in gt])

    decoder_input = torch.zeros(size=[len(dec_in),max_length_seq])
    y = torch.zeros(size=[len(gt),max_length_seq])

    for i, seq in enumerate(dec_in):
        decoder_input[i, 0:len(seq)-1] = torch.from_numpy(np.asarray([char for char in seq[:-1]]))
    
    for i, seq in enumerate(gt):
        y[i, 0:len(seq)-1] = torch.from_numpy(np.asarray([char for char in seq[1:]]))
    
    return X_train, decoder_input.long(), y.long()

class OMRIMG2SEQDataset(Dataset):
    def __init__(self, teacher_forcing_perc=0.2, augment=False) -> None:
        self.x = None
        self.y = None
        self.teacher_forcing_error_rate = teacher_forcing_perc
        self.augment = augment

        super().__init__()
    
    def apply_teacher_forcing(self, sequence):
        errored_sequence = sequence.clone()
        for token in range(1, len(sequence)):
            if np.random.rand() < self.teacher_forcing_error_rate and sequence[token] != self.padding_token:
                errored_sequence[token] = np.random.randint(0, len(self.w2i))
        
        return errored_sequence

    def __len__(self):
        return len(self.x)

    def get_max_hw(self):
        m_width = np.max([img.shape[1] for img in self.x])
        m_height = np.max([img.shape[0] for img in self.x])

        return m_height, m_width
    
    def get_max_seqlen(self):
        return np.max([len(seq) for seq in self.y])

    def vocab_size(self):
        return len(self.w2i)

    def get_gt(self):
        return self.y
    
    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i['<pad>']
    
    def get_dictionaries(self):
        return self.w2i, self.i2w
    
    def get_i2w(self):
        return self.i2w

class SyntheticOMRDataset(OMRIMG2SEQDataset):
    def __init__(self, data_path, split="train", number_of_systems=1, teacher_forcing_perc=0.2, reduce_ratio=0.5, 
                 dataset_length=40000, augment=False, tokenization_mode="standard") -> None:
        super().__init__(teacher_forcing_perc, augment)
        self.generator = VerovioGenerator(sources=data_path, split=split, tokenization_mode=tokenization_mode)
        
        self.num_sys_gen = number_of_systems
        self.dataset_len = dataset_length
        self.reduce_ratio = reduce_ratio
        self.tokenization_mode = tokenization_mode
    
    def __getitem__(self, index):
        
        x, y = self.generator.generate_music_system_image()

        if self.augment:
            x = augment(x)
        else:
            x = convert_img_to_tensor(x)

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y]))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y
    
    def __len__(self):
        return self.dataset_len

class RealDataset(OMRIMG2SEQDataset):
    def __init__(self, data_path, split, teacher_forcing_perc=0.2, reduce_ratio=1.0, 
                augment=False, tokenization_mode="standard") -> None:
       super().__init__(teacher_forcing_perc, augment)
       self.reduce_ratio = reduce_ratio
       self.tokenization_mode = tokenization_mode
       self.x, self.y = load_from_files_list(data_path, split, tokenization_mode, reduce_ratio=reduce_ratio)
       
    def __getitem__(self, index):
       
       x = self.x[index]
       y = self.y[index]

       if self.augment:
           x = augment(x)
       else:
           x = convert_img_to_tensor(x)

       y = torch.from_numpy(np.asarray([self.w2i[token] for token in y if token != '']))
       decoder_input = self.apply_teacher_forcing(y)
       return x, decoder_input, y

    def __len__(self):
       return len(self.x)

class CurriculumTrainingDataset(OMRIMG2SEQDataset):
    def __init__(self, data_path, split, teacher_forcing_perc=0.2, reduce_ratio=1.0, 
                augment=False, tokenization_mode="standard") -> None:
       super().__init__(teacher_forcing_perc, augment)
       self.reduce_ratio = reduce_ratio
       self.tokenization_mode = tokenization_mode
       self.x, self.y = load_from_files_list(data_path, split, tokenization_mode, reduce_ratio=reduce_ratio)
       self.generator = VerovioGenerator(sources="antoniorv6/grandstaff-ekern", 
                                         split="train",
                                         tokenization_mode=tokenization_mode)
       
       self.max_synth_prob = 0.9
       self.min_synth_prob = 0.2
       self.finetune_steps = 200000
       self.increase_steps = 40000
       self.num_cl_steps = 3
       self.max_cl_steps = self.increase_steps * self.num_cl_steps
       self.curriculum_stage_beginning = 2
    
    def set_trainer_data(self, trainer):
        self.trainer = trainer
    
    def linear_scheduler_synthetic(self, step):
        return self.max_synth_prob + round((step - self.max_cl_steps) * (self.min_synth_prob - self.max_synth_prob) / self.finetune_steps, 4)

    def __getitem__(self, index):
        step = self.trainer.global_step
        stage = (self.trainer.global_step // self.increase_steps) + self.curriculum_stage_beginning
        gen_author_title = np.random.rand() > 0.5
        wandb.log({'Stage': stage})
        if stage < (self.num_cl_steps + self.curriculum_stage_beginning):
           num_sys_to_gen = random.randint(1, stage)
           x, y = self.generator.generate_full_page_score(
               max_systems = num_sys_to_gen,
               strict_systems=True,
               strict_height=(random.random() < 0.3),
               include_author=gen_author_title,
               include_title=gen_author_title,
               reduce_ratio=0.5)
        else:
            probability = max(self.linear_scheduler_synthetic(step), self.min_synth_prob)
            wandb.log({'Synthetic Probability': probability})
            if random.random() > probability:
                x = self.x[index]
                y = self.y[index]
            else:
                x, y = self.generator.generate_full_page_score(
                    max_systems = random.randint(3, 4),
                    strict_systems=False,
                    strict_height=(random.random() < 0.3),
                    include_author=gen_author_title,
                    include_title=gen_author_title,
                    reduce_ratio=0.5)

        if self.augment:
           x = augment(x)
        else:
           x = convert_img_to_tensor(x)

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y if token != '']))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y

    def __len__(self):
       return len(self.x)


class PretrainingDataset(LightningDataModule):
    def __init__(self, config:ExperimentConfig) -> None:
        super().__init__()
        self.data_path = config.data.data_path
        self.vocab_name = config.data.vocab_name
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.tokenization_mode = config.data.tokenization_mode

        self.train_dataset = SyntheticOMRDataset(data_path=self.data_path, split="train", augment=True, tokenization_mode=self.tokenization_mode)
        self.val_dataset = SyntheticOMRDataset(data_path=self.data_path, split="val", dataset_length=1000, augment=False, tokenization_mode=self.tokenization_mode)
        self.test_dataset = SyntheticOMRDataset(data_path=self.data_path, split="test", dataset_length=1000, augment=False, tokenization_mode=self.tokenization_mode)
        w2i, i2w = check_and_retrieveVocabulary([self.train_dataset.get_gt(), self.val_dataset.get_gt(), self.test_dataset.get_gt()], "vocab/", f"{self.vocab_name}")#
    
        self.train_dataset.set_dictionaries(w2i, i2w)
        self.val_dataset.set_dictionaries(w2i, i2w)
        self.test_dataset.set_dictionaries(w2i, i2w)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=batch_preparation_img2seq)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)


class FinetuningDataset(LightningDataModule):
    def __init__(self, config:ExperimentConfig, fold=0) -> None:
        super().__init__()
        self.data_path = config.data.data_path
        self.vocab_name = config.data.vocab_name
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.tokenization_mode = config.data.tokenization_mode
        self.train_dataset = CurriculumTrainingDataset(data_path=self.data_path, split="train", augment=True, tokenization_mode=self.tokenization_mode, reduce_ratio=config.data.reduce_ratio)
        self.val_dataset = RealDataset(data_path=self.data_path, split="val", augment=False, tokenization_mode=self.tokenization_mode, reduce_ratio=config.data.reduce_ratio)
        self.test_dataset = RealDataset(data_path=self.data_path, split="test", augment=False, tokenization_mode=self.tokenization_mode, reduce_ratio=config.data.reduce_ratio)
        w2i, i2w = check_and_retrieveVocabulary([self.train_dataset.get_gt(), self.val_dataset.get_gt(), self.test_dataset.get_gt()], "vocab/", f"{self.vocab_name}")#
    
        self.train_dataset.set_dictionaries(w2i, i2w)
        self.val_dataset.set_dictionaries(w2i, i2w)
        self.test_dataset.set_dictionaries(w2i, i2w)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=batch_preparation_img2seq)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)

class SyntheticCLDataset(OMRIMG2SEQDataset):
    def __init__(self, data_path, base_folder, teacher_forcing_perc=0.2, reduce_ratio=1.0, 
                augment=False, tokenization_mode="standard") -> None:
       super().__init__(teacher_forcing_perc, augment)
       self.reduce_ratio = reduce_ratio
       self.tokenization_mode = tokenization_mode
       self.generator = VerovioGenerator(sources=["Data/GrandStaff/partitions_grandstaff/types/train.txt"], 
                                         base_folder="Data/GrandStaff/",
                                         tokenization_mode=tokenization_mode)
       
       self.max_synth_prob = 0.9
       self.min_synth_prob = 0.2
       self.finetune_steps = 200000
       self.increase_steps = 40000
       self.num_cl_steps = 3
       self.max_cl_steps = self.increase_steps * self.num_cl_steps
       self.curriculum_stage_beginning = 2
    
    def set_trainer_data(self, trainer):
        self.trainer = trainer

    def __getitem__(self, index):
        stage = (self.trainer.global_step // self.increase_steps) + self.curriculum_stage_beginning
        gen_author_title = np.random.rand() > 0.5
        wandb.log({'Stage': stage})
        if stage < (self.num_cl_steps + self.curriculum_stage_beginning):
           num_sys_to_gen = random.randint(1, stage)
           x, y = self.generator.generate_full_page_score(
               max_systems = num_sys_to_gen,
               strict_systems=True,
               strict_height=(random.random() < 0.3),
               include_author=gen_author_title,
               include_title=gen_author_title,
               texturize_image=(random.random() > 0.5),
               reduce_ratio=0.5)
        else:
            x, y = self.generator.generate_full_page_score(
                max_systems = random.randint(3, 4),
                strict_systems=False,
                strict_height=(random.random() < 0.3),
                include_author=gen_author_title,
                include_title=gen_author_title,
                texturize_image=(random.random() > 0.5),
                reduce_ratio=0.5)

        if self.augment:
           x = augment(x)
        else:
           x = convert_img_to_tensor(x)

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y if token != '']))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y

    def __len__(self):
       return 80000

class SynthFinetuningDataset(LightningDataModule):
    def __init__(self, config:ExperimentConfig, fold=0) -> None:
        super().__init__()
        self.data_path = config.data.data_path + f"fold_{fold}/"
        self.base_folder = config.data.base_folder
        self.vocab_name = config.data.vocab_name
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.tokenization_mode = config.data.tokenization_mode
        self.train_dataset = SyntheticCLDataset(data_path=f"{self.data_path}/train.txt", base_folder=self.base_folder, augment=True, tokenization_mode=self.tokenization_mode, reduce_ratio=config.data.reduce_ratio)
        self.val_dataset = RealDataset(data_path=f"{self.data_path}/val.txt", base_folder=self.base_folder, augment=False, tokenization_mode=self.tokenization_mode, reduce_ratio=config.data.reduce_ratio)
        self.test_dataset = RealDataset(data_path=f"{self.data_path}/test.txt", base_folder=self.base_folder, augment=False, tokenization_mode=self.tokenization_mode, reduce_ratio=config.data.reduce_ratio)
        w2i, i2w = check_and_retrieveVocabulary([self.train_dataset.get_gt(), self.val_dataset.get_gt(), self.test_dataset.get_gt()], "vocab/", f"{self.vocab_name}")#
    
        self.train_dataset.set_dictionaries(w2i, i2w)
        self.val_dataset.set_dictionaries(w2i, i2w)
        self.test_dataset.set_dictionaries(w2i, i2w)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=batch_preparation_img2seq)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)

if __name__ == "__main__":
    dataset = RealDataset(
        data_path='Data/Polish_Scores/partitions_polish_scores/excerpts/fold_0/train.txt',
        base_folder='Data/Polish_Scores/',
        reduce_ratio=0.5,
        tokenization_mode='bekern'
    )

    w2i, i2w = check_and_retrieveVocabulary([], "vocab/", "Polish_Scores_BeKern")

    dataset.set_dictionaries(w2i, i2w)

    print(dataset.__getitem__(0))