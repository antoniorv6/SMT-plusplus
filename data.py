import re
import cv2
import torch
import numpy as np

from rich import progress
from utils import check_and_retrieveVocabulary
from lightning.pytorch import LightningDataModule
from data_augmentation.data_augmentation import augment, convert_img_to_tensor
from torch.utils.data import Dataset
from Generator.MusicSynthGen import VerovioGenerator

def load_set(path, base_folder="GrandStaff", fileformat="jpg", krn_type="bekrn", reduce_ratio=0.5):
    x = []
    y = []
    with open(path) as datafile:
        lines = datafile.readlines()
        for line in progress.track(lines):
            excerpt = line.replace("\n", "")
            try:
                with open(f"Data/{base_folder}/{'.'.join(excerpt.split('.')[:-1])}.{krn_type}") as krnfile:
                    krn_content = krnfile.read()
                    fname = ".".join(excerpt.split('.')[:-1])
                    img = cv2.imread(f"Data/{base_folder}/{fname}.{fileformat}")
                    width = int(np.ceil(img.shape[1] * reduce_ratio))
                    height = int(np.ceil(img.shape[0] * reduce_ratio))
                    img = cv2.resize(img, (width, height))
                    y.append([content + '\n' for content in krn_content.strip().split("\n")])
                    x.append(img)
            except Exception:
                print(f'Error reading Data/GrandStaff/{excerpt}')

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
    
class GrandStaffSingleSystem(OMRIMG2SEQDataset):
    def __init__(self, data_path, augment=False) -> None:
        self.augment = augment
        self.teacher_forcing_error_rate = 0.2
        self.x, self.y = load_set(data_path)
        self.y = self.preprocess_gt(self.y)
        self.num_sys_gen = 1
        self.fixed_systems_num = False
    
    def erase_numbers_in_tokens_with_equal(self, tokens):
        return [re.sub(r'(?<=\=)\d+', '', token) for token in tokens]

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.augment:
            x = augment(x)
        else:
            x = convert_img_to_tensor(x)
        
        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y]))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y
    
    def __len__(self):
        return len(self.x)
    
    def preprocess_gt(self, Y):
        for idx, krn in enumerate(Y):
            krnlines = []
            krn = "".join(krn)
            krn = krn.replace(" ", " <s> ")
            krn = krn.replace("·", "")
            krn = krn.replace("\t", " <t> ")
            krn = krn.replace("\n", " <b> ")
            krn = krn.split(" ")
                    
            Y[idx] = self.erase_numbers_in_tokens_with_equal(['<bos>'] + krn[4:-1] + ['<eos>'])
        return Y

class SyntheticOMRDataset(OMRIMG2SEQDataset):
    def __init__(self, data_path, number_of_systems=1, teacher_forcing_perc=0.2, reduce_ratio=0.5, 
                 dataset_length=40000, include_texture=False, augment=False, fixed_systems=False) -> None:
        super().__init__(teacher_forcing_perc, augment)
        self.generator = VerovioGenerator(gt_samples_path=data_path, fixed_number_systems=fixed_systems)
        self.num_sys_gen = number_of_systems
        self.dataset_len = dataset_length
        self.reduce_ratio = reduce_ratio
        self.include_texture = include_texture
    
    def __getitem__(self, index):
        
        x, y = self.generator.generate_score(num_sys_gen=self.num_sys_gen, reduce_ratio=self.reduce_ratio,
                                             check_generated_systems=True, add_texture=self.include_texture,
                                             include_title=False, include_author=False)
        
        if self.augment:
            x = augment(x)
        else:
            x = convert_img_to_tensor(x)

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y]))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y
    
    def __len__(self):
        return self.dataset_len

class PretrainingLinesDataset(LightningDataModule):
    def __init__(self, data_path, vocab_name, batch_size=1, num_workers=24) -> None:
        super().__init__()
        self.data_path = data_path
        self.vocab_name = vocab_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = SyntheticOMRDataset(data_path=f"{self.data_path}/train.txt", augment=True, fixed_systems=True)
        self.val_dataset = SyntheticOMRDataset(data_path=f"{self.data_path}/val.txt", dataset_length=1000, augment=False, fixed_systems=True)
        self.test_dataset = SyntheticOMRDataset(data_path=f"{self.data_path}/test.txt", dataset_length=1000, augment=False, fixed_systems=True)
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
    #train_dataset, val_dataset, test_dataset = load_data_single_pretraining("Data/GrandStaff/partitions_grandstaff/types/", "GrandStaffGlobal")
    #next(iter(train_dataset))
    #next(iter(val_dataset))
    data_module = PretrainingLinesDataset("Data/GrandStaff/partitions_grandstaff/types/", "GrandStaffGlobal")