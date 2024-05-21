import re
import cv2
import torch
import numpy as np

from config_typings import DataConfig, CLConfig
from rich import progress
from utils import check_and_retrieveVocabulary
from lightning.pytorch import LightningDataModule
from data_augmentation.data_augmentation import augment, convert_img_to_tensor
from torch.utils.data import Dataset
from Generator.MusicSynthGen import VerovioGenerator

def erase_whitespace_elements(tokens):
    return [token for token in tokens if token != ""]

def load_set(path, base_folder="GrandStaff", fileformat="jpg", krn_type="bekrn", reduce_ratio=0.5):
    x = []
    y = []
    with open(path) as datafile:
        lines = datafile.readlines()
        for line in progress.track(lines):
            excerpt = line.replace("\n", "")
            #try:
            with open(f"Data/{base_folder}/{'.'.join(excerpt.split('.')[:-1])}.{krn_type}") as krnfile:
                krn_content = krnfile.read()
                fname = ".".join(excerpt.split('.')[:-1])
                img = cv2.imread(f"Data/{base_folder}/{fname}.{fileformat}")
                width = int(np.ceil(img.shape[1] * reduce_ratio))
                height = int(np.ceil(img.shape[0] * reduce_ratio))
                img = cv2.resize(img, (width, height))
                y.append([content + '\n' for content in krn_content.strip().split("\n")])
                x.append(img)
            #except Exception e:
            #    print(f'Error reading Data/GrandStaff/{excerpt}')

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
    
    def preprocess_gt(self, Y, tokenization_method="standard"):
        for idx, krn in enumerate(Y):
            krnlines = []
            krn = "".join(krn)
            krn = krn.replace(" ", " <s> ")
            
            if tokenization_method == "bekern":
                krn = krn.replace("·", " ")
                krn = krn.replace("@", " ")
            if tokenization_method == "ekern":
                krn = krn.replace("·", " ")
                krn = krn.replace("@", "")
            if tokenization_method == "standard":
                krn = krn.replace("·", "")
                krn = krn.replace("@", "")
                
            krn = krn.replace("/", "")
            krn = krn.replace("\\", "")
            krn = krn.replace("\t", " <t> ")
            krn = krn.replace("\n", " <b> ")
            krn = krn.split(" ")
                    
            Y[idx] = self.erase_numbers_in_tokens_with_equal(['<bos>'] + krn[4:-1] + ['<eos>'])
        return Y

class GrandStaffSingleSystem(OMRIMG2SEQDataset):
    def __init__(self, data_path, config: DataConfig, augment=False) -> None:
        super().__init__(augment)
        self.augment = augment
        self.teacher_forcing_error_rate = 0.2
        self.x, self.y = load_set(data_path, base_folder=config.base_folder, 
                                  fileformat=config.file_format, krn_type=config.krn_type, reduce_ratio=config.reduce_ratio)
        self.y = self.preprocess_gt(self.y, tokenization_method=config.tokenization_mode)
        self.num_sys_gen = 1
        self.fixed_systems_num = False

    @staticmethod
    def erase_numbers_in_tokens_with_equal(tokens):
        return [re.sub(r'(?<=\=)\d+', '', token) for token in tokens]

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        
        if self.augment:
            x = augment(x)
        else:
            x = convert_img_to_tensor(x)
        
        y = torch.from_numpy(np.asarray([self.w2i[token] for token in erase_whitespace_elements(y)]))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y
    
    def __len__(self):
        return len(self.x)

class SyntheticOMRDataset(OMRIMG2SEQDataset):
    def __init__(self, data_path, number_of_systems=1, teacher_forcing_perc=0.2, reduce_ratio=0.5, 
                 dataset_length=40000, include_texture=False, augment=False, fixed_systems=False, tokenization_mode="standard") -> None:
        super().__init__(teacher_forcing_perc, augment)
        self.generator = VerovioGenerator(gt_samples_path=data_path, fixed_number_systems=fixed_systems, tokenization_method=tokenization_mode)
        self.num_sys_gen = number_of_systems
        self.dataset_len = dataset_length
        self.reduce_ratio = reduce_ratio
        self.include_texture = include_texture
        self.tokenization_mode = tokenization_mode
    
    def __getitem__(self, index):
        
        x, y = self.generator.generate_system()

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
    def __init__(self, config:DataConfig, batch_size=1, num_workers=24) -> None:
        super().__init__()
        self.data_path = config.data_path
        self.vocab_name = config.vocab_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = SyntheticOMRDataset(data_path=f"{self.data_path}/train.txt", augment=True, fixed_systems=True, tokenization_mode=config.tokenization_mode)
        self.val_dataset = SyntheticOMRDataset(data_path=f"{self.data_path}/val.txt", dataset_length=1000, augment=False, fixed_systems=True, tokenization_mode=config.tokenization_mode)
        self.test_dataset = SyntheticOMRDataset(data_path=f"{self.data_path}/test.txt", dataset_length=1000, augment=False, fixed_systems=True, tokenization_mode=config.tokenization_mode)
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
    
    
class CLOMRDataset(OMRIMG2SEQDataset):
    def __init__(self, data_config:DataConfig, cl_config:CLConfig, teacher_forcing_perc=0.2, augment=False) -> None:
        super().__init__(teacher_forcing_perc, augment)
        self.x, self.y = load_set(f"{data_config.data_path}{data_config.fold}/train.txt", base_folder=data_config.base_folder, fileformat=data_config.file_format, 
                                  krn_type=data_config.krn_type, reduce_ratio=data_config.reduce_ratio)
        self.generator = VerovioGenerator(gt_samples_path=f"{data_config.synth_path}/train.txt", fixed_number_systems=False, tokenization_method=data_config.tokenization_mode)
        self.y = self.preprocess_gt(self.y, tokenization_method=data_config.tokenization_mode)
        self.num_sys_gen = 1

        # CL parameters
        self.max_synth_prob = cl_config.max_synth_prob
        self.min_synth_prob = cl_config.min_synth_prob
        self.perc_synth_samples = cl_config.max_synth_prob
        self.num_steps_decrease = cl_config.finetune_steps
        self.increase_steps = cl_config.increase_steps
        self.num_cl_steps = cl_config.num_cl_steps
        self.max_cl_steps = self.increase_steps * self.num_cl_steps
        self.curriculum_stage_beginning = cl_config.curriculum_stage_beginning
        self.tokenization_mode = data_config.tokenization_mode
        
        self.skip_progressive = cl_config.skip_progressive
        self.offset = 0
        if self.skip_progressive:
            self.offset = (self.num_cl_steps + self.curriculum_stage_beginning) * self.increase_steps 
        self.skip_cl = cl_config.skip_cl
    
    def set_logger(self, logger):
        self.logger = logger
    
    def erase_numbers_in_tokens_with_equal(self, tokens):
       return [re.sub(r'(?<=\=)\d+', '', token) for token in tokens]

    def linear_scheduler_synthetic(self):
        return self.max_synth_prob + round(((self.trainer.global_step + self.offset) - self.max_cl_steps) * (self.min_synth_prob - self.max_synth_prob) / self.num_steps_decrease, 4)

    def set_trainer_data(self, trainer):
        self.trainer = trainer
    
    def __getitem__(self, index):
        stage = (self.trainer.global_step // self.increase_steps) + self.curriculum_stage_beginning
        
        if (stage < self.num_cl_steps + self.curriculum_stage_beginning) and not self.skip_progressive:
            probability = 1
            num_sys_to_gen = np.random.randint(1, stage)
            #Set the variable gen_author_title to True if a random number if above 0.5
            #add_texture = np.random.rand() > 0.3
            gen_author_title = np.random.rand() > 0.5
            cut_height = np.random.rand() > 0.7
            x, y = self.generator.generate_score(num_sys_gen=num_sys_to_gen,
                                                 check_generated_systems=True, cut_height=cut_height, add_texture=True, 
                                                 include_author=gen_author_title, include_title=gen_author_title)
        else:
            if self.skip_cl:
                probability = 0
            else:
                probability = max(self.min_synth_prob, self.linear_scheduler_synthetic())
            if np.random.random() > probability:
                x = self.x[index]
                y = erase_whitespace_elements(self.y[index])
            else:
                num_sys_to_gen = np.random.randint(1, 4)
                gen_texture = np.random.rand() > 0.3
                gen_author_title = np.random.rand() > 0.5
                cut_height = np.random.rand() > 0.7
                x, y = self.generator.generate_score(num_sys_gen=num_sys_to_gen,
                                                     check_generated_systems=False, cut_height=cut_height, add_texture=gen_texture, 
                                                     include_author=gen_author_title, include_title=gen_author_title)

        if self.augment:
            x = augment(x)
        else:
            x = convert_img_to_tensor(x)
        
        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y]))
        decoder_input = self.apply_teacher_forcing(y)

        if self.logger != None:
            self.logger.experiment.log({'synth_proba': probability, 'training_stage': stage})
    
        return x, decoder_input, y


class GraphicCLDataModule(LightningDataModule):
    def __init__(self, data_config:DataConfig, cl_config:CLConfig, fold=0, batch_size=1, num_workers=24) -> None:
        super().__init__()
        self.data_path = f"{data_config.data_path}_{fold}"
        self.synth_data_path = data_config.synth_path
        self.vocab_name = data_config.vocab_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = CLOMRDataset(data_config=data_config, cl_config=cl_config, augment=True)
        
        self.val_dataset = GrandStaffSingleSystem(data_path=f"{data_config.data_path}{fold}/val.txt",config=data_config, augment=False)
        self.test_dataset = GrandStaffSingleSystem(data_path=f"{data_config.data_path}{fold}/test.txt", config=data_config, augment=False)
        w2i, i2w = check_and_retrieveVocabulary([self.train_dataset.get_gt(), 
                                                 self.val_dataset.get_gt(), 
                                                 self.test_dataset.get_gt()], "vocab/", f"{self.vocab_name}")#
    
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