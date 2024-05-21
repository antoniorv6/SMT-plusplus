from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class ExperimentConfig:
    metric_to_watch: str
    metric_mode: str
    max_epochs: int
    val_after: int
    pretrain_weights : str = ""

@dataclass
class DataConfig:
    data_path: str
    file_format: str
    vocab_name:str
    krn_type: str 
    reduce_ratio: float
    fold: int
    synth_path: str = ""
    base_folder: str = ""
    tokenization_mode: str = "standard"

@dataclass
class CLConfig:
    num_cl_steps: int
    max_synth_prob: float
    min_synth_prob: float
    increase_steps: int
    finetune_steps: int
    teacher_forcing_perc: float
    curriculum_stage_beginning: int
    skip_progressive: bool = False
    skip_cl: bool = False

@dataclass
class SMTConfig:
    in_channels: int
    d_model: int
    dim_ff: int
    num_dec_layers: int
    encoder_type: int
    max_height: int
    max_width: int
    max_len: int
    lr: float

@dataclass
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    model_setup: SMTConfig
    cl: CLConfig


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="experiment", node=ExperimentConfig)
cs.store(name="data", node=DataConfig)
cs.store(name="model_setup", node=SMTConfig)
cs.store(name="cl", node=CLConfig)