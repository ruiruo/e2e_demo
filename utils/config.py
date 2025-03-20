import datetime
import os
from dataclasses import dataclass
from typing import List
import torch
import yaml
import numpy as np
from loguru import logger


@dataclass
class Configuration:
    # Dataset Options
    data_dir: str
    training_dir: str
    validation_dir: str
    token_nums: int
    tokenizer: str
    detokenizer: str
    multi_agent_info: bool
    max_frame: int
    max_agent: int
    # Basic Options
    data_mode: str
    num_gpus: int
    cuda_device_index: str
    log_root_dir: str
    checkpoint_root_dir: str
    log_every_n_steps: int
    check_val_every_n_epoch: int
    epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    num_workers: int

    # Target Options
    add_noise_to_target: bool
    target_noise_threshold: float

    # Encoder Options
    embedding_dim: int
    # Decoder Options
    item_number: int
    tf_de_dim: int
    tf_de_heads: int
    tf_de_layers: int
    tf_de_dropout: float

    # Tokenizer
    x_boundaries: List | np.ndarray
    y_boundaries: List | np.ndarray
    bos_token: int
    eos_token: int
    pad_token: int
    # Optional extras
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resume_path: str = None
    config_path: str = None
    log_dir: str = None
    checkpoint_dir: str = None
    is_train: bool = True


@dataclass
class InferenceConfiguration:
    model_ckpt_path: str
    training_config: str
    predict_mode: str

    trajectory_pub_frequency: int
    cam_info_dir: str
    progress_threshold: float

    train_meta_config: Configuration = None


def get_train_config_obj(config_path: str):
    exp_name = get_exp_name()
    with open(config_path, 'r') as yaml_file:
        try:
            config_yaml = yaml.safe_load(yaml_file)
            config_obj = Configuration(**config_yaml)
            config_obj.config_path = config_path
            config_obj.log_dir = os.path.join(config_obj.log_root_dir, exp_name)
            config_obj.x_boundaries = np.array(config_obj.x_boundaries)
            config_obj.y_boundaries = np.array(config_obj.y_boundaries)
            config_obj.checkpoint_dir = os.path.join(config_obj.checkpoint_root_dir, exp_name)
        except yaml.YAMLError:
            logger.exception("Open {} failed!", config_path)
    return config_obj


def get_exp_name():
    today = datetime.datetime.now()
    today_str = "{}_{}_{}_{}_{}_{}".format(today.year, today.month, today.day,
                                           today.hour, today.minute, today.second)
    exp_name = "exp_{}".format(today_str)
    return exp_name


def get_inference_config_obj(config_path: str):
    with open(config_path, 'r') as yaml_file:
        try:
            config_yaml = yaml.safe_load(yaml_file)
            inference_config_obj = InferenceConfiguration(**config_yaml)
        except yaml.YAMLError:
            logger.exception("Open {} failed!", config_path)
    training_config_path = os.path.join(os.path.dirname(config_path),
                                        "{}.yaml".format(inference_config_obj.training_config))
    inference_config_obj.train_meta_config = get_train_config_obj(training_config_path)
    return inference_config_obj
