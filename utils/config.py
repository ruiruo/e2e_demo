import datetime
import os
from dataclasses import dataclass
import torch
import yaml
import numpy as np
from loguru import logger


@dataclass
class Configuration:
    # Dataset Options
    experiment_name: str
    data_dir: str
    training_dir: str
    validation_dir: str
    token_nums: int
    tokenizer: str
    detokenizer: str
    multi_agent_info: bool
    max_frame: int
    max_agent: int
    simple_deduction: bool
    tfrecord_niofs_path: str
    local_data_save_dir: str
    save_tf_to_local_tmp_path: str
    save_data_num_per: int
    case_size: int
    bucket: str
    access_key: str
    secret_key: str
    max_lane_num: int
    max_node_num_per_lane: int
    vector_graph_feature_fea_dim: list
    max_train: int
    max_val: int
    sampling_strategy: str
    sampling_temperature: float
    config_top_k: int
    config_top_p: float
    sample_rate: int

    # Basic Options
    with_pad: bool
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
    customized_metric: bool
    ar_start_epoch: int
    ar_warmup_epochs: int
    max_grad_norm: float
    # Model Options
    dropout: float
    # Encoder Options
    embedding_dim: int
    num_topy_layers: int
    # Decoder Options
    item_number: int
    tf_de_dim: int
    tf_de_heads: int
    tf_de_layers: int

    # Tokenizer
    x_boundaries: np.ndarray
    y_boundaries: np.ndarray
    bos_token: int
    eos_token: int
    pad_token: int
    # Optional extras
    device: torch.device = torch.device('cpu')
    resume_path: str = None
    config_path: str = None
    log_dir: str = None
    checkpoint_dir: str = None
    ignore_eos_loss: bool = True


@dataclass
class InferenceConfiguration:
    model_ckpt_path: str
    training_config: str

    train_meta_config: Configuration = None


def get_train_config_obj(config_path: str):
    exp_name = get_exp_name()
    with open(config_path, 'r') as yaml_file:
        try:
            config_yaml = yaml.safe_load(yaml_file)
            if "tfrecord_files_list" in config_yaml:
                config_yaml.pop("tfrecord_files_list")
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
