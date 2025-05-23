import os
import json
import pickle
import datetime
import logging
import torch
import ray
from dynaconf import Dynaconf
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import ModelConfigDict
from ray.tune.registry import register_env
from tqdm import tqdm
from utils import env_creator, check_dir
from utils.config import get_inference_config_obj
from model.trajectory_generator_ppo_model import TrajectoryGenerator

# Initialize Ray
ray.init(
    num_cpus=8,
    num_gpus=1,
    include_dashboard=False,
    object_store_memory=10 * 1000 ** 3,
    _system_config={"maximum_gcs_destroyed_actor_cached_count": 300},
)

# Paths and experiment setup
env_name = "TrajectoryGenerator-v0"
run_name = f"PPO_{datetime.datetime.now():%Y%m%d}"

# Load hyperparameters
setting = Dynaconf(envvar_prefix="DYNACONF", settings_files=["configs/ppo.yaml"])
hyper_parameters = setting.hyper_parameters.to_dict()

# Directory bases from settings
log_base = setting.log.get("log_base")
ckpt_base = setting.log.get("ckpt_base")

# Ensure directories exist log
check_dir(log_base)
log_base = os.path.join(log_base, run_name)
check_dir(log_base)

# Ensure directories exist checkpoint
check_dir(ckpt_base)
ckpt_base = os.path.join(ckpt_base, run_name)
check_dir(ckpt_base)

# Add environment config
hyper_parameters["env_config"] = {
    "id": env_name,
    "test_img": "/home/nio/test_img/",
    "task_paths": "/home/nio/data/test/"
}

# Initialize and register the environment
register_env(env_name, env_creator)
env = env_creator(hyper_parameters["env_config"])
env.reset()
obs, reward, terminated, truncated, info = env.step([1.5, 1.4])
print(env.action_space, env.observation_space)
env.close()
print(f"Environment '{env_name}' registered.")

# Register Custom Model
custom_model_name_registered = "trajectory_generator_v0"
ModelCatalog.register_custom_model(custom_model_name_registered, TrajectoryGenerator)
print(f"Custom model '{custom_model_name_registered}' registered.")


# PPO Configuration
config = PPOConfig().environment(env_name)
config = config.framework("torch")
config = config.rl_module(_enable_rl_module_api=False)
config = config.training(_enable_learner_api=False)
config.rollouts(rollout_fragment_length="auto")
trainer = config.build()