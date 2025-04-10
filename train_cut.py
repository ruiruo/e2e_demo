import sys
import io
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint, ModelSummary

# -------------------------------
# Monkey-patch sys.stdout to ignore BrokenPipeError
# -------------------------------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, errors='ignore')

# -------------------------------
# Import your project modules
# -------------------------------
from utils.config import get_train_config_obj
from utils.common import setup_callbacks  # This function creates ModelCheckpoint, LRMonitor, etc.
from dataset.dataloader import TrajectoryDataloaderModule
from model.trajectory_generator_train import TrajectoryTrainingModule

# -------------------------------
# Set random seed and load config
# -------------------------------
cfg_path = "./configs/training_cut.yaml"
seed_everything(15)
config_obj = get_train_config_obj(config_path=cfg_path)

# -------------------------------
# Initialize model and dataloader
# -------------------------------
model = TrajectoryTrainingModule(config_obj)
print("Model summary:", model.gen_model)

data = TrajectoryDataloaderModule(cfg=config_obj)

# -------------------------------
# Setup MLFlow logger
# -------------------------------
run_name = (
    f"Emb={config_obj.embedding_dim}, Dim={config_obj.tf_de_dim}, "
    f"Head={config_obj.tf_de_heads}, Layer={config_obj.tf_de_layers}"
)
mlflow_logger = MLFlowLogger(
    experiment_name="e2e_planner",
    run_name=run_name,
    tracking_uri="http://172.21.191.16:9999"
)

# -------------------------------
# Setup callbacks
# -------------------------------
# Create callbacks from your utility function
base_callbacks = setup_callbacks(config_obj, monitor="val_accuracy", mode="max")

# Create a TQDMProgressBar that outputs to sys.stderr instead of sys.stdout
progress_bar = TQDMProgressBar(file=sys.stderr)

# Add the custom progress bar callback to your list of callbacks
callbacks = base_callbacks + [progress_bar]

# -------------------------------
# Create the Trainer
# -------------------------------
trainer = Trainer(
    callbacks=callbacks,
    logger=[mlflow_logger],
    accelerator="gpu",
    devices=config_obj.num_gpus,
    max_epochs=config_obj.epochs,
    log_every_n_steps=config_obj.log_every_n_steps,
    check_val_every_n_epoch=config_obj.check_val_every_n_epoch,
    profiler='simple'
)

# -------------------------------
# Start the training process
# -------------------------------
trainer.fit(model=model, datamodule=data, ckpt_path=config_obj.resume_path)

# -------------------------------
# (Optional) Additional utility functions and decorators, if needed below.
# -------------------------------
import json
from utils.decorator_train import finish, init

def get_json_content(json_file_path: str):
    try:
        with open(json_file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error reading JSON file '{json_file_path}': {e}!")

def decorator_function(train_function):
    def wrapper_function(*args, **kwargs):
        init(*args, **kwargs)
        train_function(*args, **kwargs)
        finish(*args, **kwargs)
    return wrapper_function
