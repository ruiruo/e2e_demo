from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from utils.config import get_train_config_obj
from utils.common import setup_callbacks
from dataset.dataloader import TrajectoryDataloaderModule
from model.trajectory_generator_train import TrajectoryTrainingModule
import os
import pickle

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

cfg_path = "./configs/training.yaml"
seed_everything(16)
config_obj = get_train_config_obj(config_path=cfg_path)
model = TrajectoryTrainingModule(config_obj)

print(model.gen_model)

data = TrajectoryDataloaderModule(cfg=config_obj)
# data.setup("train")
# print(len(data.train_loader), len(data.val_loader))

trainer = Trainer(
    callbacks=setup_callbacks(config_obj),
    logger=TensorBoardLogger(save_dir=config_obj.log_dir, default_hp_metric=False),
    accelerator="gpu",
    # strategy='ddp_find_unused_parameters_true',
    devices='auto',
    max_epochs=config_obj.epochs,
    log_every_n_steps=config_obj.log_every_n_steps,
    check_val_every_n_epoch=config_obj.check_val_every_n_epoch,
    profiler='simple'
)

seed_everything(175)
trainer.fit(model=model, datamodule=data, ckpt_path=config_obj.resume_path)
