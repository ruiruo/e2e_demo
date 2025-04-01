from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import Trainer, seed_everything
from utils.config import get_train_config_obj
from utils.common import setup_callbacks
from utils.logger_utils import PrintLogger
from dataset.dataloader import TrajectoryDataloaderModule
from model.trajectory_generator_train import TrajectoryTrainingModule

cfg_path = "./configs/training.yaml"
seed_everything(15)
config_obj = get_train_config_obj(config_path=cfg_path)
model = TrajectoryTrainingModule(config_obj)

print(model.gen_model)

data = TrajectoryDataloaderModule(cfg=config_obj)
# data.setup("train")
# print(len(data.train_loader), len(data.val_loader))

mlflow_logger = MLFlowLogger(
    experiment_name="e2e_planner",
    tracking_uri="http://172.21.191.16:9999"
)

print_logger = PrintLogger()

trainer = Trainer(
    callbacks=setup_callbacks(config_obj),
    logger=[mlflow_logger, print_logger],
    accelerator="gpu",
    # strategy='ddp_find_unused_parameters_true',
    devices=config_obj.num_gpus,
    max_epochs=config_obj.epochs,
    log_every_n_steps=config_obj.log_every_n_steps,
    check_val_every_n_epoch=config_obj.check_val_every_n_epoch,
    profiler='simple'
)

trainer.fit(model=model, datamodule=data, ckpt_path=config_obj.resume_path)
