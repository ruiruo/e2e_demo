from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import Trainer, seed_everything
from utils.config import get_train_config_obj
from utils.common import setup_callbacks
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

run_name = (
    f"Emb={config_obj.embedding_dim}, Dim={config_obj.tf_de_dim}, "
    f"Head={config_obj.tf_de_heads}, Layer={config_obj.tf_de_layers}"
)

mlflow_logger = MLFlowLogger(
    experiment_name="e2e_planner",
    run_name=run_name,
    tracking_uri="http://172.21.191.16:9999"
)

trainer = Trainer(
    callbacks=setup_callbacks(config_obj, "val_accuracy", "max"),
    logger=[mlflow_logger],
    accelerator="gpu",
    # strategy='ddp_find_unused_parameters_true',
    devices=config_obj.num_gpus,
    max_epochs=config_obj.epochs,
    log_every_n_steps=config_obj.log_every_n_steps,
    check_val_every_n_epoch=config_obj.check_val_every_n_epoch,
    profiler='simple'
)

trainer.fit(model=model, datamodule=data, ckpt_path=config_obj.resume_path)
