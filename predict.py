from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import Trainer, seed_everything
from utils.config import get_train_config_obj, get_inference_config_obj
from utils.common import setup_callbacks
from utils.logger_utils import PrintLogger
from dataset.dataloader import DataLoader, TrajectoryDataModule
from model.trajectory_generator_predict import TrajectoryPredictModule
import pytorch_lightning as pl

seed_everything(15)
pred_config_obj = get_inference_config_obj("./configs/predict.yaml")
train_config_obj = pred_config_obj.train_meta_config
train_config_obj.log_every_n_steps = 2
train_config_obj.max_train = 10
train_config_obj.max_val = 5
train_config_obj.data_dir = "/home/nio/data/"
train_config_obj.log_dir = train_config_obj.log_dir.replace("shaoqian.li", "nio")
train_config_obj.checkpoint_dir = train_config_obj.checkpoint_dir.replace("shaoqian.li", "nio")
train_config_obj.checkpoint_root_dir = "/home/nio/checkpoints/"
train_config_obj.local_data_save_dir = "/home/nio/"
train_config_obj.tokenizer = "/home/nio/reparke2e/configs/local2token.npy"
train_config_obj.detokenizer = "/home/nio/reparke2e/configs/token2local.json"
train_config_obj.batch_size = 4
model = TrajectoryPredictModule(infer_cfg=pred_config_obj,
                                train_cfg=train_config_obj,
                                device="gpu")

data = DataLoader(dataset=TrajectoryDataModule(config=train_config_obj, is_train=1),
                  batch_size=train_config_obj.batch_size,
                  shuffle=True,
                  num_workers=train_config_obj.num_workers,
                  pin_memory=True,
                  drop_last=True)

print(len(data))

trainer = Trainer(
    accelerator="gpu",
    # strategy='ddp_find_unused_parameters_true',
    devices=1,
    profiler='simple'
)

predictor = TrajectoryPredictModule.load_from_checkpoint(pred_config_obj, cfg=pred_config_obj)

# Run prediction using the Lightning Trainer
predictions = trainer.predict(predictor, dataloaders=data)

# mlflow_logger = MLFlowLogger(
#     experiment_name="e2e_planner",
#     tracking_uri="http://172.21.191.16:9999"
# )
#
# print_logger = PrintLogger()
#
# trainer = Trainer(
#     callbacks=setup_callbacks(config_obj),
#     logger=[mlflow_logger, print_logger],
#     accelerator="gpu",
#     # strategy='ddp_find_unused_parameters_true',
#     devices=config_obj.num_gpus,
#     max_epochs=config_obj.epochs,
#     log_every_n_steps=config_obj.log_every_n_steps,
#     check_val_every_n_epoch=config_obj.check_val_every_n_epoch,
#     profiler='simple'
# )
#
# trainer.fit(model=model, datamodule=data, ckpt_path=config_obj.resume_path)
