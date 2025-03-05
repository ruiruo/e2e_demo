from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from dataset.dataloader import ParkingDataloaderModule
from model.parking_model_train import ParkingTrainingModuleTrain
from utils.decorator_train import finish, init
from utils.config import get_train_config_obj


def decorator_function(train_function):
    def wrapper_function(*args, **kwargs):
        init(*args, **kwargs)
        train_function(*args, **kwargs)
        finish(*args, **kwargs)

    return wrapper_function


def setup_callbacks(cfg_obj):
    ckpt_callback = ModelCheckpoint(dirpath=cfg_obj.checkpoint_dir,
                                    monitor='val_loss',
                                    save_top_k=3,
                                    mode='min',
                                    filename='{epoch:02d}-{val_loss:.2f}',
                                    save_last=True)
    progress_bar = TQDMProgressBar()
    model_summary = ModelSummary(max_depth=2)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    return [ckpt_callback, progress_bar, model_summary, lr_monitor]


# def main(cfg_path):
cfg_path = "./config.yaml"
seed_everything(16)
config_obj = get_train_config_obj(config_path=cfg_path)
model = ParkingTrainingModuleTrain(config_obj)
data = ParkingDataloaderModule(config_obj)

trainer = Trainer(
    callbacks=setup_callbacks(config_obj),
    logger=TensorBoardLogger(save_dir=config_obj.log_dir, default_hp_metric=False),
    accelerator='gpu',
    strategy='ddp_find_unused_parameters_true',
    devices=config_obj.num_gpus,
    max_epochs=config_obj.epochs,
    log_every_n_steps=config_obj.log_every_n_steps,
    check_val_every_n_epoch=config_obj.check_val_every_n_epoch,
    profiler='simple'
)

trainer.fit(
    model=model,
    datamodule=data,
    ckpt_path=config_obj.resume_path
)

# if __name__ == '__main__':
#     main("./config.yaml")
