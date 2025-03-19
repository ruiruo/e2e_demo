import json
from utils.decorator_train import finish, init
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary, TQDMProgressBar


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
