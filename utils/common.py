import json
import time
from utils.decorator_train import finish, init
from pytorch_lightning.callbacks import Callback
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


class ScalingLawCallback(Callback):
    """
    1) Logs per-epoch `scaling_compute_epoch`
    2) At the end, logs a single `scaling_compute_total` = sum of all epochs.
    """

    def __init__(self):
        super().__init__()
        self.tokens = 0
        self.epoch_start = None
        self.epoch_compute_values = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.tokens = 0
        self.epoch_start = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        B, T = batch['input_ids'].shape[:2]
        self.tokens += B * T

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start
        num_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        C_e = num_params * self.tokens * epoch_time

        # log per-epoch
        pl_module.log('scaling_compute_epoch', C_e, prog_bar=True, on_epoch=True)
        self.epoch_compute_values.append(C_e)

    def on_train_end(self, trainer, pl_module):
        C_total = sum(self.epoch_compute_values)
        pl_module.log('scaling_compute_total', C_total)
        print(f"\n>>> Total scaling_compute: {C_total:.3e}")


def setup_callbacks(cfg_obj, monitor, mode):
    ckpt_callback = ModelCheckpoint(
        dirpath=cfg_obj.checkpoint_dir,
        monitor=monitor,
        save_top_k=100,
        mode=mode,
        every_n_epochs=20,
        filename="{epoch:02d}-{%s:.2f}" % monitor,
        save_last=True
    )
    progress_bar = TQDMProgressBar()
    model_summary = ModelSummary(max_depth=3)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    scaling_law = ScalingLawCallback(cfg_obj.max_frame)

    return [ckpt_callback, progress_bar, model_summary, lr_monitor, scaling_law]
