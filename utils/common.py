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
    Logs a 'scaling_compute' metric at the end of each training epoch:
      compute = num_params * tokens_processed * epoch_time
    """

    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.tokens_processed = 0
        self.epoch_start_time = 0

    def on_train_epoch_start(self, trainer, pl_module):
        # reset counters at the start of each epoch
        self.epoch_start_time = time.time()
        self.tokens_processed = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # assume batch is a dict with 'input_tokens' of shape [B, T] or similar
        # count tokens = batch_size * sequence_length
        self.tokens_processed += batch['input_ids'].shape[0] * batch['input_ids'].shape[1]

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        # total number of model parameters
        num_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        compute = num_params * self.tokens_processed * epoch_time
        # log both to the LightningModule logger and to the progress bar
        pl_module.log('scaling_compute', compute, prog_bar=True, on_epoch=True, sync_dist=True)
        print({'scaling_compute': compute})
        if trainer.logger:
            trainer.logger.log_metrics({'scaling_compute': compute}, step=trainer.current_epoch)


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
