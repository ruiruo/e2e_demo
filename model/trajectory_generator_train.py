import numpy as np

from loss.traj_point_loss import TokenTrajWayPointLoss
from model.trajectory_generator_model import TrajectoryGenerator
from utils.config import Configuration
from utils.metrics import TrajectoryGeneratorMetric
import pytorch_lightning as pl
import torch


class TrajectoryTrainingModule(pl.LightningModule):
    def __init__(self, cfg: Configuration):
        super(TrajectoryTrainingModule, self).__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.loss_func = TokenTrajWayPointLoss(self.cfg)

        self.gen_model = TrajectoryGenerator(self.cfg)

        self.init_model_weights()

    def init_model_weights(self):
        """
        Initialize weights for the generator model.
        """

        def init_func(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        self.gen_model.apply(init_func)

    def training_step(self, batch, batch_idx):
        # --- teacher‑forcing forward ---
        pred_logits, _, _ = self.gen_model(batch)
        if self.cfg.ignore_eos_loss:
            labels = batch['labels'][:, :-1]
            pred_logits = pred_logits[:, :-1]
        else:
            labels = batch['labels']

        B, T, V = pred_logits.shape
        logits_flat = pred_logits.reshape(B * T, V)
        labels_flat = labels.reshape(-1).to(self.cfg.device)

        # mask out PAD positions
        non_pad = labels_flat != self.cfg.pad_token
        logits_flat = logits_flat[non_pad]
        labels_flat = labels_flat[non_pad]

        tf_loss = self.loss_func(logits_flat, labels_flat)

        # --- scheduled‑sampling / AR loss ---
        if self.current_epoch > self.cfg.ar_start_epoch:
            # prepare true tokens (same slicing as above)
            if self.cfg.ignore_eos_loss:
                true_tokens = batch['labels'][:, :-1].to(self.cfg.device)
            else:
                true_tokens = batch['labels'].to(self.cfg.device)

            # generate logits for AR steps
            _, ar_logits = self.gen_model.predict(
                batch,
                predict_token_num=true_tokens.shape[1],
                with_logits=True
            )
            B_ar, T_ar, V_ar = ar_logits.shape

            ar_flat = ar_logits.reshape(B_ar * T_ar, V_ar)
            true_flat = true_tokens.reshape(-1)

            # mask out pads
            non_pad_ar = true_flat != self.cfg.pad_token
            ar_flat = ar_flat[non_pad_ar]
            true_flat = true_flat[non_pad_ar]

            ar_loss = self.loss_func(ar_flat.float(), true_flat)
            alpha = min(1.0,
                        (self.current_epoch - self.cfg.ar_start_epoch)
                        / self.cfg.ar_warmup_epochs)
            train_loss = (1 - alpha) * tf_loss + alpha * ar_loss
        else:
            train_loss = tf_loss

        self.log("train_loss", train_loss, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        # teacher‑forcing forward
        pred_logits, _, _ = self.gen_model(batch)
        if self.cfg.ignore_eos_loss:
            labels = batch['labels'][:, :-1]
            pred_logits = pred_logits[:, :-1]
        else:
            labels = batch['labels']

        B, T, V = pred_logits.shape
        logits_flat = pred_logits.reshape(B * T, V)
        labels_flat = labels.reshape(-1).to(self.cfg.device)

        # mask out pads for loss
        nonpad = labels_flat != self.cfg.pad_token
        logits_flat = logits_flat[nonpad]
        labels_flat = labels_flat[nonpad]

        tf_loss = self.loss_func(logits_flat, labels_flat)
        metrics = {"val_loss": tf_loss.detach().cpu()}

        # optional custom distance metrics
        if self.cfg.customized_metric:
            metric = TrajectoryGeneratorMetric(self.cfg)
            # Use the **un-flattened** pred_logits and labels
            d = metric.calculate_distance(pred_logits.detach(), labels.to(self.cfg.device))
            if "L2_distance" in d:
                metrics["val_L2_dis"] = d["L2_distance"]
            if "hausdorff_distance" in d:
                metrics["val_hausdorff_dis"] = d["hausdorff_distance"]
            if "fourier_difference" in d:
                metrics["val_fourier_diff"] = d["fourier_difference"]

        # autoregressive accuracy (ignore PADs)
        if self.cfg.ignore_eos_loss:
            true_tokens = batch['labels'][:, :-1].to(self.cfg.device)
        else:
            true_tokens = batch['labels'].to(self.cfg.device)

        gen_tokens = self.gen_model.predict(
            batch,
            predict_token_num=true_tokens.shape[1]
        )
        mask = torch.Tensor(true_tokens != self.cfg.pad_token)
        correct = torch.Tensor((gen_tokens == true_tokens) & mask)
        acc = correct.sum() / mask.sum()
        metrics["val_accuracy"] = acc.detach().cpu()

        self.log_dict(metrics, prog_bar=True, logger=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.cfg.epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def on_before_zero_grad(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.max_grad_norm)
