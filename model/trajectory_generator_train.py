from torch import nn

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
        # 1) Teacher-forcing loss (the usual teacher-forcing forward)

        pred_logits, _, _ = self.gen_model(batch)
        if self.cfg.ignore_eos_loss:
            label = batch['labels'][:, :-1]
            pred_logits = pred_logits[:, :-1]
        else:
            label = batch['labels']

        bz, t, vocab = pred_logits.shape
        pred_logits = pred_logits.reshape([bz * t, vocab])
        label = label.reshape(-1).to(self.cfg.device)
        train_loss = self.loss_func(pred_logits, label)
        # 2) Scheduled sampling / AR loss
        if self.current_epoch > self.cfg.ar_start_epoch:
            # Autoregressive scheduled sampling
            if self.cfg.ignore_eos_loss:
                true_tokens = batch['labels'][:, :-1].to(self.cfg.device)
            else:
                true_tokens = batch['labels'].to(self.cfg.device)
            pred_logits_ar = self.gen_model.predict(batch,
                                                    predict_token_num=true_tokens.shape[-1], with_logits=True)[1]
            bz, t, vocab = pred_logits_ar.shape
            autoregressive_loss = self.loss_func(pred_logits_ar.reshape([bz * t, vocab]).float(),
                                                 true_tokens.reshape(-1))
            alpha = min(1.0, (self.current_epoch - self.cfg.ar_start_epoch) / self.cfg.ar_warmup_epochs)
            train_loss = (1 - alpha) * train_loss + alpha * autoregressive_loss

        metrics = {"train_loss": float(train_loss.cpu())}
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss_dict = {}
        pred_logits, _, _ = self.gen_model(batch)
        if self.cfg.ignore_eos_loss:
            label = batch['labels'][:, :-1]
            pred_logits = pred_logits[:, :-1]
        else:
            label = batch['labels']

        bz, t, vocab = pred_logits.shape
        pred_logits = pred_logits.reshape([bz * t, vocab])
        label = label.reshape(-1).to(self.cfg.device)
        teacher_forcing_loss = self.loss_func(pred_logits, label)

        if self.cfg.customized_metric:
            customized_metric = TrajectoryGeneratorMetric(self.cfg)
            pred_logits = pred_logits.reshape([bz, t, vocab])
            true_tokens = label.reshape([bz, t])
            dis = customized_metric.calculate_distance(pred_logits, true_tokens)
            if dis.get("L2_distance", None) is not None:
                val_loss_dict.update({"val_L2_dis": dis.get("L2_distance")})
            if dis.get("hausdorff_distance", None) is not None:
                val_loss_dict.update({"val_hausdorff_dis": dis.get("hausdorff_distance")})
            if dis.get("fourier_difference", None) is not None:
                val_loss_dict.update({"val_fourier_diff": dis.get("fourier_difference")})

        # ----- Autoregressive (Predict) Mode -----
        # Use the predict() method to perform autoregressive generation.
        # 'predict_token_num' should be set according to your generation length requirement.
        if self.cfg.ignore_eos_loss:
            true_tokens = batch['labels'][:, :-1].to(self.cfg.device)
        else:
            true_tokens = batch['labels'].to(self.cfg.device)
        predicted_tokens = self.gen_model.predict(batch, predict_token_num=true_tokens.shape[1])
        # Compute a simple accuracy metric between the generated tokens and the ground truth tokens.
        # Replace with a more task-specific metric if needed.
        val_prediction_accuracy = torch.Tensor(predicted_tokens == true_tokens).to(torch.float32).mean()

        metrics = {
            "val_loss": float(teacher_forcing_loss.to("cpu")),
            "val_accuracy": float(val_prediction_accuracy.to("cpu"))
        }
        val_loss_dict.update(metrics)
        self.log_dict(val_loss_dict, on_epoch=True, prog_bar=True, logger=True)
        return val_loss_dict

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
