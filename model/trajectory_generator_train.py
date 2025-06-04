from loss.traj_point_loss import TokenTrajWayPointLoss
from model.trajectory_generator_model import TrajectoryGenerator
from utils.config import Configuration
from utils.metrics import TrajectoryGeneratorMetric
import pytorch_lightning as pl
import torch


class TrajectoryTrainingModule(pl.LightningModule):
    def __init__(self, cfg: Configuration):
        super(TrajectoryTrainingModule, self).__init__()
        self.cfg = cfg

        self.loss_func = TokenTrajWayPointLoss(cfg)
        self.gen_model = TrajectoryGenerator(cfg)
        self._init_model_weights()

    def _init_model_weights(self) -> None:
        """
        Initialize weights for the generator model.
        """

        def init_func(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        self.gen_model.apply(init_func)

    # ──────────────────────────────────────────────────────────────────────────
    # Train / Val steps
    # ──────────────────────────────────────────────────────────────────────────
    def _compute_loss(self, pred_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, T, V = pred_logits.shape
        logits_flat = pred_logits.reshape(B * T, V)
        labels_flat = labels.reshape(-1)
        mask = labels_flat != self.cfg.pad_token
        if self.cfg.ignore_eos_loss:
            mask &= labels_flat != self.cfg.eos_token

        logits_sel = logits_flat[mask]
        labels_sel = labels_flat[mask]

        return self.loss_func(logits_sel, labels_sel)

    def training_step(self, batch, batch_idx):
        # ── Teacher‑Forcing ────────────────────────────────────────────────
        pred_logits, self_state, rep_env = self.gen_model(batch)  # (B, T‑1, V) from new generator
        labels = batch["labels"][:, 1:].to(self.device)
        if self.cfg.ignore_eos_loss:
            labels = labels[:, :-1]
            pred_logits = pred_logits[:, :-1]
        tf_loss = self._compute_loss(pred_logits, labels)

        # Scheduled‑sampling after *ar_start_epoch*
        if self.current_epoch >= self.cfg.ar_start_epoch:
            true_tokens = labels
            _, ar_logits = self.gen_model.predict(batch, predict_token_num=true_tokens.size(1), with_logits=True)
            ar_loss = self._compute_loss(ar_logits, true_tokens)
            alpha = min(1.0, (self.current_epoch - self.cfg.ar_start_epoch) / self.cfg.ar_warmup_epochs)
            loss = (1 - alpha) * tf_loss + alpha * ar_loss
        else:
            loss = tf_loss

        self.log_dict({"train_loss": float(loss)}, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, _):
        labels = batch["labels"][:, 1:].to(self.device)
        # AR rollout
        pred_tokens, pred_logits = self.gen_model.predict(batch, predict_token_num=labels.size(1), with_logits=True)
        if self.cfg.ignore_eos_loss:
            labels = labels[:, :-1]
            pred_tokens = pred_tokens[:, :-1]
            pred_logits = pred_logits[:, :-1]

        val_loss = self._compute_loss(pred_logits, labels)

        trj_metric = TrajectoryGeneratorMetric(self.cfg)
        dists = trj_metric.calculate_distance(pred_tokens.detach(), labels.detach())

        # accuracy (exclude PAD)
        mask = labels != self.cfg.pad_token
        acc = ((pred_tokens == labels) & mask).sum() / mask.sum()

        metrics = {"val_accuracy": acc.detach().cpu(), "train_loss": float(val_loss), **dists}
        self.log_dict(metrics, prog_bar=True, logger=True)
        return metrics


    def configure_optimizers(self):
        opt = torch.optim.RMSprop(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.epochs)
        return {"optimizer": opt, "lr_scheduler": sch}

    def on_before_zero_grad(self, _):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.max_grad_norm)