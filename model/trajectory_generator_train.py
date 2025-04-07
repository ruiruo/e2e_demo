import pytorch_lightning as pl
import torch

from loss.traj_point_loss import TokenTrajWayPointLoss
from model.trajectory_generator_model import TrajectoryGenerator
from utils.config import Configuration
from utils.metrics import TrajectoryGeneratorMetric


class TrajectoryTrainingModule(pl.LightningModule):
    def __init__(self, cfg: Configuration):
        super(TrajectoryTrainingModule, self).__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.loss_func = TokenTrajWayPointLoss(self.cfg)

        self.gen_model = TrajectoryGenerator(self.cfg)

        # Initialize model weights
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

    def batch_one_step(self, batch):
        pred_label, _, _ = self.gen_model(batch)
        if self.cfg.ignore_bos_loss:
            label = batch['labels'][:, 1:].reshape(-1).to(self.cfg.device)
            pred_label = pred_label[:, 1:]
            bz, t, vocab = pred_label.shape
            pred_label = pred_label.reshape([bz * t, vocab])
        else:
            label = batch['labels'].reshape(-1).to(self.cfg.device)
            bz, t, vocab = pred_label.shape
            pred_label = pred_label.reshape([bz * t, vocab])
        return pred_label, label

    def training_step(self, batch, batch_idx):
        pred_label, label = self.batch_one_step(batch)
        train_loss = self.loss_func(pred_label, label)
        metrics = {"train_loss": float(train_loss.to("cpu"))}
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        # ----- Teacher Forcing Mode -----
        # Use the forward() pass to get the logits and compute the cross-entropy loss.
        pred_label, label = self.batch_one_step(batch)
        # Adjust labels if needed, for example, ignoring the BOS token
        teacher_forcing_loss = self.loss_func(pred_label, label)
        customized_metric = TrajectoryGeneratorMetric(self.cfg)
        teacher_forcing_dis = customized_metric.calculate_distance(pred_label, batch)

        # ----- Autoregressive (Predict) Mode -----
        # Use the predict() method to perform autoregressive generation.
        # 'predict_token_num' should be set according to your generation length requirement.
        predicted_tokens = self.gen_model.predict(batch, predict_token_num=self.cfg.max_frame)

        # For demonstration, we assume the ground truth tokens to compare against are the labels,
        # skipping the BOS token if needed. Adjust the slicing as appropriate.
        true_tokens = batch['labels'][:, 1:1 + predicted_tokens.size(1)].to(self.cfg.device)

        # Compute a simple accuracy metric between the generated tokens and the ground truth tokens.
        # Replace with a more task-specific metric if needed.
        val_prediction_accuracy = torch.Tensor(predicted_tokens == true_tokens).to(torch.float32).mean()

        # ----- Log both metrics -----
        metrics = {
            "val_teacher_forcing_loss": float(teacher_forcing_loss.to("cpu")),
            "val_teacher_forcing_L2_dis": teacher_forcing_dis["L2_distance"],
            "val_prediction_accuracy": float(val_prediction_accuracy.to("cpu"))
        }
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(),
                                        lr=self.cfg.learning_rate,
                                        weight_decay=self.cfg.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.cfg.epochs)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
