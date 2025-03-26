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

    def batch_one_step(self, batch):
        pred_label, _, _ = self.gen_model(batch)
        if self.cfg.ignore_bos_loss:
            label = batch['labels'][:, 1:].reshape(-1).cuda()
            pred_label = pred_label[:, 1:]
            bz, t, vocab = pred_label.shape
            pred_label = pred_label.reshape([bz * t, vocab])
        else:
            label = batch['labels'].reshape(-1).cuda()
            bz, t, vocab = pred_label.shape
            pred_label = pred_label.reshape([bz * t, vocab])
        return pred_label, label

    def training_step(self, batch, batch_idx):
        loss_dict = {}
        pred_label, label = self.batch_one_step(batch)
        train_loss = self.loss_func(pred_label, label)
        loss_dict.update({"train_loss": train_loss})
        self.log_dict(loss_dict)
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss_dict = {}
        pred_label, label = self.batch_one_step(batch)

        val_loss = self.loss_func(pred_label, label)

        val_loss_dict.update({"val_loss": val_loss})

        customized_metric = TrajectoryGeneratorMetric(self.cfg)
        val_loss_dict.update(customized_metric.calculate_distance(pred_label, batch))

        self.log_dict(val_loss_dict)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(),
                                        lr=self.cfg.learning_rate,
                                        weight_decay=self.cfg.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.cfg.epochs)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
