from dataset.dataset_driving import TrajectoryDataModule
from torch.utils.data import DataLoader
from utils.config import Configuration
import numpy as np
import pytorch_lightning as pl
import random
import torch
import os


class TrajectoryDataloaderModule(pl.LightningDataModule):
    def __init__(self, cfg: Configuration, pred_path=None):
        super().__init__()
        self.cfg = cfg
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.pred_path = pred_path

    def setup(self, stage: str = 'fit'):
        if stage == 'fit':
            train_path = os.path.join(self.cfg.data_dir, self.cfg.training_dir)
            self.train_loader = DataLoader(dataset=TrajectoryDataModule(config=self.cfg,
                                                                        data_path=train_path,
                                                                        max_allow=self.cfg.max_train),
                                           batch_size=self.cfg.batch_size,
                                           shuffle=True,
                                           num_workers=self.cfg.num_workers,
                                           pin_memory=True,
                                           worker_init_fn=self.seed_worker,
                                           drop_last=True,
                                           )
            val_path = os.path.join(self.cfg.data_dir, self.cfg.validation_dir)
            self.val_loader = DataLoader(dataset=TrajectoryDataModule(config=self.cfg,
                                                                      data_path=val_path,
                                                                      max_allow=self.cfg.max_val),
                                         batch_size=self.cfg.batch_size,
                                         shuffle=False,
                                         num_workers=self.cfg.num_workers,
                                         pin_memory=True,
                                         worker_init_fn=self.seed_worker,
                                         drop_last=True)
        elif stage == 'test' or stage == 'predict':
            self.test_loader = DataLoader(dataset=TrajectoryDataModule(config=self.cfg,
                                                                       data_path=self.pred_path,
                                                                       max_allow=100000000),
                                          batch_size=self.cfg.batch_size,
                                          shuffle=False,
                                          num_workers=self.cfg.num_workers,
                                          pin_memory=True,
                                          worker_init_fn=self.seed_worker,
                                          drop_last=True)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
