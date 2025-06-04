import torch
from torch import nn

from utils.config import Configuration


class TokenTrajWayPointLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(TokenTrajWayPointLoss, self).__init__()
        self.cfg = cfg
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.cfg.pad_token)

    def forward(self, pred_vocab, labels):
        traj_point_loss = self.ce_loss(pred_vocab, labels.to(torch.long))
        return traj_point_loss
