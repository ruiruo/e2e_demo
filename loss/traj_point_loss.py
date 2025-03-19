from torch import nn

from utils.config import Configuration


class TokenTrajWayPointLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(TokenTrajWayPointLoss, self).__init__()
        self.cfg = cfg
        self.PAD_token = self.cfg.token_nums + 2
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.PAD_token)

    def forward(self, pred_label, data):
        pred_label = pred_label[:, :-1, :]
        pred_labels = pred_label.reshape(-1, pred_label.shape[-1])
        labels = data['labels'][:, 1:-1].reshape(-1).cuda()
        traj_point_loss = self.ce_loss(pred_labels, labels)
        return traj_point_loss
