import numpy as np
import torch

from utils.config import Configuration
from utils.trajectory_utils import TrajectoryDistance, detokenize_traj_waypoint


class TrajectoryGeneratorMetric:
    def __init__(self, cfg: Configuration, pred_traj_waypoint, batch) -> None:
        self.cfg = cfg
        self.BOS_token = self.cfg.token_nums
        self.distance_dict = self.calculate_distance(pred_traj_waypoint, batch)

    def calculate_distance(self, pred_traj_waypoints, batch):
        distance_dict = {}
        prediction_waypoints = self.get_predict_waypoints(pred_traj_waypoints)
        gt_waypoints = self.get_gt_waypoints(batch['labels'])

        prediction_waypoints = prediction_waypoints.reshape(self.cfg.batch_size, -1, self.cfg.item_number)
        gt_waypoints = gt_waypoints.reshape(self.cfg.batch_size, -1, self.cfg.item_number)
        valid_mask = ((gt_waypoints < self.BOS_token) & (prediction_waypoints < self.BOS_token)).all(dim=-1)
        prediction_waypoints_np = []
        gt_waypoints_np = []
        for index in range(self.cfg.batch_size):
            prediction_waypoints_np.append(self.get_valid_np_waypoints(prediction_waypoints[index], valid_mask[index]))
            gt_waypoints_np.append(self.get_valid_np_waypoints(gt_waypoints[index], valid_mask[index]))

        l2_list, haus_list, fourier_difference = [], [], []
        for index in range(self.cfg.batch_size):
            distance_obj = TrajectoryDistance(prediction_waypoints_np[index], gt_waypoints_np[index])
            if distance_obj.get_len() < 1:
                continue
            l2_list.append(distance_obj.get_l2_distance())
            if distance_obj.get_len() > 1:
                haus_list.append(distance_obj.get_haus_distance())
                fourier_difference.append(distance_obj.get_fourier_difference())
        if len(l2_list) > 0:
            distance_dict.update({"L2_distance": np.mean(l2_list)})
        if len(haus_list) > 0:
            distance_dict.update({"hausdorff_distance": np.mean(haus_list)})
        if len(fourier_difference) > 0:
            distance_dict.update({"fourier_difference": np.mean(fourier_difference)})
        return distance_dict

    def get_valid_np_waypoints(self, torch_waypoints, valid_mask):
        torch_waypoints_valid = torch_waypoints[valid_mask]
        torch_waypoints_valid_detoken = detokenize_traj_waypoint(torch_waypoints_valid,
                                                              token_nums=self.cfg.token_nums,
                                                              item_num=self.cfg.item_number,
                                                              xy_max=self.cfg.xy_max)
        torch_waypoints_valid_detoken = torch_waypoints_valid_detoken[:, :2]
        np_waypoints_valid_detoken = np.array(torch_waypoints_valid_detoken.cpu())
        return np_waypoints_valid_detoken

    def get_predict_waypoints(self, pred_traj_waypoint):
        prediction = torch.softmax(pred_traj_waypoint, dim=-1)
        prediction = prediction[:, :-2, :]
        prediction = prediction.argmax(dim=-1).view(-1, self.cfg.item_number)
        return prediction

    def get_gt_waypoints(self, batch_gt_waypoints):
        return batch_gt_waypoints[:, 1:-2].reshape(-1).view(-1, self.cfg.item_number)
