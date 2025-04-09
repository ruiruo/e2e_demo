import numpy as np
import torch
import json
from utils.config import Configuration
from utils.trajectory_utils import TrajectoryDistance, detokenize_traj_waypoints


class TrajectoryGeneratorMetric:
    def __init__(self, cfg: Configuration) -> None:
        self.cfg = cfg
        with open(self.cfg.detokenizer, "r") as f:
            self.detokenizer = json.load(f)

    def calculate_distance(self, pred_traj_waypoints, true_traj_waypoints):
        distance_dict = {}
        _prediction_waypoints = self.get_hp_predict_waypoints(pred_traj_waypoints)
        _gt_waypoints = self.get_hp_waypoints(true_traj_waypoints)
        gt_prediction_waypoints_np = []
        gt_waypoints_np = []
        for index in range(self.cfg.batch_size):
            gt_prediction_waypoints_np.append(self.get_valid_np_waypoints(_prediction_waypoints[index]))
            gt_waypoints_np.append(self.get_valid_np_waypoints(_gt_waypoints[index]))
        l2_list, haus_list, fourier_difference = [], [], []
        for index in range(self.cfg.batch_size):
            distance_obj = TrajectoryDistance(gt_prediction_waypoints_np[index], gt_waypoints_np[index])
            if distance_obj.get_len() < 1:
                # If you need at least two points to compute meaningful distances, skip
                continue
            l2_list.append(distance_obj.get_l2_distance())
            if distance_obj.get_len() > 1:
                haus_list.append(distance_obj.get_haus_distance())
                fourier_difference.append(distance_obj.get_fourier_difference())
        if len(l2_list) > 0:
            distance_dict.update({"L2_distance": float(np.mean(l2_list))})
        if len(haus_list) > 0:
            distance_dict.update({"hausdorff_distance": float(np.mean(haus_list))})
        if len(fourier_difference) > 0:
            distance_dict.update({"fourier_difference": float(np.mean(fourier_difference))})
        return distance_dict

    def get_valid_np_waypoints(self, torch_waypoints):
        np_waypoints_valid_detoken = detokenize_traj_waypoints(torch_waypoints, self.detokenizer,
                                                               self.cfg.bos_token,
                                                               self.cfg.eos_token,
                                                               self.cfg.pad_token)
        return np_waypoints_valid_detoken

    def get_hp_predict_waypoints(self, pred_traj_waypoint):
        waypoints = torch.argmax(pred_traj_waypoint, dim=-1)
        return self.get_hp_waypoints(waypoints)

    @staticmethod
    def get_hp_waypoints(batch_waypoints):
        return batch_waypoints[:, 1:-1]
