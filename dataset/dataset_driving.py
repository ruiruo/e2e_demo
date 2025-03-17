import json

from utils.config import Configuration
from utils.trajectory_utils import TrajectoryInfoParser, TopologyHistory
from utils.trajectory_utils import tokenize_traj_waypoints, detokenize_traj_waypoint
import numpy as np
import os
import pandas as pd
import pickle
import torch
import torch.utils.data
import tqdm

from utils.config import Configuration


class TrajectoryDataModule(torch.utils.data.Dataset):
    def __init__(self, config: Configuration, is_train):
        super(TrajectoryDataModule, self).__init__()
        self.cfg = config
        self.BOS_token = self.cfg.token_nums
        self.EOS_token = self.cfg.token_nums + 1
        self.PAD_token = self.cfg.token_nums + 2
        self.x_boundaries = np.array(self.cfg.x_boundaries)
        self.y_boundaries = np.array(self.cfg.y_boundaries)
        self.local2token = np.load(self.cfg.tokenizer)
        with open(self.cfg.detokenizer, "r") as f:
            self.detokenizer = json.load(f)

        self.root_dir = self.cfg.data_dir
        self.is_train = is_train
        self.raw_data = []

        # self.feature = {}
        # self.task_index_list = []
        #
        # self.fuzzy_target_point = []
        # self.traj_point = []
        # self.traj_point_token = []
        # self.target_point = []
        self.create_gt_data()

    # def __len__(self):>
    #     # 这里假设 feature 中某个字段的长度代表数据集大小
    #     return len(self.feature.get("", []))

    def __getitem__(self, index):
        data = {}
        keys = ['target_point', 'gt_traj_point', 'gt_traj_topology', 'fuzzy_target_point']
        # for key in keys:
        #     data[key] = []

        # data["gt_traj_point"] = torch.from_numpy(np.array(self.traj_point[index]))
        # # 依次调用每个 Topology 对象的 preprocess 方法，并转为 tensor
        # processed_topology = [torch.from_numpy(topo.preprocess()) for topo in self.raw_topologies[index]]
        # data['gt_traj_topology'] = processed_topology
        # data['target_point'] = torch.from_numpy(self.target_point[index])
        # data["fuzzy_target_point"] = torch.from_numpy(self.fuzzy_target_point[index])

        return data

    def process_record(self, record_path):
        _ = self.cfg
        return None

    def create_gt_data(self):
        all_tasks = self._get_all_tasks()
        for task_index, task_path in tqdm.tqdm(enumerate(all_tasks)):
            # task iteration
            # todo, could be mutil process
            traje_info_obj = TrajectoryInfoParser(task_index, task_path)

            for traje_id, trajectory in enumerate(traje_info_obj.trajectories):
                ego_pose = trajectory.info["ego_info"][:, 0:2]
                ego_token = tokenize_traj_waypoints(ego_pose,
                                                    self.x_boundaries, self.y_boundaries,
                                                    self.local2token)
                print(ego_token)



    #             world2ego_mat = ego_pose.get_homogeneous_transformation().get_inverse_matrix()
    #             # 生成预测轨迹点（token 化和原始数值）
    #             predict_point_token_gt, predict_point_gt = self.create_predict_point_gt(traje_info_obj, ego_index, world2ego_mat)
    #             # 生成停车目标
    #             fuzzy_parking_goal, parking_goal = self.create_parking_goal_gt(traje_info_obj, world2ego_mat)
    #
    #             self.traj_point.append(predict_point_gt)
    #             self.traj_point_token.append(predict_point_token_gt)
    #             self.target_point.append(parking_goal)
    #             self.fuzzy_target_point.append(fuzzy_parking_goal)
    #             # 为每个 token 预留原始拓扑信息，使用 Topology 对象封装，当前以零矩阵作为占位
    #             topology_list = [Topology(np.zeros((200, 8), dtype=np.float32)) for _ in range(len(predict_point_token_gt))]
    #             self.raw_topologies.append(topology_list)
    #             self.task_index_list.append(task_index)
    #
    #     self.format_transform()
    #
    # def create_predict_point_gt(self, traje_info_obj: TrajectoryInfoParser, ego_index: int,
    #                             world2ego_mat: np.array) -> list:
    #     predict_point, predict_point_token = [], []
    #     for predict_index in range(self.cfg.autoregressive_points):
    #         predict_stride_index = self.get_clip_stride_index(
    #             predict_index=predict_index,
    #             start_index=ego_index,
    #             max_index=traje_info_obj.total_frames - 1,
    #             stride=self.cfg.traj_downsample_stride
    #         )
    #         predict_pose_in_world = traje_info_obj.get_trajectory_point(predict_stride_index)
    #         predict_pose_in_ego = predict_pose_in_world.get_pose_in_ego(world2ego_mat)
    #         progress = traje_info_obj.get_progress(predict_stride_index)
    #         predict_point.append([predict_pose_in_ego.x, predict_pose_in_ego.y])
    #         tokenize_ret = tokenize_traj_point(predict_pose_in_ego.x, predict_pose_in_ego.y,
    #                                            progress, self.cfg.token_nums, self.cfg.xy_max)
    #         tokenize_ret_process = tokenize_ret[:2] if self.cfg.item_number == 2 else tokenize_ret
    #         predict_point_token.append(tokenize_ret_process)
    #
    #         if predict_stride_index == traje_info_obj.total_frames - 1 or predict_index == self.cfg.autoregressive_points - 1:
    #             break
    #
    #     predict_point_gt = [item for sublist in predict_point for item in sublist]
    #     append_pad_num = self.cfg.autoregressive_points * self.cfg.item_number - len(predict_point_gt)
    #     assert append_pad_num >= 0
    #     if len(predict_point_gt) > 0:
    #         predict_point_gt = predict_point_gt + (append_pad_num // 2) * [predict_point_gt[-2], predict_point_gt[-1]]
    #     predict_point_token_gt = [item for sublist in predict_point_token for item in sublist]
    #     predict_point_token_gt.insert(0, self.BOS_token)
    #     predict_point_token_gt.append(self.EOS_token)
    #     predict_point_token_gt.append(self.PAD_token)
    #     append_pad_num = self.cfg.autoregressive_points * self.cfg.item_number + self.cfg.append_token - len(predict_point_token_gt)
    #     assert append_pad_num >= 0
    #     predict_point_token_gt = predict_point_token_gt + append_pad_num * [self.PAD_token]
    #     return predict_point_token_gt, predict_point_gt
    #
    # def to_trajectory(self, traje_info_obj: TopologyHistory):
    #     # frames = traje_info_obj.
    #
    #     candidate_target_pose_in_world = traje_info_obj.get_random_candidate_target_pose()
    #     candidate_target_pose_in_ego = candidate_target_pose_in_world.get_pose_in_ego(world2ego_mat)
    #     fuzzy_parking_goal = [candidate_target_pose_in_ego.x, candidate_target_pose_in_ego.y]
    #
    #     target_pose_in_world = traje_info_obj.get_precise_target_pose()
    #     target_pose_in_ego = target_pose_in_world.get_pose_in_ego(world2ego_mat)
    #     parking_goal = [target_pose_in_ego.x, target_pose_in_ego.y]
    #
    #     return fuzzy_parking_goal, parking_goal

    def _get_all_tasks(self):
        all_tasks = []
        train_data_dir = os.path.join(self.root_dir, self.cfg.training_dir)
        val_data_dir = os.path.join(self.root_dir, self.cfg.validation_dir)
        data_dir = train_data_dir if self.is_train == 1 else val_data_dir
        for scene_item in os.listdir(data_dir):
            scene_path = os.path.join(data_dir, scene_item)
            task_path = os.path.join(scene_path, scene_path)
            all_tasks.append(task_path)
        return all_tasks

    # def format_transform(self):
    #     self.traj_point = np.array(self.traj_point).astype(np.float32)
    #     self.traj_point_token = np.array(self.traj_point_token).astype(np.int64)
    #     self.target_point = np.array(self.target_point).astype(np.float32)
    #     self.fuzzy_target_point = np.array(self.fuzzy_target_point).astype(np.float32)
    #     self.task_index_list = np.array(self.task_index_list).astype(np.int64)
    #
    # def get_clip_stride_index(self, predict_index, start_index, max_index, stride):
    #     return int(np.clip(start_index + stride * (1 + predict_index), 0, max_index))
