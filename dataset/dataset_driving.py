from utils.config import Configuration
from utils.trajectory_utils import TrajectoryInfoParser, TrajectoryDistance
from utils.trajectory_utils import parallel_find_bin, create_sample, detokenize_traj_waypoints
from types import SimpleNamespace
import json
import numpy as np
import os
import torch
import tqdm


class TrajectoryDataModule(torch.utils.data.Dataset):
    def __init__(self, config: Configuration, is_train):
        super(TrajectoryDataModule, self).__init__()
        self.cfg = config
        self.BOS_token = self.cfg.bos_token
        self.EOS_token = self.cfg.eos_token
        self.PAD_token = self.cfg.pad_token
        self.x_boundaries = np.array(self.cfg.x_boundaries)
        self.y_boundaries = np.array(self.cfg.y_boundaries)
        self.local2token = np.load(self.cfg.tokenizer)
        # maybe not necessary here ?
        with open(self.cfg.detokenizer, "r") as f:
            self.detokenizer = SimpleNamespace(**json.load(f))

        self.root_dir = self.cfg.data_dir
        self.is_train = is_train
        self.ego_info = []
        self.goal_info = []
        self.trajectories = []
        self.trajectories_gt = []
        self.trajectories_goals = []
        self.trajectories_agent_info = []
        # TODO: for eval
        self.trajectories_raw = []
        self.trajectories_gt_raw = []
        self.error_under_12 = []
        self.error_over_12 = []
        self.task_index_list = {}
        self.create_gt_data()

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        """
        Retrieve a training sample from the preprocessed trajectory arrays for autoregressive (forward-only) model training.

        Returns a dictionary containing:
            - input_ids: The input sequence for the autoregressive model
                         (e.g., [BOS_token, token1, token2, ..., tokenN])
            - labels: The corresponding target sequence (e.g., [token1, token2, ..., tokenN, EOS_token])
            - goal: The goal information (shape: (2,))
            - agent_features: (Optional) Agent or background features if available
        """
        # Extract the training sample from the preprocessed arrays based on the index.
        trajectories = torch.from_numpy(self.trajectories[index]).to(torch.int)  # shape (11,)
        labels = torch.from_numpy(self.trajectories_gt[index]).to(torch.int)  # shape (11,)
        goal = torch.from_numpy(self.trajectories_goals[index]).to(torch.int)  # shape (2,)
        agent_info = torch.from_numpy(self.trajectories_agent_info[index]).to(torch.float32)
        ego_features = torch.from_numpy(self.ego_info[index]).to(torch.float32)
        return {
            "input_ids": trajectories,
            "labels": labels,
            "agent_info": agent_info,
            "ego_info": ego_features,
            "goal": goal,
        }

    def create_gt_data(self):
        all_tasks = self._get_all_tasks()
        id_s, id_e = 0, 0
        for task_index, task_path in tqdm.tqdm(enumerate(all_tasks)):
            # task iteration
            # todo, could be mutil process
            traje_info_obj = TrajectoryInfoParser(task_index, task_path, self.cfg.max_frame)
            for traje_id, trajectory in enumerate(traje_info_obj.trajectories):

                if self.cfg.multi_agent_info:
                    # TODO: generate (n-1) * trajectories by multi agent info
                    raise NotImplementedError
                else:
                    input_ids, labels, agent_info = create_sample(trajectory.info["ego_info"][:, 0]
                                                                  , trajectory.info["agent_info"],
                                                                  self.BOS_token, self.EOS_token,
                                                                  self.PAD_token, self.cfg.max_frame)
                    goal_info = parallel_find_bin(np.expand_dims(trajectory.info["goal_info"], 0),
                                                  self.x_boundaries, self.y_boundaries)
                    goal_info = (int(goal_info[0]), int(goal_info[1]))
                    self.ego_info.append(trajectory.info["ego_info"][0, 1:])
                    self.goal_info.append(trajectory.info["ego_info"][-1, 1:])
                    self.trajectories.append(input_ids)
                    self.trajectories_gt.append(labels)
                    self.trajectories_goals.append(np.array([self.local2token[goal_info]]))
                    self.trajectories_agent_info.append(agent_info)
                    # TODO: move those computing to TrajectoryInfoParser
                    # self.trajectories_raw.append(ego_pose)
                    # self.trajectories_gt_raw.append(ego_pose[1:] + self.EOS_token)
                    # self._calc_error_with_token(trajectory.info["ego_info"], ego_token)
                id_e += 1
            self.task_index_list[task_index] = [id_s, id_e]
            id_s, id_e = id_e, id_e
        # print(sum(self.error_under_12) / len(self.error_under_12), sum(self.error_over_12) / len(self.error_over_12))
        # print(self.error_under_12, self.error_over_12)
        self.format_transform()

    def _calc_error_with_token(self, raw_data: np.ndarray, ego_token: np.ndarray):
        ego_raw_traj = raw_data[:, 0:2]
        if ego_raw_traj[-1, 0] < 12:
            ego_traj_with_token = detokenize_traj_waypoints(ego_token, self.detokenizer)
            token_error = TrajectoryDistance(ego_traj_with_token, ego_raw_traj)
            self.error_under_12.append(token_error.get_l2_distance())
        else:
            ego_traj_with_token = detokenize_traj_waypoints(ego_token, self.detokenizer)
            token_error = TrajectoryDistance(ego_traj_with_token, ego_raw_traj)
            self.error_over_12.append(token_error.get_l2_distance())

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

    def format_transform(self):
        self.trajectories = np.array(self.trajectories).astype(np.float32)
        self.trajectories_gt = np.array(self.trajectories_gt).astype(np.float32)
        self.trajectories_goals = np.array(self.trajectories_goals).astype(np.float32)
