from utils.config import Configuration
from utils.trajectory_utils import TrajectoryInfoParser
from utils.trajectory_utils import tokenize_traj_waypoints, parallel_find_bin, create_sample
import json
import numpy as np
import os
import torch
import tqdm


class TrajectoryDataModule(torch.utils.data.Dataset):
    def __init__(self, config: Configuration, is_train):
        super(TrajectoryDataModule, self).__init__()
        self.cfg = config
        self.BOS_token = self.cfg.bos_id
        self.EOS_token = self.cfg.token_nums + 1
        self.PAD_token = self.cfg.token_nums + 2
        self.x_boundaries = np.array(self.cfg.x_boundaries)
        self.y_boundaries = np.array(self.cfg.y_boundaries)
        self.local2token = np.load(self.cfg.tokenizer)
        with open(self.cfg.detokenizer, "r") as f:
            self.detokenizer = json.load(f)

        self.root_dir = self.cfg.data_dir
        self.is_train = is_train
        self.trajectories = []
        self.trajectories_gt = []
        self.trajectories_goals = []
        self.trajectories_features = []
        # TODO: for eval
        self.trajectories_raw = []
        self.trajectories_gt_raw = []
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
        trajectories = torch.from_numpy(self.trajectories[index])  # shape (11,)
        labels = torch.from_numpy(self.trajectories_gt[index])  # shape (11,)
        goal = torch.from_numpy(self.trajectories_goals[index])  # shape (2,)
        # If agent features were generated in create_gt_data (e.g., self.trajectories_features), return them as well.
        agent_features = torch.from_numpy(self.trajectories_features[index])
        return {
            "input_ids": trajectories,
            "labels": labels,
            "goal": goal,
            "agent_features": agent_features
        }

    def create_gt_data(self):
        all_tasks = self._get_all_tasks()
        id_s, id_e = 0, 0
        for task_index, task_path in tqdm.tqdm(enumerate(all_tasks)):
            # task iteration
            # todo, could be mutil process
            traje_info_obj = TrajectoryInfoParser(task_index, task_path, self.cfg.max_frame)
            for traje_id, trajectory in enumerate(traje_info_obj.trajectories):
                ego_pose = trajectory.info["ego_info"][:, 0:2]
                ego_token = tokenize_traj_waypoints(ego_pose,
                                                    self.x_boundaries, self.y_boundaries,
                                                    self.local2token)
                if self.cfg.multi_agent_info:
                    # TODO: generate (n-1) * trajectories by multi agent info
                    raise NotImplementedError
                else:
                    input_ids, labels, agent_info = create_sample(ego_token, trajectory.info["agent_info"][0],
                                                                  self.BOS_token, self.EOS_token,
                                                                  self.PAD_token, self.cfg.max_frame)
                    goal_info = parallel_find_bin(np.expand_dims(trajectory.info["goal_info"], 0),
                                                  self.x_boundaries, self.y_boundaries)
                    goal_info = (int(goal_info[0]), int(goal_info[1]))
                    self.trajectories.append(input_ids)
                    self.trajectories_gt.append(labels)
                    self.trajectories_goals.append(np.array([self.local2token[goal_info]]))
                    self.trajectories_features.append(agent_info)
                id_e += 1
            self.task_index_list[task_index] = [id_s, id_e]
            id_s, id_e = id_e, id_e
        self.format_transform()

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
