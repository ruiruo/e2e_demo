from utils.config import Configuration
from utils.trajectory_utils import TrajectoryInfoParser
from utils.trajectory_utils import parallel_find_bin, create_sample
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
        trajectories = torch.from_numpy(self.trajectories[index]).to(torch.int)  # shape (10,)
        labels = torch.from_numpy(self.trajectories_gt[index]).to(torch.int)  # shape (10,)
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
            try:
                traje_info_obj = TrajectoryInfoParser(task_index, task_path, self.cfg)
            except:
                continue
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
                id_e += 1
            self.task_index_list[task_index] = [id_s, id_e]
            id_s, id_e = id_e, id_e
        self.format_transform()

    def _get_all_tasks(self):
        all_tasks = []
        train_data_dir = os.path.join(self.root_dir, self.cfg.training_dir)
        val_data_dir = os.path.join(self.root_dir, self.cfg.validation_dir)
        data_dir = train_data_dir if self.is_train == 1 else val_data_dir
        allow = self.cfg.max_train if self.is_train == 1 else self.cfg.max_val
        for scene_item in os.listdir(data_dir):
            task_path = os.path.join(data_dir, scene_item)
            all_tasks.append(task_path)
            if len(all_tasks) >= allow:
                break
        return all_tasks

    def format_transform(self):
        """
        Convert lists to numpy arrays and call after_process to remove duplicates.
        """
        self.trajectories = np.array(self.trajectories, dtype=np.float32)
        self.trajectories_gt = np.array(self.trajectories_gt, dtype=np.float32)
        self.trajectories_goals = np.array(self.trajectories_goals, dtype=np.float32)
        self.trajectories_agent_info = np.array(self.trajectories_agent_info, dtype=np.float32)
        self.ego_info = np.array(self.ego_info, dtype=np.float32)

        # Remove duplicated entries
        self.after_process()

    def after_process(self):
        """
        Remove duplicate samples (trajectories, trajectories_gt, trajectories_goals).
        Make sure we keep all relevant arrays in sync by reindexing, and *ignore any*
        samples that include '-1' in trajectories, trajectories_gt, or trajectories_goals.
        """
        # 1. FILTER OUT samples that contain -1 in any of the three arrays
        # --------------------------------------------------------------------------
        # Build a boolean mask for each array: `True` means "no -1 inside"
        valid_mask_traj = ~np.any(self.trajectories == -1, axis=1)
        valid_mask_traj_gt = ~np.any(self.trajectories_gt == -1, axis=1)
        valid_mask_traj_goal = ~np.any(self.trajectories_goals == -1, axis=1)

        # Combine them: only keep rows where all three are True
        valid_mask = valid_mask_traj & valid_mask_traj_gt & valid_mask_traj_goal

        # Apply this mask to all arrays
        self.trajectories = self.trajectories[valid_mask]
        self.trajectories_gt = self.trajectories_gt[valid_mask]
        self.trajectories_goals = self.trajectories_goals[valid_mask]
        self.trajectories_agent_info = self.trajectories_agent_info[valid_mask]
        self.ego_info = self.ego_info[valid_mask]

        # --------------------------------------------------------------------------
        # 2. REMOVE DUPLICATES among the filtered samples
        # --------------------------------------------------------------------------
        seen = {}
        unique_indices = []
        for i, (traj, traj_gt) in enumerate(
                zip(self.trajectories, self.trajectories_gt)
        ):
            # Build a hashable key from arrays (or just from traj & traj_gt if you prefer)
            key = (traj.tobytes(), traj_gt.tobytes())
            if key not in seen:
                seen[key] = i
                unique_indices.append(i)

        # Re-slice to keep only unique items across all arrays
        self.trajectories = self.trajectories[unique_indices]
        self.trajectories_gt = self.trajectories_gt[unique_indices]
        self.trajectories_goals = self.trajectories_goals[unique_indices]
        self.trajectories_agent_info = self.trajectories_agent_info[unique_indices]
        self.ego_info = self.ego_info[unique_indices]
