from utils.config import Configuration
from utils.trajectory_utils import TrajectoryInfoParser
from utils.trajectory_utils import parallel_find_bin, create_sample, detokenize_traj_waypoints
import multiprocessing as mp
import numpy as np
import os
import torch
import tqdm
import json


def process_task(args):
    task_index, task_path, cfg, BOS_token, EOS_token, PAD_token, x_boundaries, y_boundaries, local2token, detokenizer = args
    trajectories_list = []
    trajectories_gt_list = []
    trajectories_goals_list = []
    trajectories_agent_info_list = []
    ego_info_list = []
    try:
        traje_info_obj = TrajectoryInfoParser(task_index, task_path, cfg)
    except Exception as e:
        print("Ignore TrajectoryInfoParser task")
        # Log exception or ignore this task
        return trajectories_list, trajectories_gt_list, trajectories_goals_list, trajectories_agent_info_list, ego_info_list

    for trajectory in traje_info_obj.trajectories:
        if cfg.multi_agent_info:
            # Handle multi-agent if implemented, or skip with a log
            raise NotImplementedError
        else:
            input_ids, labels, agent_info = create_sample(
                trajectory.info["ego_info"][:, 0],
                trajectory.info["agent_info"],
                BOS_token, EOS_token, PAD_token,
                cfg.max_frame
            )
            goal_info = parallel_find_bin(
                np.expand_dims(trajectory.info["goal_info"], 0),
                x_boundaries, y_boundaries
            )
            goal_info = (int(goal_info[0]), int(goal_info[1]))
            trajectories_list.append(input_ids)
            trajectories_gt_list.append(labels)
            trajectories_goals_list.append(np.array([local2token[goal_info]]))
            trajectories_agent_info_list.append(agent_info)
            ego_info_list.append(trajectory.info["ego_info"][0, 1:])
    return trajectories_list, trajectories_gt_list, trajectories_goals_list, trajectories_agent_info_list, ego_info_list


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
        with open(self.cfg.detokenizer, "r") as f:
            self.detokenizer = json.load(f)

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
        all_tasks = self._get_all_tasks()  # list of task paths
        # Build a list of arguments for each task
        args_list = [
            (task_index, task_path, self.cfg, self.BOS_token, self.EOS_token, self.PAD_token,
             self.x_boundaries, self.y_boundaries, self.local2token, self.detokenizer)
            for task_index, task_path in enumerate(all_tasks)
        ]
        # Create a process pool
        with mp.Pool(processes=self.cfg.num_workers) as pool:
            # Use imap_unordered for possibly faster results and integration with tqdm
            results = list(tqdm.tqdm(pool.imap(process_task, args_list), total=len(args_list)))

        for res in results:
            traj, traj_gt, traj_goal, traj_agent, ego = res
            self.trajectories.extend(traj)
            self.trajectories_gt.extend(traj_gt)
            self.trajectories_goals.extend(traj_goal)
            self.trajectories_agent_info.extend(traj_agent)
            self.ego_info.extend(ego)
        # Optionally, rebuild task_index_list if needed.
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
        Additionally, filter out samples whose final detokenized displacement is <= 10.
        """
        # --------------------------------------------------------------------------
        # 1. REMOVE DUPLICATES among the filtered samples
        # --------------------------------------------------------------------------
        seen = {}
        unique_indices = []
        if self.cfg.simple_deduction:
            data = zip(self.trajectories, self.trajectories_gt, self.trajectories_agent_info[:, 0, :, 0])
        else:
            data = zip(self.trajectories, self.trajectories_gt, self.trajectories_agent_info[:, :, 0])
        for i, (traj, traj_gt, traj_agent) in enumerate(data):
            key = (traj.tobytes(), traj_gt.tobytes(), traj_agent.tobytes())
            if key not in seen:
                seen[key] = i
                unique_indices.append(i)

        # Re-slice to keep only unique items across all arrays
        self.trajectories = self.trajectories[unique_indices]
        self.trajectories_gt = self.trajectories_gt[unique_indices]
        self.trajectories_goals = self.trajectories_goals[unique_indices]
        self.trajectories_agent_info = self.trajectories_agent_info[unique_indices]
        self.ego_info = self.ego_info[unique_indices]

        # --------------------------------------------------------------------------
        # 2. FILTER OUT samples that contain -1 in any of the three arrays
        # --------------------------------------------------------------------------
        valid_mask_traj = ~np.any(self.trajectories == -1, axis=1)
        valid_mask_traj_gt = ~np.any(self.trajectories_gt == -1, axis=1)
        valid_mask_traj_goal = ~np.any(self.trajectories_goals == -1, axis=1)
        valid_mask = valid_mask_traj & valid_mask_traj_gt & valid_mask_traj_goal

        self.trajectories = self.trajectories[valid_mask]
        self.trajectories_gt = self.trajectories_gt[valid_mask]
        self.trajectories_goals = self.trajectories_goals[valid_mask]
        self.trajectories_agent_info = self.trajectories_agent_info[valid_mask]
        self.ego_info = self.ego_info[valid_mask]

        # --------------------------------------------------------------------------
        # 3. FILTER OUT samples with final detokenized displacement (dis) <= 10
        # --------------------------------------------------------------------------
        # For each sample, detokenize the goal token back to coordinate space and compute
        # the Euclidean norm (distance from the origin). If the norm is <= 10, we mark
        # the sample for removal.
        valid_mask_dis = []
        for token in self.trajectories_goals:
            # token is stored as an array with a single element (created in create_gt_data),
            # so extract the scalar value.
            token_val = token[0] if isinstance(token, np.ndarray) else token

            # Detokenize the token to obtain the displacement coordinates.
            dis = detokenize_traj_waypoints(
                np.array([token_val]),
                self.detokenizer,
                self.cfg.bos_token,
                self.cfg.eos_token,
                self.cfg.pad_token
            )
            # Assume that 'dis' is a 2D array, e.g. shape (1, 2) with (x, y) coordinates.
            # Compute the Euclidean distance using the first (and only) row.
            if np.linalg.norm(dis[0]) > 10:
                valid_mask_dis.append(True)
            else:
                valid_mask_dis.append(False)
        valid_mask_dis = np.array(valid_mask_dis)

        # Apply the dis-based filter to all arrays.
        self.trajectories = self.trajectories[valid_mask_dis]
        self.trajectories_gt = self.trajectories_gt[valid_mask_dis]
        self.trajectories_goals = self.trajectories_goals[valid_mask_dis]
        self.trajectories_agent_info = self.trajectories_agent_info[valid_mask_dis]
        self.ego_info = self.ego_info[valid_mask_dis]

    def get_statistics(self):
        """
        Compute and return various statistics of the processed dataset.

        In addition to the usual counts and shapes, this function now calculates
        the number of valid tokens (ignoring the BOS, EOS, and PAD tokens) in each trajectory
        and computes summary statistics of these counts.

        Returns:
            stats (dict): A dictionary containing dataset statistics including:
                - total_samples: Number of samples in the dataset.
                - shapes: Shapes of trajectories, labels, goals, agent info, and ego info arrays.
                - goal_distance_mean: Mean Euclidean distance of the goal coordinates.
                - goal_distance_std: Standard deviation of the goal distances.
                - goal_distance_min: Minimum goal distance.
                - goal_distance_max: Maximum goal distance.
                - valid_tokens_count_mean: Mean number of valid tokens per trajectory.
                - valid_tokens_count_std: Standard deviation of valid tokens per trajectory.
                - valid_tokens_count_min: Minimum valid tokens count in a trajectory.
                - valid_tokens_count_max: Maximum valid tokens count in a trajectory.
        """
        stats = {"total_samples": len(self.trajectories)}

        # Goal distance statistics.
        goal_distances = []
        for token in self.trajectories_goals:
            token_val = token[0] if isinstance(token, np.ndarray) else token
            dis = detokenize_traj_waypoints(
                np.array([token_val]),
                self.detokenizer,
                self.cfg.bos_token,
                self.cfg.eos_token,
                self.cfg.pad_token
            )
            goal_distances.append(np.linalg.norm(dis[0]))
        goal_distances = np.array(goal_distances)
        stats["goal_distance_mean"] = float(np.mean(goal_distances))
        stats["goal_distance_std"] = float(np.std(goal_distances))
        stats["goal_distance_min"] = float(np.min(goal_distances))
        stats["goal_distance_max"] = float(np.max(goal_distances))

        # Valid tokens statistics: count the tokens in each trajectory ignoring BOS, EOS, and PAD tokens.
        invalid_tokens = np.array([self.BOS_token, self.EOS_token, self.PAD_token])
        valid_tokens_counts = []
        for traj in self.trajectories:
            # Count tokens not equal to any invalid token
            count_valid = np.sum(~np.isin(traj, invalid_tokens))
            valid_tokens_counts.append(count_valid)
        valid_tokens_counts = np.array(valid_tokens_counts)
        stats["valid_tokens_count_mean"] = float(np.mean(valid_tokens_counts))
        stats["valid_tokens_count_std"] = float(np.std(valid_tokens_counts))
        stats["valid_tokens_count_min"] = int(np.min(valid_tokens_counts))
        stats["valid_tokens_count_max"] = int(np.max(valid_tokens_counts))
        stats["valid_tokens_count_total"] = int(np.max(valid_tokens_counts)) * stats["total_samples"]

        return stats
