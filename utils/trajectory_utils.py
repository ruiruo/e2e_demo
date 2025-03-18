from shapely.geometry import LineString
from shapely.measurement import hausdorff_distance
from typing import List

from sympy import limit

from utils.common import get_json_content
# from utils.pose_utils import CustomizePose
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch


def create_sample(ego_tokens, agent_info, bos_token, eos_token, pad_token, target_seq_len):
    """
    Create an autoregressive training sample for a forward-only model (similar to GPT).

    Parameters:
        ego_tokens (List[int]): The tokenized ego trajectory.
        agent_info (Any): Background agent information (e.g., topological descriptions for each time step).
        bos_token (int): The beginning-of-sequence token.
        eos_token (int): The end-of-sequence token.
        pad_token (int): The padding token.
        target_seq_len (int): The desired sequence length after adding BOS, EOS, and PAD tokens.

    Returns:
        input_ids (List[int]): The input sequence for the autoregressive model (tokens up to the last token).
        labels (List[int]): The target tokens (the input sequence shifted by one position).
        agent_info (Any): The background agent information, unchanged.
    """
    # Build the ego trajectory with BOS and EOS tokens.
    trajectory = np.concatenate([np.array([bos_token]), ego_tokens, np.array([eos_token])])
    # Calculate the number of PAD tokens needed to reach the target sequence length.
    n_pad = max(0, target_seq_len - len(trajectory))
    trajectory = np.concatenate([trajectory, np.array([pad_token] * n_pad)])
    # For autoregressive training, the model inputs are the tokens up to the last token,
    # and the labels are the tokens shifted one position to the left.
    input_ids = trajectory[:-1]
    labels = trajectory[1:]
    return input_ids, labels, agent_info


def parallel_find_bin(points, m_boundaries, n_boundaries):
    """
    Vectorized bin-finding for a batch of (x, y) coordinates.

    Args:
        points (np.ndarray): shape (n, 2). Each row is (x, y).
        m_boundaries (np.ndarray): shape (M+1,), sorted boundaries along the m-axis (x-axis).
        n_boundaries (np.ndarray): shape (N+1,), sorted boundaries along the n-axis (y-axis).

    Returns:
        i (np.ndarray): shape (n,). Bin indices along the m-axis for each point.
        j (np.ndarray): shape (n,). Bin indices along the n-axis for each point.

    Explanation:
      - For each point p_k = (x_k, y_k), we find indices i_k, j_k such that:
          m_boundaries[i_k] <= x_k < m_boundaries[i_k + 1]
          n_boundaries[j_k] <= y_k < n_boundaries[j_k + 1]
      - If x_k or y_k is out of the valid range, we set i_k = j_k = -1.
    """
    # Extract x and y columns
    x_vals = points[:, 0]
    y_vals = points[:, 1]

    # Search for bin indices along m-axis (x-axis)
    i = np.searchsorted(m_boundaries, x_vals, side='right') - 1
    # Search for bin indices along n-axis (y-axis)
    j = np.searchsorted(n_boundaries, y_vals, side='right') - 1

    # Create a mask for out-of-range points
    out_of_range_mask = (
            (i < 0) | (i >= len(m_boundaries) - 1) |
            (j < 0) | (j >= len(n_boundaries) - 1)
    )

    # Assign -1 for any out-of-range points
    i[out_of_range_mask] = -1
    j[out_of_range_mask] = -1

    return i, j


def tokenize_traj_waypoints(waypoints, m_boundaries, n_boundaries, local2token):
    """
    Tokenize multiple (x, y) waypoints at once using a 2D local2token array.

    Args:
        waypoints (np.ndarray): shape (n, 2), each row is (x, y).
        m_boundaries (np.ndarray): sorted 1D array of x-axis boundaries, length M+1.
        n_boundaries (np.ndarray): sorted 1D array of y-axis boundaries, length N+1.
        local2token (np.ndarray): shape (M, N) integer array. local2token[i, j] = token_id.

    Returns:
        np.ndarray of shape (n,), each entry is the token_id or -1 if not found or out-of-range.
    """
    # 1) Find bin indices (i, j) for each waypoint
    i_bins, j_bins = parallel_find_bin(waypoints, m_boundaries, n_boundaries)
    # 2) Prepare an output array of token IDs (initialized to -1)
    token_ids = np.full(len(waypoints), -1, dtype=int)

    # 3) For valid bins, map (i, j) to local2token[i, j]
    valid_mask = (i_bins != -1) & (j_bins != -1)
    token_ids[valid_mask] = local2token[i_bins[valid_mask], j_bins[valid_mask]]

    return token_ids


def detokenize_traj_waypoint(token_id, token_center):
    """
    Given a token ID, return the center (cx, cy) of that grid cell.

    Parameters:
      token_id     : The token ID (int).
      token_center : A dict mapping token_id -> (center_x, center_y).

    Returns:
      (center_x, center_y) if token_id is found, else None.
    """
    return token_center.get(token_id, None)


def plot_trajectory_with_time(ego_info: np.ndarray):
    """
    Plots the trajectory using the ego_info array and annotates each point with its time information.

    Each row in ego_info is expected to be in the format (x, y, heading, v, acc).
    The time for the first point (index 0) is -50 and for the last point (index -1) is 0.

    Parameters:
        ego_info (np.ndarray): Array containing the ego vehicle data.
    """
    # Extract x and y coordinates.
    x = ego_info[:, 0]
    y = ego_info[:, 1]

    # Calculate the time for each point linearly between -50 and 0.
    num_points = len(x)
    times = np.linspace(-50, 0, num_points)

    # Create the plot.
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', label="Trajectory")

    # Annotate each point with its time value.
    for xi, yi, t in zip(x, y, times):
        plt.annotate(f"{t:.1f}", (xi, yi), textcoords="offset points", xytext=(5, 5))

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Ego Vehicle Trajectory with Time Annotations")
    plt.legend()
    plt.grid(True)
    plt.show()


class TopologyHistory:
    """
    Encapsulated raw Topology History information.
    """

    def __init__(self, frame_id: int, feature: dict, max_frame: int=10):
        # ego_info = (t, 5), (t_-50, t_0), (x, y, heading, v, acc)
        # agent_info = (t, agent, 10), (t_-50, t_0), (id, x, y, heading, v, acc, length, width, abs_dis, hit_dis)
        self.frame_id = frame_id
        self.max_frame = max_frame
        self.info = {}
        self._preprocess(feature)

    def _preprocess(self, feature):
        self._preprocess_ego(feature.get('ego_history_feature'))
        self._preprocess_agent(feature)
        self.info["goal_info"] = self.info["ego_info"][-1][0:2]
        self._cut()

    def _cut(self):
        self.info["ego_info"] = self.info["ego_info"][-self.max_frame - 1:-1, :]
        self.info["agent_info"] = self.info["agent_info"][-self.max_frame - 1:-1, :]

    def _preprocess_agent(self, feature):
        parser = AgentFeatureParser(feature)
        self.info["agent_info"] = parser.preprocess()

    def _preprocess_ego(self, ego_history):
        """
        Preprocess the TopologyHistory information by converting ego_history data into ego_info,
        formatted as (x, y, heading, v, acc).
        """
        # Extract the required columns.
        # TODO: The frame interval is unstable, varying between 100ms and 200ms.
        # Consider whether to use frame extraction or direct cropping.
        ego_info = ego_history[:, 2:7].copy()
        # Align the starting position of the ego vehicle to (0, 0).
        mask = ~np.any(ego_info[:, :2] == -300, axis=1)
        ego_info[mask, :2] -= ego_info[0, :2]

        self.info["ego_info"] = ego_info


class AgentFeatureParser:
    """
    Agent feature parser for preprocessing obstacle TopologyHistory information.

    The input feature dictionary must include:
      - 'ego_history_feature': Historical trajectory for ego vehicle (array)
      - 'agent_feature': Spatiotemporal features of agents (array), includes ego info (id: 0)
      - 'agent_attribute_feature': Attribute features of agents (array)
    """

    def __init__(self, feature: dict):
        required_keys = ['ego_history_feature', 'agent_feature', 'agent_attribute_feature']
        for key in required_keys:
            if key not in feature:
                raise KeyError(f"Missing required key '{key}' in feature dictionary.")
        # Use the first frame of ego history as the position bias.
        self.pos_bias = feature['ego_history_feature'][0, 2:4]
        self.agent = feature['agent_feature']
        self.agent_attribute = feature['agent_attribute_feature']

    def preprocess(self) -> np.ndarray:
        """
        Preprocess agent TopologyHistory information and generate an array with format:
            (id, x, y, heading, v, acc, length, width, abs_dis, hit_dis)
        Original agent_feature format:
            (s_id, id, x, y, heading, v, acc, timestamp)
        """
        # 1. Extract necessary columns and align with ego position bias.
        agent_info = self.agent[:, :, 1:7].copy()
        agent_attr = self.agent_attribute[:, :3]
        mask = ~np.any(agent_info[:, :, 1:3] == -300, axis=2)
        agent_info[mask, 1:3] -= self.pos_bias

        # 2. Concatenate agent attributes to agent_info.
        new_agent_info = self._concatenate_agent_attributes(agent_info, agent_attr)

        # 3. Append absolute distance from ego to each agent.
        new_agent_info = self._append_absolute_distance(new_agent_info)

        # 4. Append minimum polygon (bounding box) distance between ego and each agent.
        new_agent_info = self._append_min_polygon_distance(new_agent_info)

        # 5. Remove agents with id == 0 and id col
        mask_id_not_zero = new_agent_info[:, 0, 0] != 0
        new_agent_info = new_agent_info[mask_id_not_zero]
        new_agent_info = new_agent_info[:,:,1:]

        # Transpose dimensions from (agent, time, feature) to (time, agent, feature)
        return np.transpose(new_agent_info, (1, 0, 2))

    @staticmethod
    def _concatenate_agent_attributes(agent_info: np.ndarray, agent_attr: np.ndarray) -> np.ndarray:
        """
        Concatenate extra attributes from agent_attr to agent_info.

        Build a dictionary mapping agent id to its attribute data, then iterate over agent_info to append the corresponding attributes.
        """
        id_to_data = {row[0]: row[1:] for row in agent_attr}
        new_last_dim = agent_info.shape[2] + agent_attr.shape[1] - 1
        new_agent_info = np.full((agent_info.shape[0], agent_info.shape[1], new_last_dim), -300., dtype=float)
        for i in range(agent_info.shape[0]):
            for j in range(agent_info.shape[1]):
                current_id = agent_info[i, j, 0]
                if current_id == -300:
                    continue
                if current_id not in id_to_data:
                    raise ValueError(f"Agent attribute not found for id: {current_id}")
                extra_data = id_to_data[current_id]
                new_agent_info[i, j] = np.concatenate([agent_info[i, j], extra_data])
        return new_agent_info

    @staticmethod
    def _append_absolute_distance(data: np.ndarray) -> np.ndarray:
        """
        Compute the absolute distance between each agent and ego, and append it as a new feature.
        """
        ids = data[:, 0, 0]
        ego_idx_arr = np.where(ids == 0)[0]
        if ego_idx_arr.size == 0:
            raise ValueError("Ego not found in agent data")
        idx_ego = ego_idx_arr[0]

        # Get ego x, y positions over time.
        ego_x = data[idx_ego, :, 1]
        ego_y = data[idx_ego, :, 2]
        # Calculate Euclidean distances for each agent relative to ego at each time step.
        distances = np.sqrt((data[:, :, 1] - ego_x[None, :]) ** 2 + (data[:, :, 2] - ego_y[None, :]) ** 2)
        distances = distances[:, :, None]

        # Set distance to -300 for invalid positions.
        invalid_mask = np.any(data[:, :, 1:3] == -300, axis=2)
        distances[invalid_mask] = -300

        return np.concatenate([data, distances], axis=2)

    def _append_min_polygon_distance(self, data: np.ndarray) -> np.ndarray:
        """
        Compute the minimum distance between the bounding boxes of each agent and the ego vehicle,
        and append this value as a new feature.
        """
        ids = data[:, 0, 0]
        ego_idx_arr = np.where(ids == 0)[0]
        if ego_idx_arr.size == 0:
            raise ValueError("Ego information not found for bounding box distance computation")
        idx_ego = ego_idx_arr[0]

        num_agents, T, _ = data.shape

        # Precompute ego bounding boxes for each valid time step to avoid redundant calculations.
        ego_bboxes = [None] * T
        for t in range(T):
            if data[idx_ego, t, 1] == -300 or data[idx_ego, t, 2] == -300:
                ego_bboxes[t] = None
            else:
                ego_x, ego_y = data[idx_ego, t, 1], data[idx_ego, t, 2]
                ego_theta, ego_L, ego_W = data[idx_ego, t, 3], data[idx_ego, t, 6], data[idx_ego, t, 7]
                ego_bboxes[t] = self._get_bbox(ego_x, ego_y, ego_theta, ego_L, ego_W, is_ego=True)

        min_dists = np.full((num_agents, T), -300., dtype=float)
        for obj in range(num_agents):
            for t in range(T):
                if ego_bboxes[t] is None or data[obj, t, 1] == -300 or data[obj, t, 2] == -300:
                    continue

                obj_x, obj_y = data[obj, t, 1], data[obj, t, 2]
                obj_theta, obj_L, obj_W = data[obj, t, 3], data[obj, t, 6], data[obj, t, 7]
                # Compute object's bounding box.
                obj_bbox = self._get_bbox(obj_x, obj_y, obj_theta, obj_L, obj_W)
                # Compute minimum distance between ego and object bounding boxes.
                d = self._min_distance_between_polygons(ego_bboxes[t], obj_bbox)
                min_dists[obj, t] = d

        min_dists = min_dists[:, :, None]
        return np.concatenate([data, min_dists], axis=2)

    @staticmethod
    def _get_bbox(x: float, y: float, theta: float, L: float, W: float, is_ego: bool = False) -> np.ndarray:
        """
        Compute the four corners of a rotated rectangle (bounding box) given center (x, y), orientation theta (radians),
        length L and width W.

        If is_ego is True, use 3/4 of L for the front and 1/4 for the back. Otherwise, use L/2 for both front and back.
        Returns an array of shape (4, 2) with the coordinates of the corners.
        """
        if is_ego:
            front = L * 3 / 4
            back = -L * 1 / 4
        else:
            front = L / 2
            back = -L / 2

        corners = np.array([
            [front, W / 2],
            [front, -W / 2],
            [back, -W / 2],
            [back, W / 2]
        ])
        # Create rotation matrix for angle theta.
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        rotated = corners @ R.T
        return rotated + np.array([x, y])

    @staticmethod
    def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the shortest distance from a point p to the line segment ab.
        """
        v = b - a
        w = p - a
        c1 = np.dot(w, v)
        if c1 <= 0:
            return np.linalg.norm(p - a)
        c2 = np.dot(v, v)
        if c2 <= c1:
            return np.linalg.norm(p - b)
        b_ratio = c1 / c2
        pb = a + b_ratio * v
        return np.linalg.norm(p - pb)

    @classmethod
    def _min_distance_between_polygons(cls, poly1: np.ndarray, poly2: np.ndarray) -> float:
        """
        Compute the minimum distance between two convex polygons (represented by their corner points).
        Iterate over each point of poly1 and compute its distance to each edge of poly2, and vice versa.
        If the polygons intersect, the minimum distance is theoretically 0.
        """
        min_dist = float('inf')
        n1, n2 = len(poly1), len(poly2)
        # Check each point of poly1 against each edge of poly2.
        for i in range(n2):
            a = poly2[i]
            b = poly2[(i + 1) % n2]
            for p in poly1:
                d = cls._point_to_segment_distance(p, a, b)
                if d < min_dist:
                    min_dist = d
        # Check each point of poly2 against each edge of poly1.
        for i in range(n1):
            a = poly1[i]
            b = poly1[(i + 1) % n1]
            for p in poly2:
                d = cls._point_to_segment_distance(p, a, b)
                if d < min_dist:
                    min_dist = d
        return min_dist


class TrajectoryInfoParser:
    def __init__(self, task_index, task_path, max_frame):
        self.task_index = task_index
        self.task_path = task_path
        self.total_trajectory = 0
        self.max_frame = 10
        self.trajectories = []
        self._get_data()

    def _get_data(self):
        for each in os.listdir(self.task_path):
            with open(os.path.join(self.task_path, each), 'rb') as f:
                data = pickle.load(f)
                data = self._filter_useful_slice(data)
                if not data:
                    continue

                # TODO: case_id细分
                case_id = int(each.replace(".pkl", ""))
                ego_history = data['ego_history_feature']
                agent_feature = data['agent_feature']
                slice_length = ego_history.shape[0]

                for start_idx in range(slice_length - self.max_frame):
                    end_idx = start_idx + self.max_frame + 1
                    ego_window = ego_history[start_idx:end_idx]
                    agent_window = agent_feature[:, start_idx:end_idx, :]
                    self.trajectories.append(TopologyHistory(case_id, {
                        "ego_history_feature": ego_window,
                        "agent_feature": agent_window,
                        "agent_attribute_feature": data["agent_attribute_feature"]},
                         self.max_frame))
                    self.total_trajectory += 1

    def _filter_useful_slice(self, data: dict) -> dict:
        """
        Filter `ego_history_feature` (self-vehicle's historical trajectory) and `agent_feature` (features of other traffic participants)
        to obtain and process useful time segments, including:
        1. Extract data segments with large timestamp intervals (> 0.18s).
        2. Filter out segments with short intervals, ensuring that the segment length is not less than self.max_frame.
        3. Reverse the time order so that the most recent data is at the front.
        4. Filter out small values in self-vehicle's x, y, velocity (v) as well as heading and acceleration (acc).
        5. Re-select continuous segments of self movement, ensuring that the segment length is not less than self.max_frame,
           and apply coordinate offset processing.

        Returns:
        - dict: The processed `data` dictionary with updated `ego_history_feature` and `agent_feature`.
        - {} if no valid data is found.
        """
        ego_history = data['ego_history_feature']
        slice_start = None
        slice_end = 0
        for i in range(ego_history.shape[0] - 1, 0, -1):
            time_diff = ego_history[i, 7] - ego_history[i - 1, 7]
            if slice_start is None:
                if time_diff > 0.18:
                    slice_start = min(i + 1, ego_history.shape[0] - 1)
            elif time_diff < 0.18:
                if slice_start - i >= self.max_frame:
                    slice_end = min(i + 1, ego_history.shape[0] - 1)
                    break
                slice_start = None

        if slice_start is None:
            print("Task{self.task_index} no segment found with consecutive timestamp intervals greater than 0.2s")
            return {}
        slice_length = slice_start - slice_end
        if slice_length < self.max_frame:
            print(f"Task{self.task_index} subarray satisfying the timestamp interval condition has fewer than {self.max_frame} elements")
            return {}

        ego_history = ego_history[slice_end : slice_start][::-1]

        new_agent_feature = np.full((data['agent_feature'].shape[0], slice_length, data['agent_feature'].shape[2]),
                                    fill_value=-300.)
        for idx in range(len(data['agent_feature'])):
            new_agent_feature[idx] = data['agent_feature'][idx][slice_end: slice_start][::-1]
        data['agent_feature'] = new_agent_feature

        # Threshold for [x, y, v]: set values with absolute value < 1e-3 to 0.
        for col in [2, 3, 5]:
            ego_history[:, col] = np.where(np.abs(ego_history[:, col]) < 1e-3, 0, ego_history[:, col])
            data['agent_feature'][:, :, col] = np.where(np.abs(data['agent_feature'][:, :, col]) < 1e-3, 0, data['agent_feature'][:, :, col])
        # Threshold for [heading, acc]: set values with absolute value < 1e-3 to 0.
        for col in [4, 6]:
            ego_history[:, col] = np.where(np.abs(ego_history[:, col]) < 1e-5, 0, ego_history[:, col])
            data['agent_feature'][:, :, col] = np.where(np.abs(data['agent_feature'][:, :, col]) < 1e-5, 0, data['agent_feature'][:, :, col])

        slice_start = None
        slice_end = ego_history.shape[0]
        for i in range(ego_history.shape[0] - 1):
            x_diff = ego_history[i + 1, 2] - ego_history[i, 2]
            if slice_start is None:
                if x_diff > 1e-3:
                    slice_start = max(i, 0)
            elif x_diff < 1e-3:
                if slice_start - i >= self.max_frame:
                    slice_end = i
                    break
                slice_start = None

        if slice_start is None:
            print("Task{self.task_index} no segment of continuous self movement found")
            return {}
        slice_length = slice_end - slice_start
        if slice_length < self.max_frame:
            print(f"Task{self.task_index} subarray satisfying self movement condition has fewer than {self.max_frame} elements")
            return {}

        ego_history = ego_history[slice_start : slice_end]
        pos_bias = ego_history[0, 2:4].copy()
        ego_history[:, 2:4] -= pos_bias

        new_agent_feature = np.full((data['agent_feature'].shape[0], slice_length, data['agent_feature'].shape[2]), fill_value=-300.)
        for idx in range(len(data['agent_feature'])):
            new_agent_feature[idx] = data['agent_feature'][idx][slice_start : slice_end]
        new_agent_feature[:, :, 2:4] -= pos_bias

        data['ego_history_feature'] = ego_history
        data['agent_feature'] = new_agent_feature

        return data