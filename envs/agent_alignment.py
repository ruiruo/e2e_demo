from utils.config import Configuration
from utils.trajectory_utils import detokenize_traj_waypoints, tokenize_traj_waypoints
import numpy as np
import json


class TopologyHistory:
    """
    Encapsulated raw Topology History information.
    """

    def __init__(self, cfg: Configuration, ego_input_ids: list, agent_feature: np.array, ego_speed, start_t):
        self.cfg = cfg
        with open(cfg.detokenizer, "r") as f:
            self.detokenizer = json.load(f)
        self.local2token = np.load(cfg.tokenizer)
        self.max_frame = cfg.max_frame
        self.max_agent = cfg.max_agent
        self.ego_pos_id = ego_input_ids
        self.ego_speed = ego_speed
        self.start_t = start_t
        self.agent = agent_feature
        self.agent_info = []
        self.agent_pos = []
        self._preprocess()

    def _preprocess(self):
        if self.ego_speed == 0.0:
            self.agent_info = []
            return
        self._preprocess_ego()
        self._preprocess_agent()

    def _preprocess_ego(self):
        self.ego_pos = detokenize_traj_waypoints(self.ego_pos_id, self.detokenizer,
                                                   self.cfg.bos_token,
                                                   self.cfg.eos_token,
                                                   self.cfg.pad_token)

        origin = (0.0, 0.0)
        points = [origin] + [(float(x), float(y)) for x, y in self.ego_pos]
        distances = [
            np.hypot(x2 - x1, y2 - y1)
            for (x1, y1), (x2, y2) in zip(points, points[1:])
        ]
        segment_times = [d / self.ego_speed for d in distances]
        self.segment_times = [round(t / 0.2) * 0.2 + self.start_t for t in segment_times]

    def _preprocess_agent(self):
        for time in self.segment_times:
            parser = AgentFeatureParser(self.cfg, time, self.agent)
            agent_info = parser.preprocess()

            T, A, attr = agent_info.shape
            self.agent_raw_pos = agent_info[:, :, 0:2].copy()

            agent_raw_pos_flat = self.agent_raw_pos.reshape(-1, 2)
            tokenized_flat = tokenize_traj_waypoints(agent_raw_pos_flat,
                                                     np.array(self.cfg.x_boundaries),
                                                     np.array(self.cfg.y_boundaries),
                                                     self.local2token)
            agent_tokenized = tokenized_flat.reshape(T, A)

            agent_features = agent_info[:, :, 2:]
            agent_info_final = np.concatenate([agent_tokenized[..., np.newaxis], agent_features], axis=-1)

            # For each agent (second dimension), check if there is at least one token not equal to -1 across all time steps.
            global_valid = np.any(agent_info_final[:, :, 0] != -1, axis=0)
            global_valid_indices = np.where(global_valid)[0]

            # Select globally valid agents and limit the number to self.max_agent.
            if len(global_valid_indices) >= self.max_agent:
                selected_indices = global_valid_indices[:self.max_agent]
            else:
                selected_indices = global_valid_indices

            # --- Extract data for each time step following the global order ---
            # Even if an agent's token is -1 at a specific time step, its corresponding row is retained.
            agent_info_selected = agent_info_final[:, selected_indices, :]
            agent_raw_pos_selected = self.agent_raw_pos[:, selected_indices, :]

            # If the number of selected agents is less than self.max_agent, perform padding.
            n_selected = agent_info_selected.shape[1]
            if n_selected < self.max_agent:
                pad_size = self.max_agent - n_selected
                pad_agent = np.full((T, pad_size, agent_info_selected.shape[-1]), -1, dtype=agent_info_selected.dtype)
                pad_raw_pos = np.full((T, pad_size, 2), -1, dtype=agent_raw_pos_selected.dtype)
                final_agent_info = np.concatenate([agent_info_selected, pad_agent], axis=1)
                final_agent_raw_pos = np.concatenate([agent_raw_pos_selected, pad_raw_pos], axis=1)
            else:
                final_agent_info = agent_info_selected
                final_agent_raw_pos = agent_raw_pos_selected

            invalid_token_mask = final_agent_info[:, :, 0] == -1
            final_agent_info[invalid_token_mask] = self.cfg.pad_token
            final_agent_raw_pos[invalid_token_mask] = -1

            self.agent_info.append(final_agent_info)
            self.agent_pos.append(final_agent_raw_pos)


class AgentFeatureParser:
    """
    Agent feature parser for preprocessing obstacle TopologyHistory information.
    """

    def __init__(self, cfg: Configuration, time, agent: np.array):
        self.cfg = cfg
        self.pos_bias = np.ravel(agent.loc[(agent['aid'] == 0) & (agent['timestamp'] == time),
                                ['x', 'y', 'heading']].to_numpy())
        self.agent = agent[agent['timestamp'] == time].to_numpy()
        self.max_frame = cfg.max_frame

    def preprocess(self) -> np.ndarray:
        """
        Preprocess agent TopologyHistory information and generate an array with format:
            (x, y, heading, v, acc, length, width)
        Original agent_feature format:
            (id, x, y, heading, v, acc, length, width, type, timestamp, virtual)
        """
        # 1. Extract necessary columns and align with ego position bias.
        new_agent_info = self.process_agent_info()

        # 2. simple_deduction
        new_agent_info = self._simulate_future_states(new_agent_info, self.max_frame + 1)

        # 3. Remove agents id col
        new_agent_info = new_agent_info[:, :, 1:]
        # Transpose dimensions from (agent, time, feature) to (time, agent, feature)
        return np.transpose(new_agent_info, (1, 0, 2))

    def process_agent_info(self) -> np.ndarray:
        """
        Extract necessary columns from agent data and align them with the ego position bias.

        Returns:
                np.ndarray: Processed agent information after coordinate adjustments.
        """
        # Extract necessary columns and align with ego position bias.
        agent_info = self.agent[:, 0:8].copy()

        # Identify valid rows where none of the x,y positions equals -300
        mask = ~np.any(agent_info[:, 1:3] == -300, axis=1)

        # Subtract the positional bias from valid rows (only for x and y)
        agent_info[mask, 1:3] -= self.pos_bias[:2]

        # Calculate cosine and sine of the rotation angle
        cos_theta = np.cos(self.pos_bias[2])
        sin_theta = np.sin(self.pos_bias[2])

        # Rotate the x,y coordinates to align with the ego coordinate system
        rotated_xy = agent_info[:, 1:3].copy()
        rotated_xy[:, 0] = agent_info[:, 1] * cos_theta + agent_info[:, 2] * sin_theta
        rotated_xy[:, 1] = -agent_info[:, 1] * sin_theta + agent_info[:, 2] * cos_theta
        agent_info[mask, 1:3] = rotated_xy[mask]

        # Adjust the heading angle by subtracting the positional bias angle and normalize it to [-π, π]
        agent_info[mask, 3] -= self.pos_bias[2]
        agent_info[mask, 3] = (agent_info[mask, 3] + np.pi) % (2 * np.pi) - np.pi

        # Filter out ego's subarray
        mask_id_not_zero = agent_info[:, 0] != 0
        agent_info = agent_info[mask_id_not_zero]

        return agent_info

    @staticmethod
    def _simulate_future_states(agent_info: np.ndarray, steps: int = 10, dt: float = 0.2) -> np.ndarray:
        """
        Simulate future states of agents based on their state at t0.
        State format: [x, y, heading, v, acc, length, width, abs_dis, hit_dis]

        Parameters:
            agent_info (np.ndarray): Array of shape (n_agents, 9) representing the state at t0.
            dt (float): Time interval for each step (seconds), default is 0.2 seconds.
            steps (int): Number of simulation steps (excluding t0), default is 10 steps (i.e., 2 seconds).

        Returns:
            np.ndarray: Array of shape (steps+1, n_agents, 9), where the 0-th timestep is the initial state.
                        For subsequent timesteps, x and y are updated using uniform acceleration motion,
                        while heading, v, acc, and other attributes remain unchanged.
                        If any agent's state (subarray) contains -300, that agent's state is copied without simulation.
        """
        n_agents, n_attr = agent_info.shape
        states = np.full((n_agents, steps, n_attr), -300, dtype=agent_info.dtype)

        # Set the initial state at t0.
        states[:, 0, :] = agent_info.copy()

        # Create a boolean mask: True if the agent (row) does NOT contain -300.
        valid_mask = ~(agent_info == -300).any(axis=1)
        invalid_mask = ~valid_mask  # Agents that contain -300.

        # Process valid agents (simulate future movement).
        if valid_mask.any():
            valid_indices = np.where(valid_mask)[0]
            # Extract parameters for valid agents.
            x0 = agent_info[valid_indices, 1]
            y0 = agent_info[valid_indices, 2]
            heading = agent_info[valid_indices, 3]
            v = agent_info[valid_indices, 4]
            acc = agent_info[valid_indices, 5]
            cos_heading = np.cos(heading)
            sin_heading = np.sin(heading)

            # For each future time step, compute new x and y for valid agents.
            for i in range(1, steps):
                t = i * dt
                disp = v * t + 0.5 * acc * t ** 2
                dx = disp * cos_heading
                dy = disp * sin_heading

                # Copy the original state and update the x and y coordinates.
                new_state = agent_info[valid_indices].copy()
                new_state[:, 1] = x0 + dx
                new_state[:, 2] = y0 + dy

                states[valid_indices, i, :] = new_state

        # Process invalid agents: replicate their original state for all future timesteps.
        if invalid_mask.any():
            invalid_indices = np.where(invalid_mask)[0]
            for i in range(1, steps):
                states[invalid_indices, i, :] = agent_info[invalid_indices]

        return states

