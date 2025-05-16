import os
import json
import pickle
import random
import numpy as np
import pandas as pd
from gymnasium import spaces
from collections import defaultdict
from highway_env.envs.common.abstract import Observation
from highway_env.envs.common.action import Action
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork, LineType
from highway_env.road.lane import PolyLaneFixedWidth
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle, Landmark
from utils.trajectory_utils import detokenize_traj_waypoints, tokenize_traj_waypoints
from envs.agent_alignment import TopologyHistory
from utils.config import Configuration
from envs.utils import quantize_to_step
import matplotlib.pyplot as plt


class ReplayHighwayEnv(AbstractEnv):
    def __init__(self,
                 task_paths: str, env_config: dict, pre_train_config: Configuration):
        self.task_paths = task_paths
        highway_config = {
            # where to pick up your pickles:
            "task_paths": None,
            # visuals:
            "screen_width": 1000,
            "screen_height": 1000,
            # tuning:
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "offscreen_rendering": True,
        }
        self.env_config = env_config
        self.pre_train_config = pre_train_config
        self.x_boundaries = self.pre_train_config.x_boundaries
        self.y_boundaries = self.pre_train_config.y_boundaries
        self.x_max = self.x_boundaries[-1]
        self.y_max = self.y_boundaries[-1]
        self.x_min = self.x_boundaries[0]
        self.y_min = self.x_boundaries[0]
        self.local2token = np.load(self.pre_train_config.tokenizer)
        with open(self.pre_train_config.detokenizer, "r") as f:
            self.token2local = json.load(f)
        self.obs_h = len(self.x_boundaries) - 1  # number of cells in x
        self.obs_w = len(self.y_boundaries) - 1  # number of cells in y
        self.agent_feature = None
        self.vector_graph_feature = None
        self.ego_feature = None
        self.ego = None
        self.ego_input_ids = []
        self.ego_position_raw = []
        self.all_agents = {}
        self.t = 0
        self.segment_times = 0
        super().__init__(highway_config, render_mode="rgb_array")
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=255, shape=[635])

    def reset(self, *, seed=None, options=None):
        # 1) pick & load a new episode
        self._load_new()
        osb, _ = self._update(np.array([0, 0]))
        osb, _ = self._update(np.array([0, 0]))
        osb, _ = self._update(np.array([0, 0]))
        info = {}
        self.t = 0
        return osb, info

    def _load_new(self):
        self.all_agents.clear()
        self._read_file()
        self._make_road()
        self.ego_position_raw = []

    def visualize(self):
        img = self.render()
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.savefig(self.env_config["test_img"] + str(self.t) + ".jpg",
                    format='jpg', bbox_inches='tight', pad_inches=0)

    def _update(self, ego_position):
        # tokenize position
        tokenized_positions = tokenize_traj_waypoints(np.array([ego_position]),
                                                      self.x_boundaries, self.y_boundaries,
                                                      self.local2token)
        tokenized_positions = [int(i) for i in tokenized_positions]
        self.ego_input_ids.extend(tokenized_positions)

        # update background, create new agent_info, segment_times
        # todo: TopologyHistory issue
        agent_data = TopologyHistory(self.pre_train_config, self.ego_input_ids[-1:],
                                 self.agent_feature, self.ego.speed, self.t)
        segment_times = agent_data.segment_times
        # update time
        self.t += segment_times[0]
        # move agent
        self._apply_background()
        # move ego
        self._move_ego(ego_position)
        # create flatten obs
        obs = self._get_obs(agent_data)
        flattened_obs = np.concatenate([np.array(obs["input_ids"]).flatten(),
                                        np.array(obs["ego_info"]).flatten(),
                                        np.array(obs["agent_info"]).flatten(),
                                        np.array(obs["goal"]).flatten(),
                                        ])
        if self.t >= 0.5 or self.t == 0:
            self.visualize()
        return flattened_obs, segment_times[0]

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = None, None, None, None, {"image": self.render()}
        after_detokenize = detokenize_traj_waypoints(
            np.array([action]),
            self.token2local,
            self.pre_train_config.bos_token,
            self.pre_train_config.eos_token,
            self.pre_train_config.pad_token,
        )
        obs, segment_times = self._update(after_detokenize[0])
        self.segment_times = segment_times
        if round(self.t, 0) >= 3 or self.segment_times >= 2:
            terminated = True
        reward = self._reward(action)
        return obs, reward, terminated, truncated, info

    def _reward(self, action: Action) -> dict[str, float]:
        reward = 0
        if self.segment_times >= 0.4:
            reward -= 1
        return 0

    def _move_ego(self, ego_position) -> None:
        """
        Hard-warp the ego to (new_x, new_y) in world coords, then
        refresh every other vehicle’s relative position.  Use with care:
        • No dynamic constraints
        • No collision check
        """
        self.ego.position += ego_position
        self.ego_position_raw.append(self.ego.position.tolist())

    def _get_obs(self, agent_info):
        input_ids = np.array([[0,0]])
        input_ids = tokenize_traj_waypoints(input_ids, self.x_boundaries, self.y_boundaries, self.local2token)
        # heading, speed, acc
        ego_info = [self.ego.heading, self.ego.speed, self.ego.action["acceleration"]]
        goal_raw = self.ego_goal_raw - self.ego.position
        goal = self._clamp_goal(goal_raw)

        goal = tokenize_traj_waypoints(np.array([goal]), self.x_boundaries, self.y_boundaries, self.local2token)

        return {
            "goal": goal,
            "input_ids": input_ids,
            "ego_info": ego_info,
            "agent_info": agent_info.agent_info[0],
        }

    def _apply_background(self):
        """
        Overwrite all other vehicles’ states (position, heading, speed, acc)
        from self.agent_feature at the current timestamp self.t.
        Assumes:
          - self.agent_feature is a DataFrame with columns
            ['aid','x','y','heading','v','acc',…,'timestamp']
          - self.all_agents maps aid→Vehicle or Obstacle (ego is aid=0)
        """
        # round self.t to match the timestamp column’s precision
        # select rows for this time
        quantized_t = quantize_to_step(self.t, step=0.2, method="round")
        df_t = self.agent_feature[self.agent_feature['timestamp'] == quantized_t]

        for _, row in df_t.iterrows():
            aid = row['aid']
            # skip ego (we handle ego by stepping the env normally)
            # if aid == 0:
            #     continue
            veh = self.all_agents.get(aid)
            if veh is None:
                continue

            # update position
            veh.position = np.array([row['x'], row['y']])

            # update kinematic state
            veh.heading = float(row['heading'])
            veh.speed = float(row['v'])
            # if your Vehicle/Obstacle class has an acceleration attr:
            if hasattr(veh, "action"):
                veh.action["acceleration"] = float(row['acc'])
            veh.t += quantized_t

    def _reward(self, action):
        return 0

    def _read_file(self):
        task = random.sample(os.listdir(self.task_paths), 1)[0]
        pkl = random.sample(os.listdir(os.path.join(self.task_paths, task)), 1)[0]
        with open(os.path.join(self.task_paths, task, pkl), 'rb') as f:
            self.data_info = pickle.load(f)
            self.ego_feature = self.data_info['ego_history_feature'][:, 1:][::-1]
            pos_bias = self.ego_feature[0][1:3].copy()
            self.ego_feature[:, 1:3] -= pos_bias
            end_t = round(self.ego_feature[0, 6].copy(), 1)
            self.ego_feature[:, 6] = np.round(end_t - self.ego_feature[:, 6], 1)
            self.ego_feature = pd.DataFrame(self.ego_feature[:, 1:],
                                            columns=['x', 'y', 'heading', 'v', 'acc', 'timestamp'])
            self.ego_feature = self.ego_feature[(self.ego_feature.x > self.x_min) &
                                                (self.ego_feature.x < self.x_max) &
                                                (self.ego_feature.y > self.y_min) &
                                                (self.ego_feature.y < self.y_max)]

            for idx in range(len(self.data_info['agent_feature'])):
                self.data_info['agent_feature'][idx] = self.data_info['agent_feature'][idx][::-1]
            agent_feature = pd.DataFrame(self.data_info['agent_feature'].reshape([-1, 8]),
                                         columns=["_", "aid", "x", "y", "heading", "v", "acc", "timestamp"])

            agent_attribute_feature = pd.DataFrame(self.data_info['agent_attribute_feature'],
                                                   columns=["aid", "length", "width", "type",
                                                            "virtual", "key_smooth_future"])

            agent_feature = pd.merge(agent_feature, agent_attribute_feature, on="aid")
            agent_feature[['x', 'y']] = agent_feature[['x', 'y']].values - pos_bias
            agent_feature['timestamp'] = np.round(end_t - agent_feature['timestamp'], 1)
            self.agent_feature = agent_feature[["aid", "x", "y",
                                                "heading", "v", "acc", "length", "width", "type", "timestamp",
                                                "virtual"]]
            self.agent_feature = self.agent_feature[self.agent_feature.aid != -300].reset_index(drop=True)

            self.vector_graph_feature = self.data_info['vector_graph_feature'][::-1]
            self.vector_graph_feature[:, :, 0:2] -= pos_bias
            self.vector_graph_feature[:, :, 2:4] -= pos_bias
            ego_goal = self.ego_feature.sort_values('timestamp').iloc[-1]
            self.ego_goal = np.array([[ego_goal["x"], ego_goal["y"]]])
            self.ego_goal_raw = np.array([ego_goal["x"], ego_goal["y"]])

    def _make_road(self):
        """
        Build self.road from self.vector_graph_feature,
        which is now shape (n_vectors, 9):
          [start_x, start_y, end_x, end_y,
           road_id, width, left_attr, right_attr, speed_limit]
        """
        all_vg = self.vector_graph_feature[:, :, [4, 0, 1, 2, 3, 5, 6, 7, 8]]
        network = RoadNetwork()
        for vg in all_vg:
            vg = vg[vg[:, 0] != -300]
            road_groups = defaultdict(list)
            for road_id, sx, sy, ex, ey, width, left, right, speed in vg:
                road_groups[int(road_id)].append(np.array([sx, sy, ex, ey, width, speed]))
            road_groups = {k: np.array(v) for k, v in road_groups.items()}
            # 1) build map
            for road_id, features in road_groups.items():
                lane_points = []
                for sx, sy, ex, ey, width, speed in features:
                    lane_points.append((float(sx), float(sy)))
                # don’t forget the very last endpoint
                lane_points.append((float(features[-1, 2]), float(features[-1, 3])))

                w = float(features[0, 4])  # constant width
                v = float(features[0, 5])  # speed limit

                lane = PolyLaneFixedWidth(
                    lane_points,
                    width=w,
                    speed_limit=v,
                    line_types=[LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE],
                )
                # use the same id for from/to so it’s one continuous road
                network.add_lane(str(road_id), str(road_id), lane)
        agent_feature = self.agent_feature[self.agent_feature["timestamp"] == 0]
        agent_groups = agent_feature.groupby("aid")
        for aid, df in agent_groups:
            agent_start_loc = df.sort_values("timestamp").iloc[0]
            position = [float(agent_start_loc["x"]), float(agent_start_loc["y"])]
            if aid == 0:
                speed = agent_start_loc["v"]
                if speed == 0:
                    speed = 5
                agent = Vehicle(self.road,
                                position=position,
                                heading=float(agent_start_loc["heading"]),
                                speed=speed)
                agent.color = (255, 255, 255)
                agent.LENGTH = float(agent_start_loc["length"])
                agent.WIDTH = float(agent_start_loc["width"])
                agent.t = agent_start_loc["timestamp"]
                self.ego = agent
            else:
                if agent_start_loc["type"] == 791621440:
                    agent = Vehicle(self.road, position=position)
                    agent.color = (120, 50, 9)
                else:
                    agent = Obstacle(self.road, position=position)
                    agent.color = (0, 0, 0)
                agent.LENGTH = float(agent_start_loc["length"])
                agent.WIDTH = float(agent_start_loc["width"])
                agent.t = agent_start_loc["timestamp"]
                self.all_agents[aid] = agent
        self.road = Road(
            network,
            vehicles=[self.ego] + list(self.all_agents.values()),
            record_history=False,
        )
        goal_marker = Landmark(self.road, position=self.ego_goal_raw.tolist())
        goal_marker.color = (0, 255, 0)  # bright green, for example
        goal_marker.LENGTH = 3.0  # make it small
        goal_marker.WIDTH = 3.0
        self.road.objects.append(goal_marker)

    def _clamp_goal(self, goal_xy: np.ndarray,
                    eps: float = 1e-6) -> np.ndarray:
        """
        Force `goal_xy` to lie inside the 2-D boundary box defined by
        `x_boundaries` and `y_boundaries`.

        Parameters
        ----------
        goal_xy : np.ndarray
            Shape (2,) – [x, y] coordinates of the goal.
        x_boundaries : np.ndarray
            Sorted 1-D array of x-bin edges (length ≥ 2).
        y_boundaries : np.ndarray
            Sorted 1-D array of y-bin edges (length ≥ 2).
        eps : float
            Tiny margin so the point never lands *exactly* on the edge,
            which can confuse discretisers.  Default 1 × 10⁻⁶.

        Returns
        -------
        np.ndarray
            Clamped goal coordinates, same shape as `goal_xy`.
        """
        x_min, x_max = self.x_boundaries[0], self.x_boundaries[-1]
        y_min, y_max = self.y_boundaries[0], self.y_boundaries[-1]

        goal_clamped = np.asarray(goal_xy, dtype=float).copy()
        goal_clamped[0] = np.clip(goal_clamped[0], x_min + eps, x_max - eps)
        goal_clamped[1] = np.clip(goal_clamped[1], y_min + eps, y_max - eps)
        return goal_clamped
