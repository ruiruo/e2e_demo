import os
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
from utils.trajectory_utils import parallel_find_bin, tokenize_traj_waypoints
from utils.trajectory_utils import TopologyHistory
from utils.config import Configuration


class ReplayHighwayEnv(AbstractEnv):
    def __init__(self,
                 task_paths: str, configs: dict, pre_train_config: Configuration):
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
            "offscreen_rendering": False,
        }
        self.pre_train_config = pre_train_config
        self.x_boundaries = self.pre_train_config.x_boundaries
        self.y_boundaries = self.pre_train_config.y_boundaries
        self.x_max = configs["x_max"]
        self.y_max = configs["y_max"]
        self.x_min = configs["x_min"]
        self.y_min = configs["y_min"]
        self.local2token = np.load(self.pre_train_config.tokenizer)
        self.obs_h = len(self.x_boundaries) - 1  # number of cells in x
        self.obs_w = len(self.y_boundaries) - 1  # number of cells in y
        self.agent_feature = None
        self.vector_graph_feature = None
        self.ego_feature = None
        self.ego = None
        self.ego_input_ids = []
        self.all_agents = {}
        self.t = 0
        super().__init__(highway_config, render_mode="rgb_array")
        self.action_space = spaces.Discrete(5)

    def reset(self, *, seed=None, options=None):
        # 1) pick & load a new episode
        self.all_agents.clear()
        self._read_file()
        self._make_road()
        _ = self.get_current_obs_view()
        self.ego_input_ids = self.ego_input_ids + self.ego_input_ids + self.ego_input_ids
        # 2) do HighwayEnv’s normal reset (builds self.road & self.vehicles)
        obs = None
        info = {"image": self.render()}

        # 3) place all other cars at frame 0
        self.t = 0
        # self._apply_frame(self.t)

        return obs, info

    def get_current_obs_view(self):
        ego_heading = self.ego.heading
        ego_speed = self.ego.speed
        rel_xy = []
        features = []
        ids = []
        ego_position = np.array([self.ego.position])
        if not self.ego_input_ids:
            self.ego_input_ids.append(int(tokenize_traj_waypoints(ego_position,
                                                                  self.x_boundaries, self.y_boundaries,
                                                                  self.local2token)[0]))
            self.ego_input_ids.append(int(tokenize_traj_waypoints(ego_position,
                                                                  self.x_boundaries, self.y_boundaries,
                                                                  self.local2token)[0]))
            self.ego_input_ids.append(int(tokenize_traj_waypoints(ego_position,
                                                                  self.x_boundaries, self.y_boundaries,
                                                                  self.local2token)[0]))
        else:
            self.ego_input_ids.append(int(tokenize_traj_waypoints(ego_position,
                                                                  self.x_boundaries, self.y_boundaries,
                                                                  self.local2token)[0]))
        # agent_info = TopologyHistory(self.pre_train_config, 0,
        #                              feature={
        #                                  "ego_history_feature": self.ego_input_ids[-3:],
        #                                  "agent_feature": agent_window,
        #                                  "agent_attribute_feature": data["agent_attribute_feature"]
        #                              },
        #                              )

        #
        # for aid, veh in self.all_agents.items():
        #     dx = float(veh.position[0] - ego_x)
        #     dy = float(veh.position[1] - ego_y)
        #
        #     # 2. visibility cull ------------------------------------------------
        #     if not (self.x_min <= dx < self.x_max and
        #             self.y_min <= dy < self.y_max):
        #         continue
        #
        #     rel_xy.append([dx, dy])
        #     features.append([
        #         veh.speed,
        #         getattr(veh, "acceleration", 0.0),
        #         veh.heading,
        #         veh.LENGTH, veh.WIDTH
        #     ])
        #     ids.append(aid)
        #
        # if len(rel_xy) == 0:  # nothing visible
        #     rel_xy = np.empty((0, 2), dtype=np.float32)
        # else:
        #     rel_xy = np.asarray(rel_xy, dtype=np.float32)
        #     features = np.asarray(features, dtype=np.float32)
        #
        # # ------------------------------------------------------------------
        # # 3. Tokenise each visible agent **in the shifted frame**
        # # ------------------------------------------------------------------
        # token_ids = tokenize_traj_waypoints(
        #     rel_xy,
        #     self.x_boundaries,
        #     self.y_boundaries,
        #     self.local2token
        # )
        return None

    def move_ego(self, new_x: float, new_y: float) -> None:
        """
        Hard-warp the ego to (new_x, new_y) in world coords, then
        refresh every other vehicle’s relative position.  Use with care:
        • No dynamic constraints
        • No collision check
        """
        self.ego.position = np.array([new_x, new_y], dtype=np.float32)

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = None, None, None, None, {"image": self.render()}
        # # 1) step ego + traffic model
        # obs, reward, terminated, truncated, info = super().step(action)
        #
        # # 2) bump our timestep
        self.t += 0.2
        #
        # # 3) overwrite every other vehicle from the replay buffer
        self._apply_background(self.t)
        #
        # # (we leave collision/reward as-is for now)
        if round(self.t, 0) >= 20 or ():
            terminated = True
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        ego_info = self.ego.to_dict()
        ego_x, ego_y = ego_info["x"], ego_info["y"]
        # get limitation
        # get all available vehicle & obj
        for each in self.all_agents.values():
            # check it is available or not
            pass
        # shift
        # build obs

    def _apply_background(self, t):
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
        df_t = self.agent_feature[self.agent_feature['timestamp'] == t]

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
            if hasattr(veh, 'acceleration'):
                veh.acceleration = float(row['acc'])
            veh.t += t

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
            self.ego_goal = self.ego_feature.sort_values('timestamp').iloc[-1]

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
                agent = Vehicle(self.road, position=position,
                                heading=float(agent_start_loc["heading"]),
                                speed=float(agent_start_loc["v"]))
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
        goal_pos = [float(self.ego_goal['x']), float(self.ego_goal['y'])]
        goal_marker = Landmark(self.road, position=goal_pos)
        goal_marker.color = (0, 255, 0)  # bright green, for example
        goal_marker.LENGTH = 3.0  # make it small
        goal_marker.WIDTH = 3.0
        self.road.objects.append(goal_marker)
        print(self.agent_feature[self.agent_feature.timestamp == 0.0].shape[0], len(self.road.vehicles))
