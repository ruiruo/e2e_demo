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
from highway_env.vehicle.objects import Obstacle


class ReplayHighwayEnv(AbstractEnv):
    def __init__(self,
                 task_paths: str):
        self.task_paths = task_paths
        config = {
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
        self.agent_feature = None
        self.vector_graph_feature = None
        self.ego_feature = None
        self.ego = None
        self.all_agents = {}
        self.t = 0
        super().__init__(config, render_mode="rgb_array")
        self.action_space = spaces.Discrete(5)

    def reset(self, *, seed=None, options=None):
        # 1) pick & load a new episode
        self._read_file()
        self._make_road()
        # 2) do HighwayEnv’s normal reset (builds self.road & self.vehicles)
        obs = None
        info = None

        # 3) place all other cars at frame 0
        self.t = 0
        # self._apply_frame(self.t)

        return obs, info

    def get_current_obs(self):
        pass

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        # # 1) step ego + traffic model
        # obs, reward, terminated, truncated, info = super().step(action)
        #
        # # 2) bump our timestep
        # self.t += 1
        #
        # # 3) overwrite every other vehicle from the replay buffer
        # # self._apply_frame(self.t)
        #
        # # (we leave collision/reward as-is for now)
        # return obs, reward, terminated, truncated, info
        pass

    def _apply_frame(self, t: int):
        """
        Overwrite all non-ego vehicles in self.road.vehicles[1:]
        using your loaded `agent_feature` and `agent_attribute_feature`.
        """
        for i, veh in enumerate(self.road.vehicles[1:], start=0):
            print(veh)
            # feat = self.agent_feature[i, t]  # [id, x, y, heading, v, acc, Δt]
            # attr = self.agent_attribute_feature[i]  # [length, width, type, is_virtual, ...]
            # # position + heading + dynamics
            # veh.position = [float(feat[1]), float(feat[2])]
            # veh.heading = float(feat[3])
            # # veh.velocity = float(feat[4])
            # # veh.acceleration = float(feat[5])
            # # physical shape & type
            # veh.length = float(attr[0])
            # veh.width = float(attr[1])
            # veh.TYPE = int(attr[2])
            # veh.is_virtual = bool(attr[3])

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
                                                "heading", "v", "acc", "length", "width", "type", "timestamp"]]
            self.agent_feature = self.agent_feature[self.agent_feature.aid != -300].reset_index(drop=True)

            self.vector_graph_feature = self.data_info['vector_graph_feature'][::-1]
            self.vector_graph_feature[:, :, 0:2] -= pos_bias
            self.vector_graph_feature[:, :, 2:4] -= pos_bias

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
            record_history=False
        )
        print(self.agent_feature[self.agent_feature.timestamp == 0.0].shape[0], len(self.road.vehicles))
