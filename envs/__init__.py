import os
import pickle
import random
import numpy as np
from gymnasium import spaces
from highway_env.envs.common.abstract import Observation
from highway_env.envs.common.action import Action
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork, LineType, StraightLane


class ReplayHighwayEnv(AbstractEnv):
    def __init__(self,
                 task_paths: str):
        self.task_paths = task_paths
        config = {
            # where to pick up your pickles:
            "task_paths": None,
            # visuals:
            "screen_width": 600,
            "screen_height": 250,
            # tuning:
            "simulation_frequency": 15,
            "policy_frequency": 1,
        }
        super().__init__(config, render_mode="rgb_array")
        self.t = 0
        self.agent_feature = None
        self.agent_attribute_feature = None
        self.vector_graph_feature = None
        self.action_space = spaces.Discrete(5)

    def reset(self, *, seed=None, options=None):
        # 1) pick & load a new episode
        self._read_file()

        # 2) do HighwayEnv’s normal reset (builds self.road & self.vehicles)
        obs = None
        info = None

        # 3) place all other cars at frame 0
        self.t = 0
        # self._apply_frame(self.t)

        return obs, info

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        # 1) step ego + traffic model
        obs, reward, terminated, truncated, info = super().step(action)

        # 2) bump our timestep
        self.t += 1

        # 3) overwrite every other vehicle from the replay buffer
        # self._apply_frame(self.t)

        # (we leave collision/reward as-is for now)
        return obs, reward, terminated, truncated, info

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
            self.agent_feature = self.data_info['agent_feature']
            self.agent_attribute_feature = self.data_info['agent_attribute_feature']
            self.vector_graph_feature = self.data_info['vector_graph_feature']

    def _make_road(self):
        """
        Build self.road from self.vector_graph_feature,
        which is shape (n_polylines, n_vectors, 9):
          [start_x, start_y, end_x, end_y,
           road_id, width, left_attr, right_attr, speed_limit]
        """
        vg = self.vector_graph_feature
        n_polylines, n_vectors, _ = vg.shape
        network = RoadNetwork()

        for lane_idx in range(n_polylines):
            # 1) gather the chain of points for this polyline
            points = []
            for vec_idx in range(n_vectors):
                sx, sy, ex, ey, _, _, _, _, _ = vg[lane_idx, vec_idx]
                if sx < -200:  # placeholder check
                    continue
                points.append((float(sx), float(sy)))
            # also append the last end point
            last = vg[lane_idx, -1]
            points.append((float(last[2]), float(last[3])))

            # 2) for each segment, create a StraightLane
            #    and add it under a fixed (from, to) key so
            #    that all segments belong to the same logical lane
            from_node = f"lane{lane_idx}_start"
            to_node = f"lane{lane_idx}_end"
            speed = float(last[8])

            for i in range(len(points) - 1):
                p0 = np.array(points[i])
                p1 = np.array(points[i + 1])
                # you can refine line_types based on left/right attrs if you like
                lt = [LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE]
                lane_geom = StraightLane(p0, p1, line_types=lt, speed_limit=speed)
                network.add_lane(from_node, to_node, lane_geom)

        # 3) wrap into a Road for simulation
        self.road = Road(
            network,
            record_history=self.config["show_trajectories"]
        )
