from __future__ import annotations
from collections import defaultdict
from envs.utils import quantize_to_step
from gymnasium import spaces
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.lane import PolyLaneFixedWidth
from highway_env.road.road import Road, RoadNetwork, LineType
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle, Landmark
from pathlib import Path
from typing import Any, Dict
from utils.config import Configuration
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random

LIMIT = 200.0
epsilon = 1e-3

class ReplayHighwayCoreEnv(AbstractEnv):
    metadata = {"render_modes": ["rgb_array"]}


    def __init__(self, task_paths: str | Path, cfg: Configuration, figure_path="./"):
        self.task_paths = Path(task_paths)
        self.figure_path = Path(figure_path)
        self.cfg = cfg
        highway_cfg = {
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "offscreen_rendering": True,
        }

        # placeholders filled in `_load_new()` / `_make_road()`
        self.road: Road | None = None
        self.ego: Vehicle | None = None
        self.ego_meta = None
        self.all_agents: dict[int, Vehicle | Obstacle] = {}
        self.goal_xy: np.ndarray | None = None  # (2,)
        self.t: float = 0.0
        self.max_agents = 20
        super().__init__(highway_cfg, render_mode="rgb_array")
        # --- gym spaces (continuous, unbounded → wrappers will clamp) ---
        self.action_space = spaces.Box(low=-LIMIT, high=LIMIT, shape=(2,), dtype=np.float32)
        # obs is dict; space defined lazily in `reset()` when we know #agents
        self.observation_space = spaces.Dict({
            "ego": spaces.Box(low=-LIMIT, high=LIMIT, dtype=np.float32, shape=[5]),
            "agents": spaces.Box(low=-LIMIT, high=LIMIT, dtype=np.float32, shape=[20, 7]),
            "goal": spaces.Box(low=-LIMIT, high=LIMIT, dtype=np.float32, shape=[2]),
            "time": spaces.Box(low=-LIMIT, high=LIMIT, dtype=np.float32, shape=[1]),
        })

        self._prev_dist = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._load_new_episode()
        self.draw_a_image()
        obs = self._make_observation()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: Action):
        # Action = delta‑xy motion in world frame (no physics)
        self._move_world(action)
        obs = self._make_observation()
        reward = self._reward()
        terminated = bool(np.linalg.norm(self.ego.position - self.goal_xy) < 2.0)
        truncated = False  # no time limit here – wrapper can add
        info = {"image": self.render()}
        return obs, reward, terminated, truncated, info

    def _load_new_episode(self):
        """Pick a random pickle and rebuild the world"""
        self.all_agents.clear()
        self.t = 0.0

        self._read_file()
        self._make_road_world()
        self._prev_dist = float(np.linalg.norm(self.ego.position - self.goal_xy))

    def _make_observation(self) -> Dict[str, Any]:
        ego_state = np.asarray([
            *self.ego.position,
            self.ego.heading,
            self.ego.speed,
            getattr(self.ego.action, "acceleration", 0.0),
        ], dtype=np.float32)

        # agents except ego
        quantized_t = quantize_to_step(self.t, step=0.2, method="round")
        df_agents = self.agent_feature[abs(self.agent_feature["timestamp"] - quantized_t) < epsilon]
        df_agents = df_agents[df_agents.aid != 0]  # drop ego row if any

        agents_raw = df_agents[[
            "x", "y", "heading", "v", "acc", "length", "width"
        ]].to_numpy(dtype=np.float32) if len(df_agents) else np.zeros((0, 7), dtype=np.float32)

        # pad / truncate
        if agents_raw.shape[0] < self.max_agents:
            pad = np.zeros((self.max_agents - agents_raw.shape[0], 7), dtype=np.float32)
            agents = np.vstack([agents_raw, pad])
        else:
            agents = agents_raw[:self.max_agents]

        # goal
        goal = self.goal_xy.astype(np.float32)

        return {
            "ego": ego_state,
            "agents": agents,
            "goal": goal,
            "time": np.array([self.t], dtype=np.float32),
        }

    def _move_world(self, delta_xy: np.ndarray):
        segment_times = np.hypot(*delta_xy) / self.ego.speed
        self.ego.position += delta_xy
        self.t += segment_times
        self._apply_background_world()
        self.draw_a_image()

    def _apply_background_world(self):
        """Overwrite other vehicles at current time step (world coords)."""
        step_time = quantize_to_step(self.t, step=0.2, method="round")
        df_t = self.agent_feature[abs(self.agent_feature["timestamp"] - step_time) < epsilon]
        for _, row in df_t.iterrows():
            aid = int(row["aid"])
            veh = self.all_agents.get(aid)
            if veh is None:
                continue
            veh.position = np.array([row["x"], row["y"]], dtype=float)
            veh.heading = float(row["heading"])
            veh.speed = float(row["v"])
            if hasattr(veh, "action"):
                veh.action["acceleration"] = float(row["acc"])

    def _reward(self, *, time_penalty: float = 0.01) -> float:
        """Dense reward: positive progress toward goal minus a small time cost.

        * **Progress term**:  Δdistance (prev − current) × 10.0
          – the agent gets +10 for every metre it moves closer.
        * **Time penalty**  : constant −0.01 each step to encourage efficiency.
        """

        cur_dist = float(np.linalg.norm(self.ego.position - self.goal_xy))
        progress = (self._prev_dist - cur_dist) * 10.0
        reward = progress - time_penalty
        # update tracker for next step
        self._prev_dist = cur_dist
        return reward

    def _read_file(self):
        task_dir = random.choice(list(self.task_paths.iterdir()))
        pkl_file = random.choice(list(task_dir.iterdir()))
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        # ego & agent features now kept in world frame -----------------
        data["ego_history_feature"] = data["ego_history_feature"][::-1]
        raw_end_t = np.round(data["ego_history_feature"][0, 7].copy(), 1)
        data["ego_history_feature"][:, 7] = np.round(raw_end_t - data["ego_history_feature"][:, 7], 1)
        self.ego_feature = pd.DataFrame(
            data["ego_history_feature"][:, 2:],
            columns=["x", "y", "heading", "v", "acc", "timestamp"],
        ).iloc[0].to_dict()
        self.ego_meta = pd.DataFrame(
            data["ego_history_feature"][:, 2:],
            columns=["x", "y", "heading", "v", "acc", "timestamp"],
        ).to_dict(orient="records")

        agent_feature = pd.DataFrame(
            data["agent_feature"].reshape(-1, 8)[:, 1:][::-1],
            columns=["aid", "x", "y", "heading", "v", "acc", "timestamp"],
        )
        agent_attr = pd.DataFrame(
            data["agent_attribute_feature"],
            columns=["aid", "length", "width", "type", "virtual", "key_smooth_future"],
        )

        self.agent_feature = agent_feature.merge(agent_attr, on="aid").reset_index(drop=True)
        self.agent_feature['timestamp'] = np.round(raw_end_t - self.agent_feature['timestamp'], 1)
        self.vector_graph_feature = data["vector_graph_feature"]

        # goal is last ego row (world frame)
        goal_row = pd.DataFrame(
            data["ego_history_feature"][:, 2:],
            columns=["x", "y", "heading", "v", "acc", "timestamp"],
        ).iloc[-1].to_dict()
        self.goal_xy = np.array([goal_row["x"], goal_row["y"]], dtype=float)

    def _make_road_world(self):
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

        # ---------------- vehicles ----------------
        zero_time = self.agent_feature[abs(self.agent_feature["timestamp"]) < epsilon]
        for aid, df in zero_time.groupby("aid"):
            row = df.iloc[0]
            pos = [row["x"], row["y"]]
            if aid == 0:
                speed = row["v"] if row["v"] else 5.0
                veh = Vehicle(None, position=pos, heading=row["heading"], speed=speed)
                veh.color = (255, 255, 255)
                self.ego = veh
            else:
                veh = Vehicle(None, position=pos) if row["type"] == 791621440 else Obstacle(None, position=pos)
            veh.LENGTH = row["length"]
            veh.WIDTH = row["width"]
            self.all_agents[aid] = veh

        self.road = Road(network, vehicles=[self.ego] + list(self.all_agents.values()), record_history=False)
        marker = Landmark(self.road, position=self.goal_xy.tolist())
        self.road.objects.append(marker)

    def draw_a_image(self):
        img = self.render()
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.savefig(Path.joinpath(self.figure_path, str(self.t) + ".jpg"),
                    format='jpg', bbox_inches='tight', pad_inches=0)
