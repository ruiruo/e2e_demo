import cv2
import os
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from utils.config import Configuration


class OpenCVRecorder:
    def __init__(self, env, video_path="replay.mp4", fps=15):
        self.env = env
        self.video_path = video_path
        self.fps = fps
        self.frames = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = info.get("image")
        self.frames = [frame]
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        frame = info.get("image")
        self.frames.append(frame)
        return obs, reward, done, truncated, info

    def get_obs(self):
        pass

    def save(self):
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.video_path) or ".", exist_ok=True)
        h, w, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (w, h))
        for rgb in self.frames:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()


class TokeniseWrapper(gym.ObservationWrapper, gym.ActionWrapper):
    """Convert ego‑centric continuous xy to discrete token IDs (and back)."""

    def __init__(self, env: gym.Env, cfg: Configuration):
        super().__init__(env)
        self.cfg = cfg
        self.local2token = np.load(cfg.tokenizer)
        with open(cfg.detokenizer) as f:
            self.token2local = json.load(f)

        # new spaces: actions are still delta‑xy tokens? → keep same as before
        self.action_space = spaces.Discrete(len(self.local2token))
        # observation will be a fixed‑length flat int vector compatible with old model
        n = 1 + 3 + (cfg.max_frame + 1) * cfg.max_agent * 6 + 2  # rough
        self.observation_space = spaces.Box(low=0, high=len(self.local2token), shape=(n,), dtype=np.int32)

    # ---- helpers ----
    def _xy_to_token(self, pts):
        # naive search (replace with vectorised binning)
        ids = [int(self.local2token[tuple(p)]) if tuple(p) in self.local2token else self.cfg.pad_token for p in pts]
        return np.array(ids, dtype=np.int32)

    def _token_to_xy(self, token_id):
        return np.array(self.token2local.get(str(int(token_id)), [0, 0]), dtype=float)

    # ---- observation ----
    def observation(self, obs):
        # ego is origin; so only need to discretise agents + goal
        input_ids = self._xy_to_token(np.array([[0, 0]]))  # BOS dummy
        goal_id = self._xy_to_token(obs["goal"][None, :])

        agents_xy = obs["agents"][:, 0:2]
        agent_ids = self._xy_to_token(agents_xy).reshape(-1, 1)
        agent_feats = obs["agents"][:, 2:]  # heading, v, acc, l, w
        agent_tokens = np.concatenate([agent_ids, agent_feats], axis=1)

        flat = np.concatenate([
            input_ids.flatten(),
            obs["ego"][2:].astype(np.float32),  # heading speed acc
            agent_tokens.flatten(),
            goal_id.flatten(),
        ])
        return flat

    # ---- action ----
    def action(self, token_id):
        xy = self._token_to_xy(token_id)
        return xy.astype(np.float32)


class EgoCenterWrapper(gym.ObservationWrapper, gym.ActionWrapper):
    """Convert world‑frame env to ego‑centric frame.

    *Observation*
        - ego features unchanged (now at origin, heading=0)
        - agents / goal translated + rotated
    *Action*
        expects **delta in ego frame**; converts to world frame before step.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._last_heading = 0.0
        self._last_pos = np.zeros(2)

        # wrap spaces (same shapes)
        self.observation_space = env.observation_space  # type: ignore
        self.action_space = env.action_space  # type: ignore

    # ----- helpers -----
    @staticmethod
    def _rot_matrix(theta: float):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    # ----- observation -----
    def observation(self, obs):
        self._last_pos = obs["ego"][:2].copy()
        self._last_heading = obs["ego"][2].copy()
        R = self._rot_matrix(-self._last_heading)

        # translate+rotate agents
        agents = obs["agents"].copy()
        if agents.size:
            agents[:, 0:2] = (agents[:, 0:2] - self._last_pos) @ R.T
            agents[:, 2] -= self._last_heading
        # goal
        goal = (obs["goal"] - self._last_pos) @ R.T

        obs["agents"] = agents
        obs["goal"] = goal
        # ego is now at origin (0,0) heading 0; keep speed, acc
        obs["ego"][:2] = 0.0
        obs["ego"][2] = 0.0
        return obs

    # ----- action -----
    def action(self, action):
        # convert ego‑frame delta to world‑frame delta
        R = self._rot_matrix(self._last_heading)
        world_delta = np.asarray(action) @ R.T
        return world_delta.astype(np.float32)
