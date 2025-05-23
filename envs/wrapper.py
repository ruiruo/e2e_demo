import cv2
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType, WrapperObsType
from gymnasium.spaces.utils import flatdim


# OPENCV, RayFlatten, EGO_INFO, AGENT_INFO, EGO_STEP, AGENT_STEP, CORE

class RayFlattenWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(RayFlattenWrapper, self).__init__(env)
        self.observation_space = flatdim(self.env.observation_space)

    # todo: update after all other wrapper done
    def observation(self, obs):
        flattened_obs = np.concatenate([np.array(obs["input_ids"]).flatten(),
                                        np.array(obs["ego_info"]).flatten(),
                                        np.array(obs["agent_info"]).flatten(),
                                        np.array(obs["goal"]).flatten(),
                                        ])
        return flattened_obs


# todo: add TopologyHistory in agent_alignment into obs
# req: use abstract X, Y now.
class TopologyHistoryWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(TopologyHistoryWrapper, self).__init__(env)
        # todo: obs["agents"] -> obs["agent_info"]

    def build_agent_info(self):
        pass

    def observation(self, observation: ObsType) -> WrapperObsType:
        return observation


# todo: Trans abstract ego info into Info for auto-reg type
# req: use abstract X, Y now.
class EgoInfoWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(EgoInfoWrapper, self).__init__(env)
        # todo: obs["ego"] -> obs["input_ids"], obs["ego_info"]

    def build_ego_info(self):
        pass

    def observation(self, observation: ObsType) -> WrapperObsType:
        return observation

# todo: rebuild ego action, use waypoint token as input
# req: use abstract X, Y now.
class EgoStepWrapper(gym.ActionWrapper):
    def __init__(self, env, token_table, max_token):
        super(EgoStepWrapper, self).__init__(env)
        self.token_table = token_table
        self.max_token = max_token
        self.action_space = spaces.Box(low=0, high=len(self.token_table), shape=[self.max_token], dtype=np.int8)
        # todo: obs["agents"] -> obs["agent_info"]

    def action(self, action):
        pass


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
