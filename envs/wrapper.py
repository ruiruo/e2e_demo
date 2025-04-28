import cv2
import os


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
