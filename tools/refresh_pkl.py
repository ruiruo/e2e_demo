import os
import tqdm
import yaml
import pickle

cfg_path = "/home/nio/reparke2e/configs/npz_gen.yaml"

with open(cfg_path, 'r') as f:
    config_obj = yaml.safe_load(f)
all_scene = []
goal_path = config_obj["tar_dir"]
for scene_item in tqdm.tqdm(os.listdir(config_obj["data_dir"])):
    scene_path = os.path.join(config_obj["data_dir"], scene_item)
    with open(os.path.join(scene_path), "rb") as f:
        scene = pickle.load(f)
        framestack = {
            "agent_feature": scene["agent_feature"],  # np.array
            "ego_history_feature": scene["ego_history_feature"],  # np.array
            "agent_attribute_feature": scene["agent_attribute_feature"],  # np.array
            "total_frame": scene["agent_feature"].shape[1]
        }
        with open(os.path.join(goal_path, scene_item), "wb") as fb:
            pickle.dump(framestack, fb)
