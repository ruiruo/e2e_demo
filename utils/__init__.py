import os
import ray
import yaml
import socket
from envs.wrapper import OpenCVRecorder, RayFlattenWrapper, EgoStepWrapper, EgoInfoWrapper, TopologyHistoryWrapper
from envs.core import ReplayHighwayCoreEnv


def split_list_into_n_parts(lst, n=10):
    return [lst[i::n] for i in range(n)]


def find_free_port(start=5000, end=5100):
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no port in {start}~{end}")


def env_creator(env_config):
    env = ReplayHighwayCoreEnv(env_config["task_paths"], env_config)

    return env


def copy_params(offline, online):
    layer = list(offline.collect_params().values())
    for i in layer:
        _1 = online.collect_params().get(
            "_".join(i.name.split("_")[1:])).data().asnumpy()
        online.collect_params().get("_".join(i.name.split("_")[1:])).set_data(
            i.data())
        _2 = online.collect_params().get(
            "_".join(i.name.split("_")[1:])).data().asnumpy()


def check_dir(i):
    # create required path
    if not os.path.exists("{}/".format(i)):
        os.mkdir("{}/".format(i))


def init_ray(ray_setting=None):
    if ray_setting is not None:
        with open(ray_setting, 'r') as file:
            settings = yaml.safe_load(file)
        ray.init(**settings)
    else:
        ray.init("auto")
