from typing import Dict, Optional, List
import os
import pickle
import numpy as np
import copy
import glob
import multiprocessing
import yaml
from collections import defaultdict
from easydict import EasyDict
import tensorflow as tf

YELLOW = "\033[33m"
GREEN = "\033[32m"
RESET = "\033[0m"
RED = "\033[31m"


def parse_tfrecord(record_path, label_match_mode: str = False):
    ds = tf.data.TFRecordDataset(record_path, buffer_size=512 * 1024 * 1024)  # buffer: 512M

    def parse_func(record):
        try:
            features = {
                "fs_scenes_traj_info": tf.io.VarLenFeature(dtype=tf.float32),
                "fs_scenes_traj_info_shape": tf.io.VarLenFeature(dtype=tf.int64),
                "fs_scenes_source_tag": tf.io.VarLenFeature(dtype=tf.float32),
                "fs_scenes_source_tag_shape": tf.io.VarLenFeature(dtype=tf.int64),
                "fs_scenes_attribute": tf.io.VarLenFeature(dtype=tf.float32),
                "fs_scenes_attribute_shape": tf.io.VarLenFeature(dtype=tf.int64),
                "agent_feature": tf.io.VarLenFeature(dtype=tf.float32),
                "agent_feature_shape": tf.io.VarLenFeature(dtype=tf.int64),
                "agent_attribute_feature": tf.io.VarLenFeature(dtype=tf.float32),
                "agent_attribute_feature_shape": tf.io.VarLenFeature(dtype=tf.int64),
                "ego_history_feature": tf.io.VarLenFeature(dtype=tf.float32),
                "ego_history_feature_shape": tf.io.VarLenFeature(dtype=tf.int64),

                "lat_lon_distance_gt_2_fs": tf.io.VarLenFeature(dtype=tf.float32),
                "lat_lon_distance_gt_2_fs_shape": tf.io.VarLenFeature(dtype=tf.int64),
                # "label_distance": tf.io.VarLenFeature(dtype=tf.float32),
                # "label_distance_shape": tf.io.VarLenFeature(dtype=tf.int64),

                "behavior_category": tf.io.VarLenFeature(dtype=tf.float32),
                "behavior_category_shape": tf.io.VarLenFeature(dtype=tf.int64),
                "point_tag": tf.io.VarLenFeature(dtype=tf.int64),

                "fs_reward": tf.io.VarLenFeature(dtype=tf.float32),
                "lane_change_trigger_type": tf.io.VarLenFeature(dtype=tf.int64),

                "ref_path_info": tf.io.VarLenFeature(dtype=tf.float32),
                "ref_path_info_shape": tf.io.VarLenFeature(dtype=tf.int64),
                "ref_line_info": tf.io.VarLenFeature(dtype=tf.float32),
                "ref_line_info_shape": tf.io.VarLenFeature(dtype=tf.int64),

                "vector_graph_feature": tf.io.VarLenFeature(dtype=tf.float32),
                "vector_graph_feature_shape": tf.io.VarLenFeature(dtype=tf.int64),

                "vector_graph_feature_mask": tf.io.VarLenFeature(dtype=tf.float32),
                "vector_graph_feature_mask_shape": tf.io.VarLenFeature(dtype=tf.int64),

                "vector_line_mask": tf.io.VarLenFeature(dtype=tf.float32),
                "vector_line_mask_shape": tf.io.VarLenFeature(dtype=tf.int64),

                "env_label_info": tf.io.VarLenFeature(dtype=tf.float32),
                "env_label_info_shape": tf.io.VarLenFeature(dtype=tf.int64),

                "fs_scenes_costs_tensor": tf.io.VarLenFeature(dtype=tf.float32),
                "fs_scenes_costs_tensor_shape": tf.io.VarLenFeature(dtype=tf.int64),

                "lane_map_navi_info_feature": tf.io.VarLenFeature(dtype=tf.float32),
                "lane_map_navi_info_feature_shape": tf.io.VarLenFeature(dtype=tf.int64),

                "lane_map_navi_info_feature_mask": tf.io.VarLenFeature(dtype=tf.float32),
                "lane_map_navi_info_feature_mask_shape": tf.io.VarLenFeature(dtype=tf.int64),

                "navi_lanenr": tf.io.VarLenFeature(dtype=tf.float32),
                "navi_lanenr_shape": tf.io.VarLenFeature(dtype=tf.int64),

                "navi_lanenr_mask": tf.io.VarLenFeature(dtype=tf.float32),
                "navi_lanenr_mask_shape": tf.io.VarLenFeature(dtype=tf.int64),

                "navi_feature": tf.io.VarLenFeature(dtype=tf.float32),
                "navi_feature_shape": tf.io.VarLenFeature(dtype=tf.int64),
                "navi_debug": tf.io.VarLenFeature(dtype=tf.int64),
                "navi_debug_shape": tf.io.VarLenFeature(dtype=tf.int64),

                "vector_route_navi_feature": tf.io.VarLenFeature(dtype=tf.float32),
                "vector_route_navi_feature_shape": tf.io.VarLenFeature(dtype=tf.int64),
            }

            if label_match_mode:
                pass
                # features.update({
                #     "lane_change_task": tf.io.VarLenFeature(dtype=tf.float32),
                # })

            parsed_features = tf.io.parse_single_example(record, features)
            return parsed_features
        except tf.errors.DataLossError as e:
            print("parse_func(): ", e)
            return None

    try:
        ds = ds.map(parse_func)
    except Exception as e:
        print(e)
    return ds


def parse_tfrecord_infer_mode(record_path, fea_cfg: dict = {}):
    ds = tf.data.TFRecordDataset(record_path, buffer_size=512 * 1024 * 1024)  # buffer: 512M

    def parse_func(record):
        try:
            # # NOTE 新数据结构
            features = {
                "fs_scenes_traj_info_shape": tf.io.VarLenFeature(dtype=tf.int64),
                "fs_scenes_traj_info": tf.io.VarLenFeature(dtype=tf.float32),
                # shape = [scene num, agt num, traj len, fea dim]
                "scene_output": tf.io.VarLenFeature(dtype=tf.float32),
                "valid_scene_num": tf.io.VarLenFeature(dtype=tf.int64),
                "agent_mask": tf.io.VarLenFeature(dtype=tf.float32),
                "agent_mask_shape": tf.io.VarLenFeature(dtype=tf.int64),
                "behavior_category": tf.io.VarLenFeature(dtype=tf.float32),
                "behavior_category_shape": tf.io.VarLenFeature(dtype=tf.int64),
                "is_valid_scene": tf.io.VarLenFeature(dtype=tf.float32),
                "is_valid_scene_shape": tf.io.VarLenFeature(dtype=tf.int64),
                "point_tag": tf.io.VarLenFeature(dtype=tf.int64),

                "time_mask": tf.io.VarLenFeature(dtype=tf.float32),
                "time_mask_shape": tf.io.VarLenFeature(dtype=tf.int64),
            }

            parsed_features = tf.io.parse_single_example(record, features)
            return parsed_features
        except tf.errors.DataLossError as e:
            print("parse_func(): ", e)
            return None

    try:
        ds = ds.map(parse_func)
    except Exception as e:
        print(e)
    return ds


def set_memory_growth():
    """设置 TensorFlow 的内存增长"""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    # import pdb;pdb.set_trace()
    # print(gpus)
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def write_pkl(output_save_path, data):
    dir_path = os.path.dirname(output_save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(output_save_path, "wb") as f:
        pickle.dump(data, f)


def load_data(data_dir, prefix: str = ""):
    with open(f"{data_dir}/{prefix}train_input.pkl", "rb") as f:
        train_input = pickle.load(f)

    with open(f"{data_dir}/{prefix}train_target.pkl", "rb") as f:
        train_target = pickle.load(f)

    with open(f"{data_dir}/{prefix}test_input.pkl", "rb") as f:
        test_input = pickle.load(f)

    with open(f"{data_dir}/{prefix}test_target.pkl", "rb") as f:
        test_target = pickle.load(f)

    print(f"Loading data from {data_dir}.")
    return train_input, train_target, test_input, test_target


def load_pkl(pkl_path: str):
    if not os.path.isfile(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)
    return pkl_data


def load_spec_fea(input_data: Dict[str, np.ndarray], fea_index_map: dict):
    res = defaultdict()
    for fea_name, fea_cfg in fea_index_map.items():
        if fea_name not in input_data:
            continue
        res[fea_name] = input_data[fea_name][fea_cfg.shape]

    # for key in input_data.keys():
    #     if key == "traj_data":
    #         res[key] = input_data[key][:, :, :, :, fea_index_map["traj_data"]]
    #         # import pdb;pdb.set_trace()
    #     elif key == "env_data":
    #         res[key] = input_data[key][..., fea_index_map["env_data"]]
    #     # elif key == "lateral_distance_labels":
    #     #     res[key] = -input_data[key] + 10
    #     else:
    #         res[key] = input_data[key]

    del input_data
    return res


def load_processed_data(data_dir, prefix: str = ""):
    pkl_path = f"{data_dir}/{prefix}_processed_data.pkl"
    print(f"load data from {pkl_path}")
    with open(pkl_path, "rb") as f:
        processed_data = pickle.load(f)

    return processed_data


def load_processed_data_batch(data_dir, prefix: str = ""):
    pattern = f"{prefix}_processed_data_*.pkl"

    # 获取目录下所有匹配的文件名
    file_path_list = glob.glob(os.path.join(data_dir, pattern))

    temp = defaultdict(list)
    res = defaultdict()
    count = 0
    for file_path in file_path_list:
        if count >= 8:
            break
        with open(file_path, "rb") as f:
            print(f"load data from {file_path}")
            processed_data = pickle.load(f)
            # import pdb;pdb.set_trace()

            for key, val in processed_data.items():
                temp[key].append(val)
                print(f"append {key}")
        count += 1
    for key, val in temp.items():
        print(f"concat {key}")
        res[key] = np.concatenate(val, axis=0)
    # import pdb;pdb.set_trace()
    del temp
    return res


# Define a recursive defaultdict function
def nested_dict():
    return defaultdict(nested_dict)


def write_list_to_txt(data: list, file_path: str):
    with open(file_path, "w") as file:
        for item in data:
            file.write(item + "\n")


def load_data_traj(data_dir, prefix: str = ""):
    with open(f"{data_dir}/{prefix}_train_input.pkl", "rb") as f:
        train_input = pickle.load(f)

    with open(f"{data_dir}/{prefix}_train_target.pkl", "rb") as f:
        train_target = pickle.load(f)

    with open(f"{data_dir}/{prefix}_train_traj.pkl", "rb") as f:
        train_traj = pickle.load(f)

    with open(f"{data_dir}/{prefix}_test_input.pkl", "rb") as f:
        test_input = pickle.load(f)

    with open(f"{data_dir}/{prefix}_test_target.pkl", "rb") as f:
        test_target = pickle.load(f)

    with open(f"{data_dir}/{prefix}_test_traj.pkl", "rb") as f:
        test_traj = pickle.load(f)

    with open(f"{data_dir}/{prefix}_test_rule_value.pkl", "rb") as f:
        test_rule_val = pickle.load(f)

    print(f"Loading data from {data_dir}.")
    return train_input, train_traj, train_target, test_input, test_traj, test_target, test_rule_val

def load_input_data_by_type(data_dir, input_type: str = "cost_data", prefix: str = "", only_test: bool = False):
    # input_file_name = INPUT_DATA_2_FILENAME_MAP[input_type]
    input_file_name = input_type
    train_data, test_data = None, None

    if only_test:
        print(f"Load input data from {GREEN}{data_dir}/{prefix}_test_{input_file_name}.pkl{RESET}")
        with open(f"{data_dir}/{prefix}_test_{input_file_name}.pkl", "rb") as f:
            test_data = pickle.load(f)
    else:
        print(f"Load input data from {GREEN}{data_dir}/{prefix}_train_{input_file_name}.pkl{RESET}")
        with open(f"{data_dir}/{prefix}_train_{input_file_name}.pkl", "rb") as f:
            train_data = pickle.load(f)
        with open(f"{data_dir}/{prefix}_test_{input_file_name}.pkl", "rb") as f:
            test_data = pickle.load(f)

    return train_data, test_data


def load_label_data_by_type(data_dir, label_type: str = "behavior_labels", prefix: str = ""):
    # output_file_name = INPUT_DATA_2_FILENAME_MAP[label_type]
    output_file_name = label_type
    print(f"Load label data from {GREEN}{data_dir}/{prefix}_train_{output_file_name}.pkl{RESET}")
    with open(f"{data_dir}/{prefix}_train_{output_file_name}.pkl", "rb") as f:
        train_target = pickle.load(f)
    with open(f"{data_dir}/{prefix}_test_{output_file_name}.pkl", "rb") as f:
        test_target = pickle.load(f)

    return train_target, test_target


def load_sample_data():
    with open("/share-global/chil.qiu/vns/sample_train_input.pkl", "rb") as f:
        sample_train_input = pickle.load(f)

    with open("/share-global/chil.qiu/vns/sample_train_target.pkl", "rb") as f:
        sample_train_target = pickle.load(f)

    with open("/share-global/chil.qiu/vns/sample_test_input.pkl", "rb") as f:
        sample_test_input = pickle.load(f)
        # shape = bs, traj_num=2, cost_fea_num=32

    with open("/share-global/chil.qiu/vns/sample_test_target.pkl", "rb") as f:
        sample_test_target = pickle.load(f)
        # shape = bs, 1
    # import pdb;pdb.set_trace()
    return sample_train_input, sample_train_target, sample_test_input, sample_test_target


def read_config(path: str) -> EasyDict:
    """
    Overview:
        read configuration from path
    Arguments:
        - path (:obj:`str`): Path of source yaml
    Returns:
        - (:obj:`EasyDict`): Config data from this file with dict type
    """
    if path:
        assert os.path.exists(path), path
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    return EasyDict(config)


def deep_update(
        original: dict,
        new_dict: dict,
        new_keys_allowed: bool = False,
        whitelist: Optional[List[str]] = None,
        override_all_if_type_changes: Optional[List[str]] = None,
):
    """
    Overview:
        Updates original dict with values from new_dict recursively.

    .. note::

        If new key is introduced in new_dict, then if new_keys_allowed is not
        True, an error will be thrown. Further, for sub-dicts, if the key is
        in the whitelist, then new subkeys can be introduced.

    Arguments:
        - original (:obj:`dict`): Dictionary with default values.
        - new_dict (:obj:`dict`): Dictionary with values to be updated
        - new_keys_allowed (:obj:`bool`): Whether new keys are allowed.
        - whitelist (Optional[List[str]]): List of keys that correspond to dict
            values where new subkeys can be introduced. This is only at the top
            level.
        - override_all_if_type_changes(Optional[List[str]]): List of top level
            keys with value=dict, for which we always simply override the
            entire value (:obj:`dict`), if the "type" key in that value dict changes.
    """
    whitelist = whitelist or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise RuntimeError("Unknown config parameter `{}`. Base config have: {}.".format(k, original.keys()))

        # Both original value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if k in override_all_if_type_changes and "type" in value and "type" in original[k] and value["type"] != \
                    original[k]["type"]:
                original[k] = value
            # Whitelisted key -> ok to add new subkeys.
            elif k in whitelist:
                deep_update(original[k], value, True)
            # Non-whitelisted key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original


def deep_merge_dicts(original: dict, new_dict: dict) -> dict:
    """
    Overview:
        merge two dict using deep_update
    Arguments:
        - original (:obj:`dict`): Dict 1.
        - new_dict (:obj:`dict`): Dict 2.
    Returns:
        - (:obj:`dict`): A new dict that is d1 and d2 deeply merged.
    """
    original = original or {}
    new_dict = new_dict or {}
    merged = copy.deepcopy(original)
    if new_dict:  # if new_dict is neither empty dict nor None
        deep_update(merged, new_dict, True, [])

    return merged


def get_file_path(
        save_dir: str,
        data_prefix: str,
        processed_mode: str = "balanced",
        train_mode: str = "train",
        file_index: int = 0,
):
    # pkl_path = save_dir + f"/{data_prefix}_" + \
    #             f"processed_{processed_mode}_{train_mode}_data_{file_index}.pkl"

    pkl_path = save_dir + f"{data_prefix}/" + f"{processed_mode}/" + f"{train_mode}/" + f"{file_index}.pkl"

    return pkl_path


def modify_shared_var_security(shared_var: multiprocessing.Value, lock: multiprocessing.Lock) -> int:
    if lock:
        with lock:
            file_save_index = shared_var.value
            shared_var.value += 1
    else:
        file_save_index = shared_var.value
        shared_var.value += 1
    return file_save_index


def distribute_files(files_list: list, num_workers: int = 1) -> List[str]:
    """Distribute files evenly among workers, ensuring all files are processed

    Parameters
    ----------
    files_list : list
        文件列表
    num_workers : int
        进程数量
    Returns
    -------
    distributed_files: List[str]
        每个 worker 负责的 file list
    """
    num_files = len(files_list)
    per_worker = num_files // num_workers
    remainder = num_files % num_workers

    # Split the file list more evenly (remainder handled)
    distributed_files = []
    start_idx = 0
    for i in range(num_workers):
        end_idx = start_idx + per_worker + (1 if i < remainder else 0)
        distributed_files.append(files_list[start_idx:end_idx])
        start_idx = end_idx

    return distributed_files


def clean_tmp_data(prefix: str, tmp_path: str):
    """清除本地缓存数据

    Parameters
    ----------
    prefix: str
        tmp_path如果包含字符串 prefix 则清除tmp_path
    tmp_path : str
        case本地路径
    """
    if prefix in tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)


# 自定义 Dumper，强制单行列表
class CustomDumper(yaml.Dumper):
    def represent_list(self, data):
        # 将列表表示为单行形式
        return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


# 注册自定义列表表示方式
CustomDumper.add_representer(list, CustomDumper.represent_list)


def save_yaml(input_data: dict, save_path: str):
    with open(save_path, "w") as file:
        yaml.dump(input_data, file, Dumper=CustomDumper, default_flow_style=False, allow_unicode=True)


def read_txt(file_path: str) -> List[str]:
    """读取txt文件，以行为元素返回List
    """
    with open(file_path, "r") as f:
        all_files = [line.strip() for line in f]

    return all_files
