import numpy as np
from typing import List, Dict
from collections import defaultdict

from unpark.constants import SCENE_CONDITION_MAP


def nested_dict():
    return defaultdict(nested_dict)


def valid_frame(frame_data):
    if frame_data["cost_result"].shape[0] == 0:
        return None

    cost_fs_scene_names_v = None
    try:
        cost_fs_scene_names_v = frame_data["cost_fs_scene_names"].values[0].numpy().decode("utf-8")
    except UnicodeDecodeError:
        cost_fs_scene_names_v = frame_data["cost_fs_scene_names"].values[0].numpy().decode("ISO-8859-1")
    if cost_fs_scene_names_v[-2] == "0":
        return None

    return cost_fs_scene_names_v


def filter_data_by_fea_range(input_data: np.ndarray, condition_dict: dict) -> np.ndarray:
    """
    condition_dict = {
        fea_i: range,
        0: [-1000, 1000],
        ...,
    }
    """

    validate_condition = np.ones((input_data.shape[0]), dtype=bool)
    for fea_i, odd_range in condition_dict.items():
        validate_condition = (
                                     (input_data[:, 0, fea_i] < odd_range[1])
                                     & (input_data[:, 0, fea_i] > odd_range[0])
                                     & (input_data[:, 1, fea_i] < odd_range[1])
                                     & (input_data[:, 1, fea_i] > odd_range[0])
                             ) & validate_condition

    validate_data = input_data[validate_condition]

    return validate_data


def clamp_data_by_fea_range(input_data: np.ndarray, range_map: dict):
    for fea_i, odd in range_map.items():
        # import pdb;pdb.set_trace()
        input_data[:, :, fea_i] = np.clip(input_data[:, :, fea_i], odd[0], odd[1])

    return input_data


def get_batches(dataloader):
    for data in dataloader:
        yield data


def is_used_scene(scene_name, used_scene_prefixs):
    return any([SCENE_CONDITION_MAP[f"{item}_scene"](scene_name) for item in used_scene_prefixs])


def is_diff_too_small_between_list(fs_scene_value_list: list, diff_threshold: float):
    if fs_scene_value_list[0] - fs_scene_value_list[1] < diff_threshold:
        return True
    else:
        return False


def parse_tf_to_np(frame_data, fea_cfg):
    np_data = defaultdict()

    # for fea, data in frame_data.items():
    # if fea in fea_cfg:
    #     np_data[fea] = data.values.numpy().reshape(fea_cfg[fea].shape)

    # TODO 临时解析
    np_data["frame_idx"] = frame_data["frame_id_perception_obs"].values.numpy().item()
    np_data["cost_result_shape"] = frame_data["cost_result_shape"].values.numpy().reshape(3)
    np_data["cost_result"] = frame_data["cost_result"].values.numpy().reshape(np_data["cost_result_shape"])
    np_data["cost_result_values"] = frame_data["cost_result_values"].values.numpy()
    np_data["cost_agent_id"] = frame_data["cost_agent_id"].values.numpy().reshape(np_data["cost_result_shape"][0], -1)

    np_data["fs_scene_shape"] = frame_data["fs_scene_shape"].values.numpy()
    np_data["fs_scene"] = frame_data["fs_scene"].values.numpy().reshape(np_data["fs_scene_shape"])
    np_data["fs_agent_idx"] = frame_data["fs_agent_idx"].values.numpy().reshape(np_data["fs_scene_shape"][0], -1)
    np_data["scene_behavior_labels"] = frame_data["scene_behavior_labels"].values.numpy().reshape(np_data["cost_result_shape"][0], 2)
    np_data["lateral_distance_labels"] = frame_data["lateral_distance_labels"].values.numpy()
    return np_data


def solve_cost_scene_names_bug(frame_data, cost_fs_scene_names):
    fs_scene_names_v = None
    try:
        fs_scene_names_v = frame_data["fs_scene_names"].values[0].numpy().decode("utf-8")
    except UnicodeDecodeError:
        fs_scene_names_v = frame_data["fs_scene_names"].values[0].numpy().decode("ISO-8859-1")
    fs_scene_names = fs_scene_names_v.split(",")[:-1]
    if len(cost_fs_scene_names) > len(fs_scene_names):
        cost_fs_scene_names = fs_scene_names
    return cost_fs_scene_names


def split_np_array(input_data: dict, split_index: int):
    splited_data = defaultdict()
    remain_data = defaultdict()
    for key, val in input_data.items():
        splited_data[key] = val[:split_index, ...]
        remain_data[key] = val[split_index:, ...]
    return splited_data, remain_data


def merge_np_array(input_data: dict, merged_data: dict) -> None:
    """把input_data合并到merged_data中

    Parameters
    ----------
    input_data : dict
        input data 必须保证长度>0
    merged_data : dict

    """
    if len(input_data) != 0:
        for key, val in input_data.items():
            # 如果 val 没有维度，则直接赋值
            if len(val.shape) == 0:
                merged_data[key] = val
                continue

            if key not in merged_data:
                merged_data[key] = val
            else:
                temp_data = merged_data[key]
                merged_data[key] = np.concatenate([temp_data, val], axis=0)


def merge_multi_array(input_data: List[Dict[str, np.ndarray]]) -> defaultdict:
    """合并多个array;

    Parameters
    ----------
    input_data : List[Dict[str, np.ndarray]]
        输入数据

    Returns
    -------
    dict
        合并后的字典
    """
    # merge
    merged_data = defaultdict(list)
    for fea_name in input_data[0].keys():
        temp_data = []
        for case_data in input_data:
            temp_data.append(case_data[fea_name])
        merged_data[fea_name] = np.concatenate(temp_data, axis=0)

    return merged_data


def shuffle_data(input_data: dict) -> dict:
    shuffled_data = defaultdict()
    if len(input_data) != 0:
        temp_fea = list(input_data.keys())[0]
        data_size = input_data[temp_fea].shape[0]
        shuffle_frame_indices = np.random.permutation(data_size)
        for fea_name, np_val in input_data.items():
            shuffled_data[fea_name] = np_val[shuffle_frame_indices]
        del input_data
    return shuffled_data


def stack_frame_data(
        input_data: Dict[str, List[np.ndarray]],
        output_data: Dict[str, np.ndarray],
):
    # if "env_data" in self.fea_cfg.features:
    #     output_data["env_data"] = defaultdict()
    for fea_key in input_data.keys():
        if fea_key == "env_label_info":
            output_data["env_label_info"] = input_data[fea_key]
        try:
            # res = [i for i, item in enumerate(temp_data_list[fea_key]) if item.shape[0] != 3]
            output_data[fea_key] = np.stack(input_data[fea_key])
        except ValueError:
            import pdb;
            pdb.set_trace()


def select_data_by_indices(
        input_data: Dict[str, np.ndarray],
        selected_indices: List[np.ndarray]
) -> Dict[str, np.ndarray]:
    """对 ndarray 的第一个维度 select data

    Parameters
    ----------
    input_data : Dict[str, np.ndarray]
        input data
    selected_indices : List[np.ndarray]
        indices

    Returns
    -------
    Dict[str, np.ndarray]
        res_data
    """
    assert len(selected_indices) > 0, "the size of selected_indices is zero."

    res_data = defaultdict()
    for key, value in input_data.items():
        # 如果 shape 没有第一个维度，则直接赋值
        if len(value.shape) == 0:
            res_data[key] = value
            continue

        temp_list = []
        for indices in selected_indices:
            temp_list.append(value[indices])

        if len(temp_list) == 1:
            res_data[key] = temp_list[0]
        else:
            res_data[key] = np.concatenate(temp_list, axis=0)

        del temp_list
    return res_data
