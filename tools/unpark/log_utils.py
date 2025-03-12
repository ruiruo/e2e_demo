import os
import time
import shutil
import logging
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, List


def create_logger(log_dir=None, name="train", rank=0, log_level=logging.INFO):
    log_file = os.path.join(log_dir, f"log_{name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt")
    # import pdb;pdb.set_trace()
    logger = logging.getLogger(name)
    logger.setLevel(log_level if rank == 0 else logging.ERROR)
    formatter = logging.Formatter("%(asctime)s  %(levelname)5s  %(message)s")
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else logging.ERROR)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(filename=log_file, encoding="utf-8")
        file_handler.setLevel(log_level if rank == 0 else logging.ERROR)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def generate_log_dirs(base_dir, args):
    dirs = {}

    dirs["base_dir"] = os.path.join(base_dir, "outs", "experiments", args.common.task_name)
    dirs["log_dir"] = os.path.join(dirs["base_dir"], "logs")

    if args.mode in ("train", "finetune"):
        dirs["tb_dir"] = os.path.join(dirs["base_dir"], "tensorboards", args.mode)
        dirs["ckpt_dir"] = os.path.join(dirs["base_dir"], "checkpoints", args.mode)
    else:
        dirs["val_dir"] = os.path.join(dirs["base_dir"], "validations", datetime.now().strftime("%Y%m%d-%H%M%S"))

    args.common.update(dirs)

    if args.common.global_rank == 0:
        for v in dirs.values():
            os.makedirs(v, exist_ok=True)
        shutil.copy(args.config_path, os.path.join(args.common.base_dir, f"user_{args.mode}_config.yaml"))
    else:
        time.sleep(0.2)
    return args


class CombinedAverageMeter(object):
    def __init__(self, max_size=1):
        self.max_size = max_size
        self.reset()

    def reset(self):
        self.global_sum = 0
        self.global_count = 0
        self.rolling_values = deque(maxlen=self.max_size)
        self.rolling_sum = 0

    def update(self, val, n=1):
        # 更新全局平均
        self.global_sum += val * n
        self.global_count += n
        # 更新滚动平均
        if len(self.rolling_values) == self.max_size:
            self.rolling_sum -= self.rolling_values[0]
        self.rolling_values.append(val)
        self.rolling_sum += val

    def get_log_v(self, method):
        if method == "rolling_avg":
            return self.rolling_avg
        elif method == "global_avg":
            return self.global_avg
        elif method == "global_sum":
            return self.global_sum
        else:
            raise ValueError(f"method {method} is not supported")

    @property
    def global_avg(self):
        return self.global_sum / self.global_count if self.global_count else 0

    @property
    def rolling_avg(self):
        return self.rolling_sum / len(self.rolling_values) if self.rolling_values else 0


def convert_stats_to_str_list_of_dict(log_info_dict: Dict[str, List[int]], log_info_str: str = ""):
    for key, value_list in log_info_dict.items():
        if key == "case_id":
            continue
        if isinstance(value_list, list):
            total_val = sum(value_list)
        # elif isinstance(value, str):
        # stats_info_res[key] = value
        #     continue
        log_info_str += f"{key}={total_val}, "

    return log_info_str


def sum_stats_list_of_dict(
    log_info_dict: Dict[str, List[int]],
):
    res = defaultdict()
    for key, value_list in log_info_dict.items():
        if key == "case_id":
            continue
        if isinstance(value_list, list):
            total_val = sum(value_list)
            res[f"total_{key}"] = total_val

    return res


def convert_stats_to_str(log_info_dict: Dict[str, int], log_info_str: str = ""):
    for key, val in log_info_dict.items():
        log_info_str += f"{key}={val}, "

    return log_info_str


def print_stats_list_of_dict(
    log_info_dict: Dict[str, List[int]],
    need_keys: List[str],
) -> str:
    """打印当前case的stats info

    Parameters
    ----------
    log_info_dict : Dict[str, List[int]]
        记录了每个case的stats info
    need_keys : List[str]
        需要打印的key

    Returns
    -------
    str
        log_info_str
    """
    log_info_str = ""
    for key in need_keys:
        if key in log_info_dict:
            val = log_info_dict[key][-1]
            log_info_str += f"{key}={val}, "
    # for key, val_list in log_info_dict.items():

    return log_info_str


def print_stats_int_of_dict(
    log_info_dict: Dict[str, int],
    need_keys: List[str],
) -> str:
    """打印当前case的stats info

    Parameters
    ----------
    log_info_dict : Dict[str, int]
        记录了每个case的stats info
    need_keys : List[str]
        需要打印的key

    Returns
    -------
    str
        log_info_str
    """
    log_info_str = ""
    for key in need_keys:
        if key in log_info_dict:
            val = log_info_dict[key]
            log_info_str += f"{key}={val}, "
    # for key, val_list in log_info_dict.items():

    return log_info_str


def merge_stats_list_of_dict(input_stat_dict: Dict[str, list], target_stat_dict: Dict[str, list]):
    for key, val_list in input_stat_dict.items():
        target_stat_dict[key].extend(val_list)


def merge_stats_int_of_dict(input_stat_dict: Dict[str, int], target_stat_dict: Dict[str, list]):
    for key, val in input_stat_dict.items():
        target_stat_dict[key].append(val)
