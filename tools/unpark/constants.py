import random
from collections import defaultdict

COST_FEATURES_NAME = [
    "simulate_result_size",
    "total_cost",
    "raw_efficiency_cost",
    "raw_comfort_cost",
    "raw_safety_cost",
    "raw_navigation_cost",
    "post_cost",  # top level
    "nudge_safety",  # nudge
    "v_end",
    "v_front",
    "s_end",
    "block_s",  # effic
    "max_l",
    "max_offset",
    "max_deltheta",
    "end_l",
    "refpath_diff",
    "reflane_diff",
    "max_expand",
    "lat_acc",
    "lon_acc",
    "lon_jerk",
    "refpath_distance",
    "max_kappa",  # comfort
    "navi_priority",
    "prefer_lane_cost",  # navigation
    "lc_task_cost",
    "is_squeeze_in",
    "lc_task_type",
    "hold_time_cost",  # Post cost
]

USED_COST_FEA_INDICES = [
    COST_FEATURES_NAME.index("raw_navigation_cost") - 1,
    COST_FEATURES_NAME.index("navi_priority") - 1,
    COST_FEATURES_NAME.index("prefer_lane_cost") - 1,
]

RAW_FEATURES_NAME = [
    "vec_position_x",
    "vec_position_y",
    "angle",
    "velocity",
    "acceleration",
    "curvature",
    "curvature_rate",
    "steer",
    "is_virtual",
    "s_distance",
    "angle_rate",
]

RELATIVE_FEATURES_NAME = [
    "delta_vec_position_x",
    "delta_vec_position_y",
    "delta_angle",
    "delta_velocity",
    "delta_acceleration",
    "delta_curvature",
]

USED_RAW_FEATURES_SIZE = 4

USED_RELATIVE_FEATURES_SIZE = 2

RAW_FEA_NUM = len(RAW_FEATURES_NAME)

ENV_FEATURES = [
    "lane_change_type",
    "lane_change_status",
    "junction_is_split",
    "junction_s_end",
    "route_s",
]

MAX_LANE_SIZE = 4
# NAVI_FEATURES = [
#     *[f"navi_travelable_{lane_i}" for lane_i in range(MAX_LANE_SIZE)],
#     *[f"navi_turn_{lane_i}" for lane_i in range(MAX_LANE_SIZE)],
#     "navi_distance_to_intersection",
# ]
NAVI_FEATURES = [
    "navi_travelable",
    "navi_turn",
]

REF_LANE_SIZE = 3
# REF_LINE_FEATURES = [
#     *[f"ref_line_{lane_i}" for lane_i in range(REF_LANE_SIZE)],
# ]
REF_LINE_FEATURES = ["ref_line"]

USED_ENV_FEATURES_SIZE = len(ENV_FEATURES) + len(NAVI_FEATURES) * MAX_LANE_SIZE + 1 + len(REF_LINE_FEATURES) * REF_LANE_SIZE

TRAJ_FEATURES = ["x", "y"]

BEHAVIOR_LABELS = ["lateral_behavior", ""]  # 横向  # 纵向

random.seed(2024)


def generate_random_colors(n):
    colors = set()
    while len(colors) < n:
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        colors.add(color)
    return list(colors)


COLORS = ["#000000"] + generate_random_colors(64)

BEHAVIORS = ["K", "L", "R", "H", "P", "O", "RR", "LL"]

HOLD_INDEX = BEHAVIORS.index("H")
NUDGE_INDEDX = BEHAVIORS.index("O")
LEFT_LC_INDEX = [1, 7]
RIGHT_LC_INDEX = [2, 6]

# key: behaviors index; value: reward values
GT_BEHAVIOR_VALUE_MAP = {
    0: [2, 0, 0, 2, 2, 2, 0, 0],  # gt = K
    1: [1, 2, 0, 1, 1, 1, 0, 1],  # gt = L
    2: [1, 0, 2, 1, 1, 1, 1, 0],  # gt = R
    3: [2, 0, 0, 2, 2, 2, 0, 0],
    4: [2, 0, 0, 2, 2, 2, 0, 0],
    5: [2, 0, 0, 2, 2, 2, 0, 0],
    6: [0, 0, 1, 0, 0, 0, 2, 0],  # gt = RR
    7: [0, 1, 0, 0, 0, 0, 0, 2],  # gt = LL
}


def label_int_2_str(label_index: int):
    if label_index >= len(BEHAVIORS):
        return "Error"
    elif label_index < -1:
        return "None"
    else:
        return BEHAVIORS[label_index]


SCENE_STATS_METRIC = [
    # "lane_change_status", "lane_change_type",
    # "gt_behavior_y", "gt_behavior_x",
    "fs_scene_num_per_frame",
    "gen_scene_num_per_frame",
    "status_align_gt",
]

YELLOW = "\033[33m"
GREEN = "\033[32m"
RESET = "\033[0m"
RED = "\033[31m"

REPEATE_SIZE_TEST = 30  # 60 # 30

MAX_AGENT_SIZE = 10
USED_AGENT_SIZE = 5  # 10 # 5


HISTORY_STATE_SIZE = 25
FUTURE_STATE_SIZE = 35

MAX_STATE_SIZE = HISTORY_STATE_SIZE + FUTURE_STATE_SIZE
USED_STATE_SIZE = FUTURE_STATE_SIZE

USED_TRAJ_FEA_SIZE = USED_RAW_FEATURES_SIZE + USED_RELATIVE_FEATURES_SIZE
# USED_TRAJ_FEA_SIZE = USED_RAW_FEATURES_SIZE

# INPUT_CHANNELS = 35 * 2 * 8
# INPUT_CHANNELS = 35 * 2 * 4
INPUT_CHANNELS = USED_STATE_SIZE * USED_TRAJ_FEA_SIZE * MAX_AGENT_SIZE + USED_ENV_FEATURES_SIZE
# INPUT_CHANNELS = 35 * 2 * 1
# INPUT_CHANNELS = 52

SAMPLE_TIMES = 1


INPUT_DATA_2_FILENAME_MAP = {
    "cost_data": "input",
    "traj_data": "traj",
    "env_data": "env",
    "behavior_values": "target",
    "behavior_category": "behavior_category",
    "rule_values": "rule_value",
    "traj_labels": "traj_labels",
    "lateral_distance_labels": "lateral_distance_labels",
    "is_valid_scene": "is_valid_scene",
    "is_gt_scene": "is_gt_scene",
}

SCENE_NUM_IN_PAIR = 4


def cal_label_value_by_gt_behavior(gt_behavior_index: int):

    return GT_BEHAVIOR_VALUE_MAP[gt_behavior_index]


INFER_FEA_INDICES_MAP = defaultdict()
INFER_FEA_INDICES_MAP["traj_data"] = range(USED_TRAJ_FEA_SIZE)
INFER_FEA_INDICES_MAP["env_data"] = range(2, USED_ENV_FEATURES_SIZE)
# print("env_fea_size", USED_ENV_FEATURES_SIZE)

SCENE_CONDITION_MAP = {
    "fs_scene": lambda scene_name: True if scene_name.endswith("fs") else False,
    "gen_scene": lambda scene_name: "fs_gt_fs_ego" in scene_name,
}

TASK_STATUS = {"success": True, "fail": False}


CUR_TRAJ_SIZE = 1

GT_SCENE_INDEX = 0
LIKELY_GT_SCENE_INDEX = 1
EGO_INDEX = 0

epsilon = 1e-3

Lane_Change_Task_Trigger_Type = {
  0: "DRIVER",
  1: "EMERGENCY_AVOIDANCE",
  2: "SPLIT",
  3: "OFF_RAMP",
  4: "LANE_KEEP_SPLIT",
  5: "RAMP_TO_ROAD",
  6: "PRIORITY",
  7: "MERGE",
  8: "AVOID_CONSTRUCTION_ZONE",
  9: "PREFERED_LANE",
  10: "SPEED",
  11: "AVOID_MERGE_LANE",
  12: "RISK_OBSTACLE",
  13: "OVERTAKE",
  14: "RESERVED",
  15: "AVOID_STATIC_VEHICLE",
  16: "AVOID_VEHICLE_IN_JUNCTION",
  17: "AVOID_BUS_LANE",
  18: "AVOID_NON_MOTOR_LANE",
  99: "NONE"
}

kRefineLabelThreshold = 0.50

LonDistanceErrorThreshold = 7
LatDistanceErrorThreshold = 0.8

Traj_Source_Tags = ["DAS", "", "", "ASTRA"]