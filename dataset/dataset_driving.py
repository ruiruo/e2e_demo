import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data
import tqdm

from utils.config import Configuration
from utils.trajectory_utils import TrajectoryInfoParser, tokenize_traj_point


class Topology:
    """
    封装的原始拓扑信息
    """

    def __init__(self, frame_id: int, feature: dict):
        self.frame_id = frame_id
        self.ego_history = feature.get('ego_history_feature')
        self.ego_info = self.preprocess()
        self.agent_feature = AgentFeature(feature)

    def preprocess(self) -> np.ndarray:
        """
        预处理拓扑信息，将 ego_history 数据转换为 ego_info，格式为 (x, y, heading, v, acc)
        """
        # 提取所需的列
        # TODO: 帧间时间间隔不稳定, 会在100ms和200ms间变化, 是采用抽帧还是直接裁切
        ego_info = self.ego_history[:, 2:7].copy()
        # 将自车起始位置对齐到 (0, 0)
        mask = ~np.any(ego_info[:, :2] == -300, axis=1)
        ego_info[mask, :2] -= ego_info[-1, :2]
        return ego_info


class AgentFeature:
    """
    障碍物特征类，通过自车信息预处理障碍物拓扑信息。

    输入的 feature 字典需要包含以下键：
      - 'ego_history_feature': 用于确定自车（ego）位置偏置的历史轨迹（数组）
      - 'agent_feature': 障碍物的时空特征（数组）,包含自车信息(id:0)
      - 'agent_attribute_feature': 障碍物的属性特征（数组）
    """

    def __init__(self, feature: dict):
        self.pos_bias = feature['ego_history_feature'][-1, 2:4]
        self.agent = feature['agent_feature']
        self.agent_attribute = feature['agent_attribute_feature']
        self.agent_feature = self.preprocess()

    def preprocess(self) -> np.ndarray:
        """
        预处理障碍物拓扑信息，生成格式为：
            (id, x, y, heading, v, acc, length, width, abs_dis, hit_dis)
        """
        # 1. 提取所需列并对齐自车位置
        agent_info = self.agent[:, :, 1:7].copy()
        agent_attr = self.agent_attribute[:, :3]
        mask = ~np.any(agent_info[:, :, 1:3] == -300, axis=2)
        agent_info[mask, 1:3] -= self.pos_bias

        # 2. 拼接障碍物属性
        new_agent_info = self._concatenate_agent_attributes(agent_info, agent_attr)

        # 3. 计算绝对距离并添加到特征中
        new_agent_info = self._append_absolute_distance(new_agent_info)

        # 4. 计算碰撞盒（bounding box）之间的最小距离并添加到特征中
        new_agent_info = self._append_min_polygon_distance(new_agent_info)

        # 转置维度：从 (agent, time, feature) 转换为 (time, agent, feature)
        return np.transpose(new_agent_info, (1, 0, 2))

    def _concatenate_agent_attributes(self, agent_info: np.ndarray, agent_attr: np.ndarray) -> np.ndarray:
        """
        将障碍物的额外属性拼接到 agent_info 上。

        构建字典 id_to_data，将 agent_attr 每行的第一个元素作为 key，
        剩余部分作为 value，然后逐个遍历 agent_info 并拼接对应属性。
        """
        id_to_data = {row[0]: row[1:] for row in agent_attr}
        new_last_dim = agent_info.shape[2] + agent_attr.shape[1] - 1
        new_agent_info = np.full((agent_info.shape[0], agent_info.shape[1], new_last_dim), -300.)
        for i in range(agent_info.shape[0]):
            for j in range(agent_info.shape[1]):
                current_id = agent_info[i, j, 0]
                if current_id == -300:
                    continue
                if current_id not in id_to_data:
                    raise ValueError(f"agent_attribute中未找到 id: {current_id}")
                extra_data = id_to_data[current_id]
                new_agent_info[i, j] = np.concatenate([agent_info[i, j], extra_data])
        return new_agent_info

    def _append_absolute_distance(self, data: np.ndarray) -> np.ndarray:
        """
        计算每个障碍物与 ego 之间的绝对距离，并将其作为新的特征添加。
        """
        ids = data[:, 0, 0]
        ego_idx_arr = np.where(ids == 0)[0]
        if ego_idx_arr.size == 0:
            raise ValueError("没有找到ego")
        idx_ego = ego_idx_arr[0]

        x0 = data[idx_ego, :, 1]
        y0 = data[idx_ego, :, 2]
        distances = np.sqrt((data[:, :, 1] - x0[None, :]) ** 2 + (data[:, :, 2] - y0[None, :]) ** 2)
        distances = distances[:, :, None]

        invalid_mask = np.any(data[:, :, 1:3] == -300, axis=2)
        distances[invalid_mask] = -300

        return np.concatenate([data, distances], axis=2)

    def _append_min_polygon_distance(self, data: np.ndarray) -> np.ndarray:
        """
        计算每个障碍物与 ego 的碰撞盒（bounding box）之间的最小距离，并将该值作为新的特征添加。
        """
        ids = data[:, 0, 0]
        ego_idx_arr = np.where(ids == 0)[0]
        if ego_idx_arr.size == 0:
            raise ValueError("计算碰撞盒距离没有找到ego信息")
        idx_ego = ego_idx_arr[0]

        num_agents, T, _ = data.shape
        min_dists = np.full((num_agents, T), -300.)
        for obj in range(num_agents):
            for t in range(T):
                if (data[idx_ego, t, 1] == -300 or data[idx_ego, t, 2] == -300 or
                        data[obj, t, 1] == -300 or data[obj, t, 2] == -300):
                    continue

                ego_x, ego_y = data[idx_ego, t, 1], data[idx_ego, t, 2]
                ego_theta, ego_L, ego_W = data[idx_ego, t, 3], data[idx_ego, t, 6], data[idx_ego, t, 7]
                bbox_ego = self._get_bbox(ego_x, ego_y, ego_theta, ego_L, ego_W, True)

                obj_x, obj_y = data[obj, t, 1], data[obj, t, 2]
                obj_theta, obj_L, obj_W = data[obj, t, 3], data[obj, t, 6], data[obj, t, 7]
                bbox_obj = self._get_bbox(obj_x, obj_y, obj_theta, obj_L, obj_W)

                d = self._min_distance_between_polygons(bbox_ego, bbox_obj)
                min_dists[obj, t] = d

        min_dists = min_dists[:, :, None]
        return np.concatenate([data, min_dists], axis=2)

    @staticmethod
    def _get_bbox(x: float, y: float, theta: float, L: float, W: float, is_ego=False) -> np.ndarray:
        """
        根据中心 (x, y)、朝向 theta（弧度）、长度 L 和宽度 W，
        计算旋转矩形的四个角点，返回形状为 (4, 2) 的数组。

        若 is_ego 为 True，则车辆前部取 L 的 3/4，后部取 1/4；
        否则，车辆前后各取 L/2。
        """
        if is_ego:
            front = L * 3 / 4
            back = -L * 1 / 4
        else:
            front = L / 2
            back = -L / 2

        corners = np.array([
            [front, W / 2],
            [front, -W / 2],
            [back, -W / 2],
            [back, W / 2]
        ])
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        rotated = corners @ R.T
        return rotated + np.array([x, y])

    @staticmethod
    def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """
        计算二维点 p 到线段 ab 的最短距离。
        """
        v = b - a
        w = p - a
        c1 = np.dot(w, v)
        if c1 <= 0:
            return np.linalg.norm(p - a)
        c2 = np.dot(v, v)
        if c2 <= c1:
            return np.linalg.norm(p - b)
        b_ratio = c1 / c2
        pb = a + b_ratio * v
        return np.linalg.norm(p - pb)

    @classmethod
    def _min_distance_between_polygons(cls, poly1: np.ndarray, poly2: np.ndarray) -> float:
        """
        计算两个旋转矩形（由四个角点构成的多边形）之间的最小距离。

        遍历 poly1 每个点到 poly2 各边的距离，以及 poly2 每个点到 poly1 各边的距离，返回两者之间的最小值。
        如果两个多边形相交，理论上最小距离为 0。
        """
        min_dist = float('inf')
        n1, n2 = len(poly1), len(poly2)
        # 检查 poly1 的每个点到 poly2 各边的距离
        for i in range(n2):
            a = poly2[i]
            b = poly2[(i + 1) % n2]
            for p in poly1:
                d = cls._point_to_segment_distance(p, a, b)
                if d < min_dist:
                    min_dist = d
        # 检查 poly2 的每个点到 poly1 各边的距离
        for i in range(n1):
            a = poly1[i]
            b = poly1[(i + 1) % n1]
            for p in poly2:
                d = cls._point_to_segment_distance(p, a, b)
                if d < min_dist:
                    min_dist = d
        return min_dist


class TrajectoryDataModule(torch.utils.data.Dataset):
    def __init__(self, config: Configuration, is_train):
        super(TrajectoryDataModule, self).__init__()
        self.cfg = config

        self.BOS_token = self.cfg.token_nums
        self.EOS_token = self.cfg.token_nums + self.cfg.append_token - 2
        self.PAD_token = self.cfg.token_nums + self.cfg.append_token - 1

        self.root_dir = self.cfg.data_dir
        self.is_train = is_train

        self.feature = {}
        self.task_index_list = []

        self.fuzzy_target_point = []
        self.traj_point = []
        self.traj_point_token = []
        self.target_point = []
        # 保存每个句子中各个原始拓扑信息，每个拓扑将由 Topology 对象封装
        self.raw_topologies = []

        self.create_gt_data()

    def __len__(self):
        # 这里假设 feature 中某个字段的长度代表数据集大小
        return len(self.feature.get("", []))

    def __getitem__(self, index):
        data = {}
        keys = ['intrinsics', 'extrinsics', 'target_point', 'gt_traj_point', 'gt_traj_topology', 'fuzzy_target_point']
        for key in keys:
            data[key] = []

        # 若需要摄像头参数，可以从 self.intrinsic 与 self.extrinsic 中取；否则置空
        data['intrinsics'] = None
        data['extrinsics'] = None

        data["gt_traj_point"] = torch.from_numpy(np.array(self.traj_point[index]))
        # 依次调用每个 Topology 对象的 preprocess 方法，并转为 tensor
        processed_topology = [torch.from_numpy(topo.preprocess()) for topo in self.raw_topologies[index]]
        data['gt_traj_topology'] = processed_topology
        data['target_point'] = torch.from_numpy(self.target_point[index])
        data["fuzzy_target_point"] = torch.from_numpy(self.fuzzy_target_point[index])

        return data

    def process_record(self, record_path):
        _ = self.cfg
        return None

    def process_csv(self, csv_path):
        current_case = pd.read_csv(csv_path)

    def create_gt_data(self):
        all_tasks = self.get_all_tasks()

        for task_index, task_path in tqdm.tqdm(enumerate(all_tasks)):
            # task iteration
            traje_info_obj = TrajectoryInfoParser(task_index, task_path)

            # 若不存在，则初始化 intrinsic 与 extrinsic 属性
            if not hasattr(self, 'intrinsic'):
                self.intrinsic = {}
            if not hasattr(self, 'extrinsic'):
                self.extrinsic = {}

            for ego_index in range(traje_info_obj.total_frames):
                ego_pose = traje_info_obj.get_trajectory_point(ego_index)
                world2ego_mat = ego_pose.get_homogeneous_transformation().get_inverse_matrix()
                # 生成预测轨迹点（token 化和原始数值）
                predict_point_token_gt, predict_point_gt = self.create_predict_point_gt(traje_info_obj, ego_index, world2ego_mat)
                # 生成停车目标
                fuzzy_parking_goal, parking_goal = self.create_parking_goal_gt(traje_info_obj, world2ego_mat)

                self.traj_point.append(predict_point_gt)
                self.traj_point_token.append(predict_point_token_gt)
                self.target_point.append(parking_goal)
                self.fuzzy_target_point.append(fuzzy_parking_goal)
                # 为每个 token 预留原始拓扑信息，使用 Topology 对象封装，当前以零矩阵作为占位
                topology_list = [Topology(np.zeros((200, 8), dtype=np.float32)) for _ in range(len(predict_point_token_gt))]
                self.raw_topologies.append(topology_list)
                self.task_index_list.append(task_index)

        self.format_transform()

    def create_predict_point_gt(self, traje_info_obj: TrajectoryInfoParser, ego_index: int,
                                world2ego_mat: np.array) -> list:
        predict_point, predict_point_token = [], []
        for predict_index in range(self.cfg.autoregressive_points):
            predict_stride_index = self.get_clip_stride_index(
                predict_index=predict_index,
                start_index=ego_index,
                max_index=traje_info_obj.total_frames - 1,
                stride=self.cfg.traj_downsample_stride
            )
            predict_pose_in_world = traje_info_obj.get_trajectory_point(predict_stride_index)
            predict_pose_in_ego = predict_pose_in_world.get_pose_in_ego(world2ego_mat)
            progress = traje_info_obj.get_progress(predict_stride_index)
            predict_point.append([predict_pose_in_ego.x, predict_pose_in_ego.y])
            tokenize_ret = tokenize_traj_point(predict_pose_in_ego.x, predict_pose_in_ego.y,
                                               progress, self.cfg.token_nums, self.cfg.xy_max)
            tokenize_ret_process = tokenize_ret[:2] if self.cfg.item_number == 2 else tokenize_ret
            predict_point_token.append(tokenize_ret_process)

            if predict_stride_index == traje_info_obj.total_frames - 1 or predict_index == self.cfg.autoregressive_points - 1:
                break

        predict_point_gt = [item for sublist in predict_point for item in sublist]
        append_pad_num = self.cfg.autoregressive_points * self.cfg.item_number - len(predict_point_gt)
        assert append_pad_num >= 0
        if len(predict_point_gt) > 0:
            predict_point_gt = predict_point_gt + (append_pad_num // 2) * [predict_point_gt[-2], predict_point_gt[-1]]
        predict_point_token_gt = [item for sublist in predict_point_token for item in sublist]
        predict_point_token_gt.insert(0, self.BOS_token)
        predict_point_token_gt.append(self.EOS_token)
        predict_point_token_gt.append(self.PAD_token)
        append_pad_num = self.cfg.autoregressive_points * self.cfg.item_number + self.cfg.append_token - len(predict_point_token_gt)
        assert append_pad_num >= 0
        predict_point_token_gt = predict_point_token_gt + append_pad_num * [self.PAD_token]
        return predict_point_token_gt, predict_point_gt

    def create_parking_goal_gt(self, traje_info_obj: TrajectoryInfoParser, world2ego_mat: np.array):
        candidate_target_pose_in_world = traje_info_obj.get_random_candidate_target_pose()
        candidate_target_pose_in_ego = candidate_target_pose_in_world.get_pose_in_ego(world2ego_mat)
        fuzzy_parking_goal = [candidate_target_pose_in_ego.x, candidate_target_pose_in_ego.y]

        target_pose_in_world = traje_info_obj.get_precise_target_pose()
        target_pose_in_ego = target_pose_in_world.get_pose_in_ego(world2ego_mat)
        parking_goal = [target_pose_in_ego.x, target_pose_in_ego.y]

        return fuzzy_parking_goal, parking_goal

    def get_all_tasks(self):
        all_tasks = []
        train_data_dir = os.path.join(self.root_dir, self.cfg.training_dir)
        val_data_dir = os.path.join(self.root_dir, self.cfg.validation_dir)
        data_dir = train_data_dir if self.is_train == 1 else val_data_dir
        for scene_item in os.listdir(data_dir):
            scene_path = os.path.join(data_dir, scene_item)
            for task_item in os.listdir(scene_path):
                task_path = os.path.join(scene_path, task_item)
                all_tasks.append(task_path)
        return all_tasks

    def format_transform(self):
        self.traj_point = np.array(self.traj_point).astype(np.float32)
        self.traj_point_token = np.array(self.traj_point_token).astype(np.int64)
        self.target_point = np.array(self.target_point).astype(np.float32)
        self.fuzzy_target_point = np.array(self.fuzzy_target_point).astype(np.float32)
        self.task_index_list = np.array(self.task_index_list).astype(np.int64)

    def get_clip_stride_index(self, predict_index, start_index, max_index, stride):
        return int(np.clip(start_index + stride * (1 + predict_index), 0, max_index))
