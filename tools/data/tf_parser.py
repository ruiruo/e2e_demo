from collections import defaultdict
import tensorflow as tf
import numpy as np
from unpark.file_utils import parse_tfrecord, parse_tfrecord_infer_mode, nested_dict, set_memory_growth
from unpark.constants import RED, RESET
from unpark.base_module import DataProcessModule


class TFParser(DataProcessModule):
    def __init__(self,
                 cfg: dict,
                 func_map: dict,
                 fea_cfg: dict,
                 logger_map: dict,
                 **kwargs,
                 ):
        super().__init__(cfg, func_map, fea_cfg, logger_map, **kwargs)
        self.tf_parser_cfg = cfg.tf_parser
        self.label_match_mode = cfg.data.get("label_match_mode", False)
        self._check_func_map()

    def _check_func_map(self, ):
        for fea_name, func in self.func_map.items():
            if fea_name not in self.fea_cfg.features:
                self.info(f"Warning: {RED}fea `{fea_name}` not in fea_cfg.{RESET}")
                # raise ValueError

    def convert_to_np(self,
                      case_data_tf,
                      parsed_case_data
                      ):
        try:
            for index, frame_data_tf in enumerate(case_data_tf):
                if not self.is_valid_frame(frame_data_tf):
                    continue
                if len(frame_data_tf["fs_scenes_traj_info_shape"].values.numpy()) == 0:
                    print("fea len = 0")
                    continue
                point_tag = frame_data_tf["point_tag"].values.numpy()
                is_valid, window_size, frame_id = point_tag
                ego_history_feature_shape = frame_data_tf["ego_history_feature_shape"].values.numpy().astype(np.int32)
                ego_history_feature = frame_data_tf["ego_history_feature"].values.numpy().reshape(ego_history_feature_shape)
                parsed_case_data[frame_id]["ego_history_feature"] = ego_history_feature

        except tf.errors.DataLossError as e:
            print(e)
            pass
        # clean TensorFlow
        tf.keras.backend.clear_session()

    def tf_tensor_is_none(self, fea_name, tf_tensor_map):
        if fea_name in tf_tensor_map and tf_tensor_map[fea_name].values.numpy().size > 0:
            return False
        else:
            return True

    def convert_to_np_infer_mode(self,
                                 case_data_tf,
                                 parsed_case_data
                                 ):
        try:
            for index, frame_data_tf in enumerate(case_data_tf):
                point_tag = frame_data_tf["point_tag"].values.numpy()
                frame_id = point_tag[0]


        except tf.errors.DataLossError as e:
            pass
        # 清理 TensorFlow 会话
        tf.keras.backend.clear_session()

    def __call__(self, record_path: str, worker_i: int = -1) -> dict:
        """解析 tf record

        Parameters
        ----------
        record_path : str
            record path
        worker_i: int
            进程id，如果未使用多进程，默认值为-1
        Returns
        -------
        parsed_case_data
            解析后的数据, example:
            {
                frame_i: {
                    fea_name: val,
                    ...,
                },
                frame_i: ,
                ...
            }
        """
        set_memory_growth()

        case_data_tf = parse_tfrecord(record_path, self.fea_cfg, label_match_mode=self.label_match_mode)
        parsed_case_data = nested_dict()

        self.convert_to_np(case_data_tf, parsed_case_data)

        build_case_data = self.build_scene(parsed_case_data)

        del parsed_case_data, case_data_tf
        return build_case_data if build_case_data else {}

    def parse_infer_mode(self, record_path: str, worker_i: int = -1) -> dict:
        set_memory_growth()

        case_data_tf = parse_tfrecord_infer_mode(record_path, self.fea_cfg)
        parsed_case_data = nested_dict()

        self.convert_to_np_infer_mode(case_data_tf, parsed_case_data)
        del case_data_tf
        return parsed_case_data

    def build_per_frame_(self, frame_id: int, frame_data: dict, target_frame_data: dict):
        """构建单帧数据

        Parameters
        ----------
        frame_id: int
            帧id
        frame_data : dict
            帧数据
        target_frame_data : dict
            打标后的帧数据
        """

        # 自定义需要处理的特征
        self.flow_func_(
            need_keys=self.fea_cfg.features,
            frame_id=frame_id,
            frame_data=frame_data,
            target_data=target_frame_data,
            fea_cfg=self.fea_cfg
        )

    def is_valid_frame(self, frame_data_tf):

        point_tag = frame_data_tf["point_tag"].values.numpy()
        # print(point_tag[-1])
        return point_tag[0]

    def info(self, msg: str, worker_i: int = -1):
        """日志

        Parameters
        ----------
        msg : str
            消息
        worker_i : int
            进程id
        """
        super().info(f"[TFParser] {msg}", worker_i)
