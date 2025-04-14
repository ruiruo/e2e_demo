try:
    from read_cache_op_py import read_cache_op_py as rcop
except ImportError:
    print("import read_cache_op_py failed.")
try:
    import niofs
    from niofs.conf import Cloud
except ImportError:
    print("import niofs failed.")
import os, sys
import numpy as np
import pickle
import tensorflow as tf
import yaml
from utils.config import Configuration
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TFRecordProcessor:
    def __init__(self, config: Configuration):
        """
        Initializes the processor with configuration parameters.
        """
        self.bucket = con_obj.bucket
        self.client = niofs.Client(con_obj.access_key, con_obj.secret_key, max_pool_connections=20)
        self.tfrecord_niofs_path = config.tfrecord_niofs_path
        self.local_data_save_dir = config.local_data_save_dir
        self.save_tf_to_local_path_prefix = config.save_tf_to_local_tmp_path
        self.max_lane_num = config.max_lane_num
        self.max_node_num_per_lane = config.max_node_num_per_lane
        self.case_size = config.case_size
        self.vector_graph_feature_fea_dim = config.vector_graph_feature_fea_dim
        self.save_data_num_per = config.save_data_num_per

    def list_objects(self):
        """
        :param Bucket: Niofs桶名
        :param Prefix: 远程路径前缀
        :param Delimiter: 分隔符
            Delimiter=None或''：递归遍历目录下所有的文件
            Delimiter='/'：只遍历当前路径一级子目录和子文件
        :param MaxKeys: 最大文件数,不能超过1000，超过1000依然返回1000个
        :param NextMarker: 下次遍历的起点
        """

        file_list = []
        marker = ''
        while marker is not None:
            resp = self.client.list_objects(Bucket=self.bucket, Prefix=self.tfrecord_niofs_path, MaxKeys=1000,
                                            Delimiter='', Marker=marker)
            contents = resp.get("Contents")
            if contents:
                for result in contents:
                    key = result.get("Key")
                    size = result.get("Size")
                    if size != 0:
                        file_list.append(f"/{self.bucket}/{key} {size}")
            next_marker = resp.get("NextMarker")
            marker = next_marker
        return file_list

    def save_niofs_file_to_local(self, file_path_in_niofs: str, case_id: str):
        """
        Saves the TFRecord file from the remote file system (niofs) to a local temporary file.
        """
        random_tmp_path = f"{self.save_tf_to_local_path_prefix}_{case_id}"
        if not os.path.isfile(random_tmp_path):
            # Save the remote file locally
            try:
                rcop_byte = rcop([file_path_in_niofs])[0]
            except RuntimeError as e:
                print(e)
                print(f"clip_path: {file_path_in_niofs}")
                return None
            with open(random_tmp_path, "wb") as f_rand:
                f_rand.write(rcop_byte)
        return random_tmp_path

    @staticmethod
    def set_memory_growth():
        """
        Enables memory growth for GPUs in TensorFlow.
        """
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    @staticmethod
    def parse_tfrecord(record_path: str):
        """
        Parses a TFRecord file and returns a TensorFlow dataset.
        """
        ds = tf.data.TFRecordDataset(record_path, buffer_size=512 * 1024 * 1024)  # 512M buffer

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

    @staticmethod
    def is_valid_frame(frame_data_tf):
        """
        Checks if the current frame data is valid by examining the first element of "point_tag".
        """
        point_tag = frame_data_tf["point_tag"].values.numpy()
        return point_tag[0]

    @staticmethod
    def tf_tensor_is_none(fea_name, tf_tensor_map):
        """
        Checks whether the specified feature exists and is non-empty.
        """
        if fea_name in tf_tensor_map and tf_tensor_map[fea_name].values.numpy().size > 0:
            return False
        else:
            return True

    def convert_to_np(self, case_data_tf):
        """
        Converts the TFRecord data to numpy format organized by frame_id.
        """
        parsed_case_data = dict()
        try:
            for _, frame_data_tf in enumerate(case_data_tf):
                if not self.is_valid_frame(frame_data_tf):
                    continue

                point_tag = frame_data_tf["point_tag"].values.numpy()
                # Unpack the three elements of point_tag and convert frame_id to a Python int.
                _, _, frame_id = point_tag
                frame_id = int(frame_id.item())

                if frame_id not in parsed_case_data:
                    parsed_case_data[frame_id] = {}

                ego_history_feature_shape = frame_data_tf["ego_history_feature_shape"].values.numpy().astype(np.int32)
                ego_history_feature = frame_data_tf["ego_history_feature"].values.numpy().reshape(
                    ego_history_feature_shape)

                agent_feature_shape = frame_data_tf["agent_feature_shape"].values.numpy().astype(np.int32)
                agent_feature = frame_data_tf["agent_feature"].values.numpy().reshape(agent_feature_shape)

                agent_attribute_feature_shape = frame_data_tf["agent_attribute_feature_shape"].values.numpy().astype(
                    np.int32)
                agent_attribute_feature = frame_data_tf["agent_attribute_feature"].values.numpy().reshape(
                    agent_attribute_feature_shape)

                if not self.tf_tensor_is_none("vector_graph_feature_mask_shape", frame_data_tf):
                    vector_graph_feature_shape = frame_data_tf['vector_graph_feature_shape'].values.numpy().astype(
                        np.int32)
                    vector_graph_feature = frame_data_tf['vector_graph_feature'].values.numpy().reshape(
                        vector_graph_feature_shape)

                    vector_graph_feature_mask_shape = frame_data_tf['vector_graph_feature_mask_shape'].values.numpy()
                    vector_graph_feature_mask = frame_data_tf['vector_graph_feature_mask'].values.numpy().reshape(
                        vector_graph_feature_mask_shape)

                    vector_line_mask_shape = frame_data_tf['vector_line_mask_shape'].values.numpy()
                    vector_line_mask = frame_data_tf['vector_line_mask'].values.numpy().reshape(vector_line_mask_shape)
                else:
                    vector_graph_feature = np.full(
                        (self.max_lane_num, self.max_node_num_per_lane, len(self.vector_graph_feature_fea_dim)), -300)
                    vector_graph_feature_mask = np.full((self.max_lane_num, self.max_node_num_per_lane), 0)
                    vector_line_mask = np.full((self.max_lane_num,), 0)

                parsed_case_data[frame_id]["agent_feature"] = agent_feature
                parsed_case_data[frame_id]["agent_attribute_feature"] = agent_attribute_feature
                parsed_case_data[frame_id]["ego_history_feature"] = ego_history_feature
                parsed_case_data[frame_id]["vector_graph_feature"] = vector_graph_feature
                parsed_case_data[frame_id]["node_mask"] = vector_graph_feature_mask
                parsed_case_data[frame_id]["line_mask"] = vector_line_mask

        except tf.errors.DataLossError as e:
            # Skip the current frame on DataLossError
            pass

        # Clear TensorFlow session to free resources.
        tf.keras.backend.clear_session()
        return parsed_case_data

    def get_data(self, record_path: str) -> dict:
        """
        Sets memory growth, parses the TFRecord, and converts the data to numpy format.
        """
        self.set_memory_growth()
        case_data_tf = self.parse_tfrecord(record_path)
        return self.convert_to_np(case_data_tf)

    def clean_tmp_data(self, tmp_path: str):
        """
        Removes the local temporary file if it exists and if its name contains the temporary prefix.
        """
        if self.save_tf_to_local_path_prefix in tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    @staticmethod
    def write_pkl(output_save_path: str, data):
        """
        Saves the data as a pickle file.
        """
        dir_path = os.path.dirname(output_save_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(output_save_path, "wb") as f:
            pickle.dump(data, f)

    def get_file_path(self, case_id: str, file_index: int = 0):
        """
        Generates the pickle file save path.
        """
        pkl_path = self.local_data_save_dir + f"/{case_id}" + f"/{file_index}.pkl"
        return pkl_path

    def save_data_(self, input_data: dict, record_local_path: str):
        """
        Saves parts of the data as pickle files at set intervals.

        Saves data for every 35th key. Additionally, if the last key's index
        is more than 10 greater than the index of the last saved key, then the
        data corresponding to the last key is also saved.

        Parameters:
            input_data: Dictionary of data to save.
            record_local_path: The local record path (used to extract case_id).
        """
        keys = list(input_data.keys())
        idx = 0
        last_saved_index = None

        # Save every 35th key.
        for i, key in enumerate(keys):
            if i % self.save_data_num_per == 0:
                case_id = record_local_path.split(" ")[0].split("_")[-1]
                local_pkl_save_path = self.get_file_path(case_id, idx)
                self.write_pkl(local_pkl_save_path, input_data[key])
                last_saved_index = i
                idx += 1

        # If the last key is more than 10 indices away from the last saved index, also save it.
        if keys:
            last_index = len(keys) - 1
            if last_saved_index is None or (last_index - last_saved_index > 10):
                case_id = record_local_path.split(" ")[0].split("_")[-1]
                local_pkl_save_path = self.get_file_path(case_id, idx)
                self.write_pkl(local_pkl_save_path, input_data[keys[-1]])

    def process_all(self):
        """
        Main method to process the TFRecord files:
            1. Generate file list.
            2. For each file, save the remote file locally.
            3. Parse and convert data.
            4. Clean up temporary file.
            5. Save processed data as pickle.
        """
        file_list = self.list_objects()

        pbar = tqdm(total=min(len(file_list), self.case_size), desc="Processing cases", unit="case")

        for file in file_list:
            # Update progress bar based on the current number of files in local_data_save_dir
            if not os.path.exists(self.local_data_save_dir):
                os.makedirs(self.local_data_save_dir)
            current_count = len(os.listdir(self.local_data_save_dir))
            pbar.n = current_count
            pbar.refresh()
            if current_count >= self.case_size:
                pbar.close()
                print("Reached case size limit. Exiting loop.")
                break

            # Extract case_id from the file path (second last path component)
            case_id = file.split(" ")[0].split("/")[-2]
            record_local_path = self.save_niofs_file_to_local(file, case_id)
            if record_local_path is None:
                continue
            raw_data = self.get_data(record_local_path)
            self.clean_tmp_data(record_local_path)
            self.save_data_(raw_data, record_local_path)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.normpath(os.path.join(current_dir, "../configs/training.yaml"))

    with open(cfg_path, 'rb') as f:
        config_obj = yaml.safe_load(f)
    con_obj = Configuration(**config_obj)
    processor = TFRecordProcessor(con_obj)
    processor.process_all()
