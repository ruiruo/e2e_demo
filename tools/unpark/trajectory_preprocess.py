from typing import List, Dict, Union
import random
import numpy as np
import time
import multiprocessing
import traceback
from data.tf_parser import TFParser
from collections import defaultdict
from base_data_process import BaseDataProcessFlow
from unpark.file_utils import write_pkl, get_file_path, modify_shared_var_security, clean_tmp_data
from unpark.constants import GREEN, RED, RESET
from unpark.data_utils import nested_dict, merge_np_array
from unpark.log_utils import convert_stats_to_str_list_of_dict, sum_stats_list_of_dict, merge_stats_int_of_dict
from unpark.niofs_utils import save_niofs_file_to_local, upload_fileobj, upload_file


class TrajectoryDataProcessor(BaseDataProcessFlow):

    def __init__(self, cfg: dict, fea_cfg: dict, **kwargs: dict) -> None:
        super().__init__(cfg, fea_cfg, process_logger=True, **kwargs)
        self._init_data_process_params()

        if not self.local_debug_case_path:
            self._init_and_load_data_list()

    def _init_data_process_params(self, ):
        self.tfrecord_niofs_path = self.cfg.data.get('tfrecord_niofs_path', '')
        self.tfrecord_files_list = self.cfg.data.get('tfrecord_files_list', '')
        self.local_data_save_dir = self.cfg.data.get('local_data_save_dir', "")
        self.use_local_cache = self.cfg.data.get('use_local_cache', False)
        self.save_case_num_per = self.cfg.data.get('save_case_num_per', 4)
        self.load_from_niofs = self.cfg.data.get("load_from_niofs", False)
        self.only_data_balance = self.cfg.common.get("only_data_balance", False)
        self.processed_mode = "constructed"

    def _init_module_func(self):
        self.module_func = {
            "tf_parser": TFParser(
                cfg=self.cfg,
                fea_cfg=self.fea_cfg,
                logger_map=self.logger_map if hasattr(self, "logger_map") else {},
                logger_path=self.logger_path if hasattr(self, "logger_path") else None,
            )
        }

    def _init_and_load_data_list(self, ):
        """
        Load train, test, and validation
        """
        total_pct = self.cfg.data.pct_train + self.cfg.data.pct_test + self.cfg.data.pct_val
        if not abs(total_pct - 1.0) < 1e-6:
            raise ValueError("Train, test, and validation percentages should sum to 1")

        total_file_paths = []
        save_file_path = []
        raw_file_count = 0
        with open(self.tfrecord_files_list, 'r') as f:
            for line in f:
                # import pdb;pdb.set_trace()
                file_size, file_path = line.strip().split(" ")
                raw_file_count += 1
                if file_size == str(0):
                    self.logger.info(f"Skip: file_size is 0. path = {file_path}.")
                    continue
                total_file_paths.append(f"{self.tfrecord_niofs_path}/{file_path} {file_size}")
                save_file_path.append(f"{file_size} {file_path} ")

        total_file_paths = np.array(total_file_paths)
        len_total_files = len(total_file_paths)
        self.logger.info(GREEN + f"RemainFileNum/RawFileNum={len_total_files}/{raw_file_count}" + RESET)

        shuffled_file_indices = list(range(len_total_files))
        random.shuffle(shuffled_file_indices)
        total_file_paths = total_file_paths[shuffled_file_indices]

        self._split_train_test_validate(len_total_files, total_file_paths)

    def _split_train_test_validate(self, len_total_files: int, total_file_paths: List[str]):
        """Split the file list into train, test, and val sets
        """
        train_end = int(self.cfg.data.pct_train * len_total_files)
        test_end = train_end + int(self.cfg.data.pct_test * len_total_files)

        train_files = total_file_paths[:train_end]
        test_files = total_file_paths[train_end: test_end]
        val_files = total_file_paths[test_end:]

        self.files_list = {
            "train": train_files,
            "test": test_files,
            "validate": val_files
        }

    def execute(self):
        """
        data process
        """
        for mode in self.data_mode_list:
            self.process_files_parallel(mode)

    def save_data_(
            self,
            input_data: dict,
            mode: str = "train",
            file_save_index: Union[multiprocessing.Value, int] = None,
            lock: multiprocessing.Lock = None,
    ) -> None:
        save_path = None
        # import pdb;pdb.set_trace()
        if not isinstance(file_save_index, int) and not isinstance(file_save_index, str):
            file_save_index: int = modify_shared_var_security(file_save_index, lock)

        if self.save_to_niofs:
            niofs_pkl_save_path = get_file_path(
                self.niofs_data_save_dir,
                self.data_version,
                self.processed_mode,
                mode,
                file_save_index,
            )
            upload_fileobj(
                input_data,
                niofs_pkl_save_path,
                self.cfg.niofs.bucket,
                self.niofs_client,
            )
            save_path = niofs_pkl_save_path
        else:
            # save to local
            local_pkl_save_path = get_file_path(
                self.local_data_save_dir,
                self.data_version,
                self.processed_mode,
                mode,
                file_save_index
            )
            write_pkl(local_pkl_save_path, input_data)
            save_path = local_pkl_save_path

        return save_path

    def single_(
            self,
            worker_i: int,
            clip_i: int,
            clip_path: str,
            stats_info_map: Dict[str, int],
            mode: str,
    ) -> dict:
        """单个case的处理逻辑
            Step 1: 从 niofs 读取 tf record
            Step 2: 解析 tf record 并处理某些 fea
            Step 3: 对样本打上标签
            Step 4: 过滤无效等样本
            Step 5: 构造训练，测试，验证数据，存入 niofs
            Step 6: 删除本地缓存

        Parameters
        ----------
        worker_i: int
            进程id
        clip_i : int
            case index
        clip_path : str
            本地case路径
        stats_info_map : dict
            记录统计信息
        mode : str
            数据模型，train, test, validate

        Returns
        -------
        constructed_data: dict
            构造好的数据        
        """
        constructed_data = defaultdict()
        try:
            # Read data
            workflow_id = clip_path.split(" ")[0].split("/")[-3]
            case_id = clip_path.split(" ")[0].split("/")[-2]
            stats_info_map["workflow_id"] = int(workflow_id)
            stats_info_map["case_id"] = case_id
            record_local_path = save_niofs_file_to_local(
                clip_path,
                self.save_tf_to_local_path_prefix,
                case_id, False, self.use_local_cache
            )
            self.info(f"1. Save {clip_path} to {record_local_path}.", worker_i)

            # TF Parser
            self.info("2. Start parse tf record to frame data.", worker_i)
            parsed_case_data = self.module_func["tf_parser"](record_local_path, worker_i)

            # Tag Maker
            self.info("3. Start tag the scene.", worker_i)
            self.module_func["tag_maker"](parsed_case_data, worker_i)

            # Scene filter
            self.info("4. Start filter invalid scene.", worker_i)
            filtered_case_data = self.module_func["scene_filter"](
                parsed_case_data,
                stats_info_map,
                worker_i,
            )
            # if len(filtered_case_data) == 0:
            #     continue

            # Constructor 
            self.info("5. Start construct train, test and validate data.", worker_i)
            constructed_data = self.module_func["constructor"](
                filtered_case_data=filtered_case_data,
                res_stats_map=stats_info_map,
                mode=mode,
                worker_i=worker_i,
            )

            clean_tmp_data(self.save_tf_to_local_path_prefix, record_local_path)
            self.info(f"6. Remove local record {record_local_path}", worker_i)

        except Exception as e:
            self.info(f"{RED}process clip_idx {clip_i} failed with error: {e}{RESET}", worker_i)
            traceback.print_exc()
            # continue

        if mode == "validate" and len(constructed_data.keys()) > 0:
            constructed_data["case_id"] = case_id
            constructed_data["workflow_id"] = int(workflow_id)
        return constructed_data

    def save_constructed_data(
            self,
            constructed_data_cache: dict,
            mode: str = "train",
            file_save_index: multiprocessing.Value = int,
            lock: multiprocessing.Lock = None,
    ):
        save_path = None
        # save train and test data to pkl
        if mode != "validate" and len(constructed_data_cache) > 0:
            save_path = self.save_data_(
                input_data=constructed_data_cache,
                mode=mode,
                file_save_index=file_save_index,
                lock=lock,
            )
        elif mode == "validate" and len(constructed_data_cache.keys()) > 0:
            # save validate data to pkl
            for case_id, constructed_case_data in constructed_data_cache.items():
                save_path = self.save_data_(
                    input_data=constructed_case_data,
                    mode=mode,
                    file_save_index=case_id,
                    lock=lock,
                )
        return save_path

    def worker_process_(
            self,
            worker_i: int,
            files_list: list,
            result_queue: multiprocessing.Queue,
            kwargs: dict,
    ) -> None:
        """进程数据处理

        Parameters
        ----------
        worker_i : int
            进程 id
        files_list : list
            待处理文件列表
        result_queue : multiprocessing.Queue
            存储处理结果
        """
        assert "file_save_index" in kwargs, "`file_save_index` not in kwargs."
        assert "lock" in kwargs, "`lock` not in kwargs."
        assert "mode" in kwargs, "`mode` not in kwargs."
        file_save_index: multiprocessing.Value = kwargs["file_save_index"]
        lock: multiprocessing.Lock = kwargs["lock"]
        mode: str = kwargs["mode"]
        constructed_data_cache = self.reset_constructed_data()
        total_size = len(files_list)
        for case_i, one_clip_path in enumerate(files_list):
            processed_info = GREEN + f"[{mode}] Processed " + \
                             f"{case_i + 1}/{total_size}: " + \
                             f"{one_clip_path}" + RESET
            self.info(processed_info, worker_i)
            stats_info_map = defaultdict()

            constructed_data = self.single_(
                worker_i=worker_i,
                clip_i=case_i,
                clip_path=one_clip_path,
                stats_info_map=stats_info_map,
                mode=mode
            )

            # merge constructed_data
            if mode == "validate" and len(constructed_data.keys()) > 0:
                case_id = constructed_data["case_id"]
                constructed_data_cache[case_id] = constructed_data
            elif mode != "validate" and len(constructed_data.keys()) > 0:
                merge_np_array(constructed_data, constructed_data_cache)

            # save case to pkl file
            if (case_i > 0 and case_i % self.save_case_num_per == 0) \
                    or (case_i == total_size - 1):
                # get file save index
                save_path = self.save_constructed_data(
                    constructed_data_cache=constructed_data_cache,
                    mode=mode,
                    file_save_index=file_save_index,
                    lock=lock,
                )
                data_size = self.get_batch_size(constructed_data_cache, mode)
                msg = f"Successfully save {mode} data to " + \
                      f"{GREEN}{save_path}{RESET}. data_size = {data_size}"
                self.info(msg, worker_i)
                constructed_data_cache = self.reset_constructed_data(constructed_data_cache)

            # 每个case处理完后将统计信息存入queue
            result_queue.put(stats_info_map)

    def get_batch_size(self, constructed_data: dict, mode: str = "train") -> int:
        if mode == "validate":
            return len(constructed_data.keys())
        else:
            if "fs_scenes_traj_info" not in constructed_data:
                return 0
            else:
                return constructed_data["fs_scenes_traj_info"].shape[0]

    def reset_constructed_data(mode: str, constructed_data_cache: dict = None):
        # 如果非空，先清除数据
        if constructed_data_cache:
            del constructed_data_cache

        # 初始化
        constructed_data_cache = nested_dict() if mode == "validate" else defaultdict()
        return constructed_data_cache

    def process_files_parallel(self, mode: str = "train"):
        """
        multi process data, label, filter,

        Parameters
        ----------
        mode : str, optional
            train, test, validate, by default "train"
        """
        stats_info_total = defaultdict(list)

        file_save_index_shared = multiprocessing.Value('i', 0)
        lock = multiprocessing.Lock()

        processes, result_queue = self.exe_parallel_(
            file_list=self.files_list[mode],
            file_save_index=file_save_index_shared,
            lock=lock,
            mode=mode,
        )

        loop_num, get_num = 0, 0
        while True:
            while not result_queue.empty():
                result_per_case: dict = result_queue.get()
                merge_stats_int_of_dict(result_per_case, stats_info_total)
                get_num += 1

            loop_num += 1
            # 防止忙等待
            time.sleep(0.15)
            # self.logger.info(f"loop_num={loop_num}, get_num={get_num}")

            # 检查子进程是否全部结束
            if all(not p.is_alive() for p in processes) and result_queue.empty():
                break

        # 等待所有子进程结束
        for p in processes:
            p.join()

        result_queue.close()
        result_queue.join_thread()

        if self.logger:
            self.logger.info(GREEN + convert_stats_to_str_list_of_dict(
                stats_info_total, f"[Total {mode}] ") + RESET)
        sum_stats_info = sum_stats_list_of_dict(stats_info_total)
        stats_info_total.update(sum_stats_info)

        # NOTE save stats info
        stats_save_name = f"/{self.processed_mode}_{mode}_stats_info.yaml"
        niofs_stats_save_path = self.niofs_data_save_dir + \
                                f"{self.data_version}/{self.processed_mode}/" + \
                                stats_save_name
        local_save_path = self.logger_path + stats_save_name
        # save_yaml(stats_info_total, local_save_path)
        write_pkl(local_save_path, stats_info_total)
        upload_file(
            source_file_path=local_save_path,
            target_file_path=niofs_stats_save_path,
            bucket=self.cfg.niofs.bucket,
            client=self.niofs_client,
        )
