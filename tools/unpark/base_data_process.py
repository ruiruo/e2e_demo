from typing import Tuple
import os
from abc import abstractmethod
import shutil
import multiprocessing
from datetime import datetime
from collections import defaultdict
from unpark.niofs_utils import create_niofs_client
from unpark.log_utils import create_logger
from unpark.file_utils import distribute_files


class BaseDataProcessFlow:
    def __init__(self, cfg: dict, fea_cfg: dict, **kwargs) -> None:
        self.cfg = cfg
        self.fea_cfg = fea_cfg
        self._init_params(cfg, fea_cfg)

        if not self.local_debug_case_path:
            self._init_main_logger()
            if "process_logger" in kwargs and kwargs["process_logger"]:
                self._init_sub_process_logger()

        if self.save_to_niofs:
            self._init_niofs_client()

        self._init_module_func(kwargs)

    def _init_params(self, cfg, fea_cfg):
        self.num_processes = self.cfg.common.get('num_processes', 1)
        self.data_version = self.cfg.data.get('data_version', "")
        self.save_to_niofs = self.cfg.common.get("save_to_niofs", True)
        self.niofs_data_save_dir = self.cfg.niofs.get("niofs_data_save_dir", "/regressions-training/vn_data/")
        self.save_tf_to_local_path_prefix = "/tmp/random_tmp"
        self.data_mode_list = self.cfg.data.get("mode_list", ["train", "test", "validate"])
        self.local_debug_case_path = self.cfg.data.get("local_debug_case_path", "")

    def _init_module_func(self):
        """设置TF Parser, TagMaker, SceneFilter, Constructor的处理函数
        """
        raise NotImplementedError()

    def _init_niofs_client(self, ):
        self.niofs_client = create_niofs_client(
            self.cfg.niofs.access_key,
            self.cfg.niofs.secret_key
        )
        self.bucket = self.cfg.niofs.bucket

    def _init_main_logger(self):
        # init main process logger
        # current_directory = os.path.dirname(os.path.abspath(__file__))
        logger_dir = self.cfg.root_dir + "/outs/" + f"/logs_{self.cfg.task_mode}/"
        cur_task_log_dir = logger_dir + datetime.now().strftime('%Y%m%d-%H%M%S')
        self.logger_path = cur_task_log_dir

        self.logger_map = defaultdict()
        self.logger = create_logger(cur_task_log_dir, name="main_process", rank=0)
        self.logger_map[-1] = self.logger
        self.logger.info(f"logger_path = {self.logger_path}")
        self.logger.info("Current Base Path:{}".format(self.logger_path))

        shutil.copy(self.cfg.config_path, os.path.join(cur_task_log_dir, "user_data_process_cfg.yaml"))

    def _init_sub_process_logger(self):
        for worker_i in range(self.num_processes):
            self.logger_map[worker_i] = create_logger(
                self.logger_path, name=f"worker_{worker_i}", rank=0)

    def info(self, msg: str, worker_i: int = -1):
        self.logger_map[worker_i].info(f"[Worker {worker_i}] {msg}")

    @abstractmethod
    def execute(self, ):
        pass

    def exe_parallel_(
            self,
            file_list: list,
            **kwargs: dict,
    ) -> Tuple[multiprocessing.Process, multiprocessing.Queue]:
        # Distribute files among workers evenly
        files_per_worker = distribute_files(file_list, self.num_processes)

        result_queue = multiprocessing.Queue()
        processes = []
        for worker_i, worker_files in enumerate(files_per_worker):
            p = multiprocessing.Process(target=self.worker_process_, args=(
                worker_i,
                worker_files,
                result_queue,
                kwargs
            ))
            processes.append(p)
            p.start()

        return processes, result_queue
