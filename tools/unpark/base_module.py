from abc import abstractmethod
from unpark.niofs_utils import upload_fileobj
from unpark.file_utils import write_pkl, get_file_path
from unpark.constants import RED, RESET

# key=stat fileds, val=init val
REGISTER_STAT_INIT_INFO = {
    "scene_filter": {
        "filter__fs_gt_traj_diff": 0,
        "filter__scene_num_less_thr": 0,
        "filter__junction_env": 0,
        "filter__fs_scene_num_less_2": 0,
        "filter__ego_static": 0,
        "left_lc_num": 0,
        "right_lc_num": 0,
        "nudge_num": 0,
        "hold_num": 0,
        "keep_num": 0,
        "no_gt": 0,
        "gt_likely_gt_has_behavior_diff": 0,
        "lc_task_filterd_num": 0,
        "no_positive_scene_num": 0,
    },
    "constructor": {
        "select_scene_num_less_thr": 0,
        "diff_small_between_pos_neg": 0,
        "used_scene_num_less_thr": 0,
    }
}


class DataProcessModule:

    def __init__(
            self,
            cfg: dict,
            func_map: dict = {},
            fea_cfg: dict = {},
            logger_map: dict = {},
            **kwargs,
    ):
        self.func_map = func_map
        self.logger_map = logger_map
        if -1 in self.logger_map:
            self.logger = logger_map[-1]
        else:
            print(f"Warning: {RED}`logger_map` has no main process logger.{RESET}")

        self._init_params(cfg, fea_cfg, kwargs)

    def _init_params(self, cfg, fea_cfg, kwargs):
        self.cfg = cfg
        self.fea_cfg = fea_cfg
        self.kwargs = kwargs
        self.num_processes = cfg.common.get("num_processes", 1)
        self.save_to_niofs = cfg.common.get("save_to_niofs", False)
        self.niofs_data_save_dir = self.cfg.niofs.get("niofs_data_save_dir", "/regressions-training/vn_data/")
        self.data_version = self.cfg.data.get('data_version', "")
        self.local_data_save_dir = cfg.data.get('local_data_save_dir', "")
        self.only_data_balance = cfg.common.get("only_data_balance", False)
        self.save_frame_nums_per = cfg.data_balance.get('save_frame_nums_per', 4)
        self.logger_path = kwargs.get("logger_path", "")

    def info(self, msg: str, worker_i: int = -1):
        """带有进程id的日志

        Parameters
        ----------
        msg : str
            消息
        """
        if worker_i in self.logger_map:
            self.logger_map[worker_i].info(f"[Worker {worker_i}] {msg}")
        else:
            print(f"[Worker {worker_i}] {msg}")

    @abstractmethod
    def __call__(self, input_case_data: dict, worker_i: int = -1) -> dict:
        pass

    def flow_func_(self,
                   need_keys: dict,
                   **kwargs
                   ):
        for key in need_keys:
            if key in self.func_map:
                self.func_map[key](
                    **kwargs,
                    fea_name=key,
                )

    def save_data_(
            self,
            input_data: dict,
            save_file_index: int,
            mode: str = "train",
            **kwargs,
    ) -> None:
        save_data_version = (
            kwargs["save_data_version"]
            if "save_data_version" in kwargs
            else self.data_version
        )

        save_path = None
        if self.save_to_niofs:
            niofs_pkl_save_path = get_file_path(
                self.niofs_data_save_dir,
                save_data_version,
                "balanced",
                mode,
                save_file_index
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
                save_data_version,
                "balanced",
                mode,
                save_file_index
            )
            write_pkl(local_pkl_save_path, input_data)
            save_path = local_pkl_save_path

        return save_path
