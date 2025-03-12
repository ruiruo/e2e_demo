from unpark.trajectory_preprocess import TrajectoryDataProcessor
from unpark.file_utils import read_config
import argparse
import os


def parse_config():
    parser = argparse.ArgumentParser(description="Scene Evaluation Data Process & Analysis")
    parser.add_argument(
        "--config",
        default="./configs/data_cfgs/data_process_cfg.yaml",
        required=False,
        help="default config path",
    )
    parser.add_argument(
        "--fea_cfg",
        default="./configs/data_cfgs/fea_cfg.yaml",
        required=False,
        help="default config path",
    )
    parser_args = parser.parse_args()
    cfg = read_config(parser_args.config)
    cfg.config_path = parser_args.config
    cfg.task_mode = parser_args.task_mode

    fea_cfg = read_config(parser_args.fea_cfg)
    fea_cfg.fea_cfg_path = parser_args.fea_cfg

    root_dir = os.path.dirname(os.path.abspath(__file__))
    cfg.root_dir = root_dir

    return cfg, fea_cfg


if __name__ == '__main__':
    cfg, fea_cfg = parse_config()
    workflow = TrajectoryDataProcessor(cfg, fea_cfg)
    workflow.execute()
