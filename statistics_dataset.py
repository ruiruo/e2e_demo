from utils.config import get_train_config_obj
from dataset.dataloader import TrajectoryDataloaderModule

cfg_path = "./configs/training.yaml"
config_obj = get_train_config_obj(config_path=cfg_path)
config_obj.data_dir = "/home/nio/data/"
config_obj.log_dir = config_obj.log_dir.replace("shaoqian.li", "nio")
config_obj.checkpoint_dir = config_obj.checkpoint_dir.replace("shaoqian.li", "nio")
config_obj.checkpoint_root_dir = "/home/nio/checkpoints/"
config_obj.local_data_save_dir = "/home/nio/"
config_obj.tokenizer = "/home/nio/reparke2e/configs/local2token.npy"
config_obj.detokenizer = "/home/nio/reparke2e/configs/token2local.json"
config_obj.batch_size = 4
config_obj.max_frame = 30
config_obj.num_workers = 8

data = TrajectoryDataloaderModule(cfg=config_obj)
data.setup("train")
print("################# Train #######################")
print(data.train_loader.dataset.get_statistics())
print("################## Validation ##################")
print(data.val_loader.dataset.get_statistics())
