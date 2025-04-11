from pytorch_lightning import seed_everything
from utils.config import get_inference_config_obj
from dataset.dataloader import DataLoader, TrajectoryDataModule
from model.trajectory_generator_predict import TrajectoryPredictModule
from utils.display import plot_and_save_trajectories
seed_everything(15)
pred_config_obj = get_inference_config_obj("./configs/predict.yaml")
train_config_obj = pred_config_obj.train_meta_config
train_config_obj.log_every_n_steps = 2
train_config_obj.max_train = 10
train_config_obj.max_val = 5
train_config_obj.data_dir = "/home/nio/data/"
train_config_obj.log_dir = train_config_obj.log_dir.replace("shaoqian.li", "nio")
train_config_obj.checkpoint_dir = train_config_obj.checkpoint_dir.replace("shaoqian.li", "nio")
train_config_obj.checkpoint_root_dir = "/home/nio/checkpoints/"
train_config_obj.local_data_save_dir = "/home/nio/"
train_config_obj.tokenizer = "/home/nio/reparke2e/configs/local2token.npy"
train_config_obj.detokenizer = "/home/nio/reparke2e/configs/token2local.json"
train_config_obj.batch_size = 4
inference_obj = TrajectoryPredictModule(infer_cfg=pred_config_obj,
                                        train_cfg=train_config_obj,
                                        device="gpu")

print(inference_obj.model)

data = DataLoader(dataset=TrajectoryDataModule(config=train_config_obj, is_train=1),
                  batch_size=train_config_obj.batch_size,
                  shuffle=True,
                  num_workers=train_config_obj.num_workers,
                  pin_memory=True,
                  drop_last=True)

print(len(data))

# Run prediction using the Lightning Trainer
predictions = inference_obj.predict(data)

