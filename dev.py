from model.trajectory_generator_predict import TrajectoryPredictModule
from utils.config import get_train_config_obj
from envs import ReplayHighwayEnv
from envs.wrapper import OpenCVRecorder
from utils.config import get_inference_config_obj
import torch

config = {
    "test_img": "/home/nio/",
}
pred_config_obj = get_inference_config_obj("./configs/predict.yaml")

pretrain_config_obj = get_train_config_obj(config_path="./configs/scaling_law_eval_4s/scaling_law_0.0.yaml")
pretrain_config_obj.log_every_n_steps = 1
pretrain_config_obj.max_train = 20
pretrain_config_obj.max_val = 5
pretrain_config_obj.epochs = 1
pretrain_config_obj.data_dir = "/home/nio/data/"
pretrain_config_obj.log_dir = pretrain_config_obj.log_dir.replace("shaoqian.li", "nio")
pretrain_config_obj.checkpoint_dir = pretrain_config_obj.checkpoint_dir.replace("shaoqian.li", "nio")
pretrain_config_obj.checkpoint_root_dir = "/home/nio/checkpoints/"
pretrain_config_obj.local_data_save_dir = "/home/nio/"
pretrain_config_obj.tokenizer = "/home/nio/reparke2e/configs/local2token.npy"
pretrain_config_obj.detokenizer = "/home/nio/reparke2e/configs/token2local.json"
pretrain_config_obj.batch_size = 4
pretrain_config_obj.ar_start_epoch = 1
pretrain_config_obj.ar_warmup_epochs = 1
inference_obj = TrajectoryPredictModule(infer_cfg=pred_config_obj,
                                        train_cfg=pretrain_config_obj,
                                        device="gpu")
print("Model summary:")
print(inference_obj.model)
env = OpenCVRecorder(ReplayHighwayEnv("/home/nio/data/test/",
                                      pre_train_config=pretrain_config_obj,
                                      env_config=config), video_path="~/render.mp4", fps=5)
obs, info = env.reset()

for i in range(0, 20):
    model_input = {
        "input_ids": torch.Tensor([obs[0:1]]).to("cuda", dtype=torch.long),
        "ego_info": torch.Tensor([obs[1:4]]).to("cuda", dtype=torch.float),
        "agent_info": torch.Tensor([obs[4: 634]]).to("cuda", dtype=torch.float).reshape([1,
                                                                                         pretrain_config_obj.max_frame + 1,
                                                                                         pretrain_config_obj.max_agent,
                                                                                         -1]),
        "goal": torch.Tensor([obs[634:]]).to("cuda", dtype=torch.long),
    }
    outputs = inference_obj.inference_batch(model_input)[0]
    print(outputs)
    obs = env.step(outputs[2])[0]
