from pytorch_lightning import seed_everything
from utils.config import get_inference_config_obj
from dataset.dataloader import DataLoader, TrajectoryDataModule
from model.trajectory_generator_predict import TrajectoryPredictModule
from utils.display import plot_and_save_trajectories
import numpy as np


def compute_metrics(pred_traj, label_traj, pred_tokens, label_tokens):
    l2_distances = []
    for pred, gt in zip(pred_traj, label_traj):
        pred = np.array(pred)
        gt = np.array(gt)
        min_len = min(len(pred), len(gt))

        # Skip very short trajectories
        if min_len < 2:
            continue

        # Compute L2 distance across overlapping timestep and average
        l2 = np.sqrt(np.sum((pred[:min_len] - gt[:min_len]) ** 2, axis=-1)).mean()
        l2_distances.append(l2)

    l2_distances = [x for x in l2_distances if not np.isnan(x)]
    if len(l2_distances) > 0:
        avg_l2 = float(np.mean(l2_distances))
    else:
        avg_l2 = None

    correct = 0
    total = 0
    for pred_tok, gt_tok in zip(pred_tokens, label_tokens):
        # Only compare overlapping region if sequence lengths differ
        min_len = min(len(pred_tok), len(gt_tok))
        for p, g in zip(pred_tok[:min_len], gt_tok[:min_len]):
            if p == g:
                correct += 1
        total += min_len
    token_acc = correct / total if total > 0 else 0

    return {
        "avg_L2_distance": avg_l2,
        "token_accuracy": token_acc
    }


seed_everything(15)
pred_config_obj = get_inference_config_obj("./configs/predict.yaml")
train_config_obj = pred_config_obj.train_meta_config
train_config_obj.log_every_n_steps = 2
train_config_obj.max_train = 10000
train_config_obj.max_val = 500
train_config_obj.log_dir = train_config_obj.log_dir.replace("shaoqian.li", "nio")
train_config_obj.checkpoint_dir = train_config_obj.checkpoint_dir.replace("shaoqian.li", "nio")
train_config_obj.checkpoint_root_dir = "/home/nio/checkpoints/"
train_config_obj.local_data_save_dir = "/home/nio/"
train_config_obj.tokenizer = "/home/nio/reparke2e/configs/local2token.npy"
train_config_obj.detokenizer = "/home/nio/reparke2e/configs/token2local.json"
train_config_obj.batch_size = 4

# Create the inference object.
inference_obj = TrajectoryPredictModule(infer_cfg=pred_config_obj,
                                        train_cfg=train_config_obj,
                                        device="gpu")
print("Model summary:")
print(inference_obj.model)

# Create the DataLoader for evaluation.
data = DataLoader(dataset=TrajectoryDataModule(config=train_config_obj,
                                               data_path="/home/nio/data/road_test",
                                               max_allow=10000000),
                  batch_size=train_config_obj.batch_size,
                  shuffle=True,
                  num_workers=train_config_obj.num_workers,
                  pin_memory=True,
                  drop_last=True)
print("Number of batches:", len(data))

# Set the model to evaluation mode.
model = inference_obj.model
model.eval()

pred_tf, label_tf, agents_tf, pred_tf_token, label_tf_token = inference_obj.test_teacher_forcing(data)

metrics_tf = compute_metrics(pred_tf, label_tf, pred_tf_token, label_tf_token)

print(metrics_tf)

plot_and_save_trajectories(
    pred_tf, label_tf, agents_tf,
    f"./test_teacher_forcing_figures/"
    f"{pred_config_obj.model_ckpt_path.split('/')[-2]}/"
    f"{pred_config_obj.model_ckpt_path.split('/')[-1]}_te/"
)

pred_nor, label_nor, agents_nor, pred_nor_token, label_nor_token = inference_obj.test(data)

metrics_nor = compute_metrics(pred_nor, label_nor, pred_nor_token, label_nor_token)

print(metrics_nor)

plot_and_save_trajectories(
    pred_nor, label_nor, agents_nor,
    f"./test_teacher_forcing_figures/"
    f"{pred_config_obj.model_ckpt_path.split('/')[-2]}/"
    f"{pred_config_obj.model_ckpt_path.split('/')[-1]}_ar/"
)
