import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from utils.config import get_train_config_obj
from dataset.dataset_driving import TrajectoryDataModule
from utils.trajectory_utils import detokenize_traj_waypoints

# Set visible CUDA device (if applicable)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load configuration
cfg_path = "/home/nio/reparke2e/configs/training.yaml"
config_obj = get_train_config_obj(config_path=cfg_path)
# For evaluation purposes, we limit the number of samples.
config_obj.max_train = 10000000
config_obj.max_val = 10000000
config_obj.log_every_n_steps = 2
config_obj.data_dir = "/home/nio/data/"
config_obj.log_dir = config_obj.log_dir.replace("shaoqian.li", "nio")
config_obj.checkpoint_dir = config_obj.checkpoint_dir.replace("shaoqian.li", "nio")
config_obj.checkpoint_root_dir = "/home/nio/checkpoints/"
config_obj.local_data_save_dir = "/home/nio/"
config_obj.tokenizer = "/home/nio/reparke2e/configs/local2token.npy"
config_obj.detokenizer = "/home/nio/reparke2e/configs/token2local.json"
config_obj.batch_size = 4

# Load the token-to-location mapping from detokenizer file.
with open(config_obj.detokenizer, "r") as f:
    token2local_dict = json.load(f)


def evaluate_dataset(dataset, token2local, dataset_name="Dataset"):
    """
    Iterate over all samples in the dataset (an instance of TrajectoryDataModule),
    and collect:
      - Token frequencies (from input_ids and labels)
      - Euclidean distance from (0,0) to the goal coordinate (the last point of the detokenized trajectory)
      - Goal x and y coordinate values
    Returns a dictionary with these statistics.
    """
    token_counter = Counter()
    all_distances = []
    all_goal_x = []
    all_goal_y = []

    num_samples = len(dataset)
    print(f"{dataset_name} size: {num_samples}")
    for i in range(num_samples):
        sample = dataset[i]
        # Get tokens from input and target sequences (as lists of ints)
        input_ids = sample["input_ids"].tolist()
        labels = sample["labels"].tolist()
        token_counter.update(input_ids)
        token_counter.update(labels)

        # Detokenize target sequence into (x, y) coordinates.
        # The goal is defined as the last coordinate.
        coords = detokenize_traj_waypoints(labels, token2local)
        coords = np.array(coords)  # shape: (seq_len, 2)
        if coords.shape[0] < 1:  # Need at least one coordinate to have a goal.
            continue

        goal = coords[-1]  # The goal coordinate.
        goal_distance = np.linalg.norm(goal - np.array([0, 0]))
        all_distances.append(goal_distance)
        all_goal_x.append(goal[0])
        all_goal_y.append(goal[1])

    stats = {
        "token_counter": token_counter,
        "distances": np.array(all_distances),
        "x_coords": np.array(all_goal_x),
        "y_coords": np.array(all_goal_y),
    }
    return stats


def plot_distribution(stats_train, stats_val):
    """Produce single figures for each distribution, overlaying train and validation data."""

    # 1. Token Frequency Distribution ( bar charts)
    # Get sorted union of token IDs.
    tokens_train = set(stats_train["token_counter"].keys())
    tokens_val = set(stats_val["token_counter"].keys())
    tokens_union = np.array(sorted(tokens_train.union(tokens_val)))

    # Prepare counts: if a token is missing in a dataset, count=0.
    counts_train = [stats_train["token_counter"].get(token, 0) for token in tokens_union]
    counts_val = [stats_val["token_counter"].get(token, 0) for token in tokens_union]

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.4  # bar width
    ax.bar(tokens_union - width / 2, counts_train, width=width, label="Train")
    ax.bar(tokens_union + width / 2, counts_val, width=width, label="Val")
    ax.set_xlabel("Token ID")
    ax.set_ylabel("Frequency")
    ax.set_title("Token Frequency Distribution (Train vs Val)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 2. Distribution of Goal Distances ( histograms)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(stats_train["distances"], bins=50, alpha=0.5, label="Train", edgecolor='black')
    ax.hist(stats_val["distances"], bins=50, alpha=0.5, label="Val", edgecolor='black')
    ax.set_xlabel("Goal Distance (from (0,0))")
    ax.set_ylabel("Count")
    ax.set_title("Goal Distance Distribution (Train vs Val)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 3. X Coordinate Distribution ( histograms)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(stats_train["x_coords"], bins=50, alpha=0.5, label="Train", edgecolor='black')
    ax.hist(stats_val["x_coords"], bins=50, alpha=0.5, label="Val", edgecolor='black')
    ax.set_xlabel("Goal X Coordinate")
    ax.set_ylabel("Count")
    ax.set_title("Goal X Coordinate Distribution (Train vs Val)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 4. Y Coordinate Distribution ( histograms)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(stats_train["y_coords"], bins=50, alpha=0.5, label="Train", edgecolor='black')
    ax.hist(stats_val["y_coords"], bins=50, alpha=0.5, label="Val", edgecolor='black')
    ax.set_xlabel("Goal Y Coordinate")
    ax.set_ylabel("Count")
    ax.set_title("Goal Y Coordinate Distribution (Train vs Val)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def print_summary(stats, dataset_name="Dataset"):
    """Print summary statistics for goal distances and goal coordinates."""
    if stats["distances"].size > 0:
        print(f"{dataset_name} Goal Distance Statistics:")
        print(f"  Mean: {np.mean(stats['distances']):.4f}")
        print(f"  Std: {np.std(stats['distances']):.4f}")
        print(f"  Min: {np.min(stats['distances']):.4f}")
        print(f"  Max: {np.max(stats['distances']):.4f}")
    else:
        print(f"{dataset_name}: No distances computed.")

    if stats["x_coords"].size > 0 and stats["y_coords"].size > 0:
        print(
            f"{dataset_name} Goal X Coordinate: mean={np.mean(stats['x_coords']):.4f}, std={np.std(stats['x_coords']):.4f}")
        print(
            f"{dataset_name} Goal Y Coordinate: mean={np.mean(stats['y_coords']):.4f}, std={np.std(stats['y_coords']):.4f}")
    else:
        print(f"{dataset_name}: No coordinate data computed.")


if __name__ == '__main__':
    # Load train and validation datasets (using the same pre-processing logic)
    train_dataset = TrajectoryDataModule(config=config_obj, is_train=1)
    val_dataset = TrajectoryDataModule(config=config_obj, is_train=0)

    # Evaluate each dataset
    train_stats = evaluate_dataset(train_dataset, token2local_dict, dataset_name="Train")
    val_stats = evaluate_dataset(val_dataset, token2local_dict, dataset_name="Val")

    # Print summary statistics
    print_summary(train_stats, dataset_name="Train Dataset")
    print_summary(val_stats, dataset_name="Valid Dataset")

    # Plot distributions with train and val data overlaid in single figures.
    plot_distribution(train_stats, val_stats)
