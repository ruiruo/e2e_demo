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
# Limit the number of samples for evaluation purposes
config_obj.max_train = 100
config_obj.max_val = 100
config_obj.log_every_n_steps = 2
config_obj.data_dir = "/home/nio/data/"
config_obj.log_dir = config_obj.log_dir.replace("shaoqian.li", "nio")
config_obj.checkpoint_dir = config_obj.checkpoint_dir.replace("shaoqian.li", "nio")
config_obj.checkpoint_root_dir = "/home/nio/checkpoints/"
config_obj.local_data_save_dir = "/home/nio/"
config_obj.tokenizer = "/home/nio/reparke2e/configs/local2token.npy"
config_obj.detokenizer = "/home/nio/reparke2e/configs/token2local.json"
config_obj.batch_size = 4

# Load the token-to-location mapping from the detokenizer file.
with open(config_obj.detokenizer, "r") as f:
    token2local_dict = json.load(f)

# Check for special tokens (BOS, EOS, PAD) in the token-to-location mapping.
special_tokens = {
    "BOS": config_obj.bos_token,
    "EOS": config_obj.eos_token,
    "PAD": config_obj.pad_token,
}
missing_special_tokens = []
for name, token in special_tokens.items():
    # Check both the token directly and its string representation.
    if token not in token2local_dict and str(token) not in token2local_dict:
        missing_special_tokens.append((name, token))
if missing_special_tokens:
    print("Warning: The following special tokens are missing from the token2local mapping:")
    for name, token in missing_special_tokens:
        print(f"  {name}: {token}")


def evaluate_dataset(dataset, token2local, dataset_name="Dataset"):
    """
    Iterate over all samples in the dataset (an instance of TrajectoryDataModule)
    and collect:
      - Token frequencies (from input_ids and labels)
      - Euclidean distance from (0,0) to the goal coordinate (the last point of the detokenized trajectory)
      - Goal x and y coordinate values
    Returns a dictionary with these statistics.

    Detokenization calls are wrapped in try/except blocks. If detokenization fails,
    an error message is printed and the exception is raised.
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

        # Try detokenizing the target sequence into (x, y) coordinates.
        try:
            coords = detokenize_traj_waypoints(labels, token2local,
                                               config_obj.bos_token,
                                               config_obj.eos_token,
                                               config_obj.pad_token)
        except Exception as e:
            print(f"Error detokenizing labels in sample {i} of {dataset_name}: {e}")
            raise e

        coords = np.array(coords)  # shape: (seq_len, 2)
        if coords.shape[0] < 1:  # Need at least one coordinate as goal.
            continue

        # Try detokenizing the goal tokens.
        try:
            goal_tokens = sample["goal"].tolist()
            goal = detokenize_traj_waypoints(goal_tokens, token2local,
                                             config_obj.bos_token,
                                             config_obj.eos_token,
                                             config_obj.pad_token)
        except Exception as e:
            print(f"Error detokenizing goal in sample {i} of {dataset_name}: {e}")
            raise e

        # Assume goal is the first coordinate from the detokenized output.
        goal = np.array(goal)[0]
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
    """Plot overlaid distributions for train and validation datasets."""

    # 1. Token Frequency Distribution (bar chart)
    tokens_train = set(stats_train["token_counter"].keys())
    tokens_val = set(stats_val["token_counter"].keys())
    tokens_union = np.array(sorted(tokens_train.union(tokens_val)))

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

    # 2. Goal Distance Distribution (histogram)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(stats_train["distances"], bins=50, alpha=0.5, label="Train", edgecolor='black')
    ax.hist(stats_val["distances"], bins=50, alpha=0.5, label="Val", edgecolor='black')
    ax.set_xlabel("Goal Distance (from (0,0))")
    ax.set_ylabel("Count")
    ax.set_title("Goal Distance Distribution (Train vs Val)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 3. Goal X Coordinate Distribution (histogram)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(stats_train["x_coords"], bins=50, alpha=0.5, label="Train", edgecolor='black')
    ax.hist(stats_val["x_coords"], bins=50, alpha=0.5, label="Val", edgecolor='black')
    ax.set_xlabel("Goal X Coordinate")
    ax.set_ylabel("Count")
    ax.set_title("Goal X Coordinate Distribution (Train vs Val)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 4. Goal Y Coordinate Distribution (histogram)
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
    """Print summary statistics for goal distances and coordinates, including a check for 0-distance goals."""
    if stats["distances"].size > 0:
        mean_dist = np.mean(stats["distances"])
        std_dist = np.std(stats["distances"])
        min_dist = np.min(stats["distances"])
        max_dist = np.max(stats["distances"])
        zero_count = np.sum(np.isclose(stats["distances"], 0, atol=1e-8))
        total = stats["distances"].size
        zero_percent = (zero_count / total) * 100

        print(f"{dataset_name} Goal Distance Statistics:")
        print(f"  Mean: {mean_dist:.4f}")
        print(f"  Std: {std_dist:.4f}")
        print(f"  Min: {min_dist:.4f}")
        print(f"  Max: {max_dist:.4f}")
        print(f"  Goals with 0 distance: {zero_count} ({zero_percent:.2f}% of samples)")
    else:
        print(f"{dataset_name}: No distances computed.")

    if stats["x_coords"].size > 0 and stats["y_coords"].size > 0:
        print(f"{dataset_name} Goal X Coordinate: mean={np.mean(stats['x_coords']):.4f}, std={np.std(stats['x_coords']):.4f}")
        print(f"{dataset_name} Goal Y Coordinate: mean={np.mean(stats['y_coords']):.4f}, std={np.std(stats['y_coords']):.4f}")
    else:
        print(f"{dataset_name}: No coordinate data computed.")


def compare_token_distribution(stats_train, stats_val):
    """
    Compute and print the KL divergence between the token distributions
    of the train and validation datasets. Also plot the normalized token distribution
    for direct comparison.
    """
    tokens_train = set(stats_train["token_counter"].keys())
    tokens_val = set(stats_val["token_counter"].keys())
    tokens_union = np.array(sorted(tokens_train.union(tokens_val)))

    counts_train = np.array([stats_train["token_counter"].get(token, 0) for token in tokens_union], dtype=float)
    counts_val = np.array([stats_val["token_counter"].get(token, 0) for token in tokens_union], dtype=float)

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    total_train = np.sum(counts_train) + epsilon * len(tokens_union)
    total_val = np.sum(counts_val) + epsilon * len(tokens_union)

    p_train = (counts_train + epsilon) / total_train
    p_val = (counts_val + epsilon) / total_val

    # Compute KL divergence: KL(Train || Val)
    kl_divergence = np.sum(p_train * np.log(p_train / p_val))
    print("KL Divergence (Train || Val):", kl_divergence)

    # Plot normalized token distributions
    plt.figure(figsize=(12, 6))
    plt.plot(tokens_union, p_train, label="Train Normalized Distribution", marker='o', linestyle='-')
    plt.plot(tokens_union, p_val, label="Val Normalized Distribution", marker='o', linestyle='-')
    plt.xlabel("Token ID")
    plt.ylabel("Normalized Frequency")
    plt.title("Normalized Token Distribution Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load train and validation datasets
    train_dataset = TrajectoryDataModule(config=config_obj, is_train=1)
    val_dataset = TrajectoryDataModule(config=config_obj, is_train=0)

    # Evaluate each dataset
    train_stats = evaluate_dataset(train_dataset, token2local_dict, dataset_name="Train")
    val_stats = evaluate_dataset(val_dataset, token2local_dict, dataset_name="Val")

    # Print summary statistics including check for 0-distance goals
    print_summary(train_stats, dataset_name="Train Dataset")
    print_summary(val_stats, dataset_name="Valid Dataset")

    # Plot distributions for train and validation (overlaid)
    plot_distribution(train_stats, val_stats)

    # Compare token distributions: compute KL divergence and plot normalized distributions
    compare_token_distribution(train_stats, val_stats)
