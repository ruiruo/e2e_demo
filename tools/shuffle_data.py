import os
import shutil
import random
import argparse
from multiprocessing import Pool


def _process_folder(args):
    """
    Helper function to process a single folder.
    Unpacks arguments for use in multiprocessing.
    """
    folder, destination, raw_data_dir, copy_mode = args
    src_path = os.path.join(raw_data_dir, folder)
    dst_path = os.path.join(destination, folder)
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)  # Remove existing folder to prevent conflicts
    if copy_mode:
        shutil.copytree(src_path, dst_path)
    else:
        shutil.move(src_path, dst_path)


def split_folders(raw_data_dir: str, train_dir: str, val_dir: str, test_dir: str = None,
                  train_ratio: float = 0.8, val_ratio: float = 0.2, copy_mode: bool = True,
                  use_multiprocessing: bool = False, num_workers: int = 4):
    """
    Randomly shuffle and split subdirectories from raw_data_dir into training, validation,
    and optionally test sets, then copy (or move/link) them into train_dir, val_dir, and test_dir.

    When test_dir is not provided, the function performs a two-way split (training/validation)
    using train_ratio for training and the rest for validation. If test_dir is provided, then:
      - training gets train_ratio of the data,
      - validation gets val_ratio of the data, and
      - test gets the remaining folders (i.e. 1 - train_ratio - val_ratio).

    :param raw_data_dir: Directory containing raw data subdirectories.
    :param train_dir: Destination directory for training data.
    :param val_dir: Destination directory for validation data.
    :param test_dir: Destination directory for test data. If None, no test split is performed.
    :param train_ratio: Fraction of subdirectories to use for training.
    :param val_ratio: Fraction of subdirectories to use for validation when test_dir is provided.
                      (When test_dir is None, validation gets all remaining folders.)
    :param copy_mode: If True, copy folders; if False, move them.
    :param use_multiprocessing: If True, use multiprocessing to process folders concurrently.
    :param num_workers: Number of worker processes to use if multiprocessing is enabled.
    """
    # List all subdirectories in raw_data_dir
    all_subfolders = [entry for entry in os.listdir(raw_data_dir)
                      if os.path.isdir(os.path.join(raw_data_dir, entry))]

    if not all_subfolders:
        print(f"No subdirectories found in {raw_data_dir}")
        return

    # Shuffle the list of subdirectories
    random.shuffle(all_subfolders)
    n = len(all_subfolders)

    if test_dir is None:
        # Two-way split: training and validation
        split_idx = int(n * train_ratio)
        train_folders = all_subfolders[:split_idx]
        val_folders = all_subfolders[split_idx:]

        print(f"Total subdirectories: {n}")
        print(f"Training folders ({len(train_folders)}): {train_folders[:5]} ...")
        print(f"Validation folders ({len(val_folders)}): {val_folders[:5]} ...")
    else:
        # Three-way split: training, validation, and test
        train_count = int(n * train_ratio)
        val_count = int(n * val_ratio)
        test_count = n - train_count - val_count  # remainder goes to test
        train_folders = all_subfolders[:train_count]
        val_folders = all_subfolders[train_count:train_count + val_count]
        test_folders = all_subfolders[train_count + val_count:]

        print(f"Total subdirectories: {n}")
        print(f"Training folders ({len(train_folders)}): {train_folders[:5]} ...")
        print(f"Validation folders ({len(val_folders)}): {val_folders[:5]} ...")
        print(f"Test folders ({len(test_folders)}): {test_folders[:5]} ...")

    # Create destination directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    if test_dir is not None:
        os.makedirs(test_dir, exist_ok=True)

    # Build a list of tasks for multiprocessing
    tasks = []
    for folder in train_folders:
        tasks.append((folder, train_dir, raw_data_dir, copy_mode))
    for folder in val_folders:
        tasks.append((folder, val_dir, raw_data_dir, copy_mode))
    if test_dir is not None:
        for folder in test_folders:
            tasks.append((folder, test_dir, raw_data_dir, copy_mode))

    # Process folders either sequentially or in parallel
    if use_multiprocessing:
        with Pool(num_workers) as pool:
            pool.map(_process_folder, tasks)
    else:
        for task in tasks:
            _process_folder(task)

    print("Split complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for trajectory generator.")
    parser.add_argument(
        "--layer",
        type=str,
        default="6k"
    )
    # Define paths for raw data, training, validation, and test directories
    args = parser.parse_args()
    layer = args.layer
    RAW_DATA_DIR = os.path.expanduser("~/data/scaling_law/scaling_law_%s" % layer)
    TRAIN_DIR = os.path.expanduser("~/data/reparke2e_sl/%s/train/" % layer)
    VAL_DIR = os.path.expanduser("~/data/reparke2e_sl/%s/val/")
    TEST_DIR = None

    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2
    # Note: When test_dir is provided, test ratio is computed as 1 - TRAIN_RATIO - VAL_RATIO

    COPY_MODE = True
    USE_MULTIPROCESSING = True  # Enable multiprocessing
    NUM_WORKERS = 15  # Adjust based on your system's CPU cores

    # Set a random seed for reproducibility
    random.seed(42)

    # Execute the split function with test split enabled and multiprocessing option
    split_folders(
        raw_data_dir=RAW_DATA_DIR,
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        test_dir=TEST_DIR,  # Provide test directory to enable three-way split
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        copy_mode=COPY_MODE,
        use_multiprocessing=USE_MULTIPROCESSING,
        num_workers=NUM_WORKERS
    )
