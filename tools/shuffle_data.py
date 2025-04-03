import os
import shutil
import random


def split_folders(raw_data_dir: str, train_dir: str, val_dir: str, test_dir: str = None,
                  train_ratio: float = 0.8, val_ratio: float = 0.2, copy_mode: bool = True):
    """
    Randomly shuffle and split subdirectories from raw_data_dir into training, validation,
    and optionally test sets, then copy (or move/link) them into train_dir, val_dir, and test_dir.

    When test_dir is not provided, the function performs a two-way split (training/validation)
    using train_ratio for training and the rest for validation. If test_dir is provided, then:
      - training gets train_ratio of the data,
      - validation gets val_ratio of the data, and
      - test gets the remaining folders (i.e. 1 - train_ratio - val_ratio).

    :param raw_data_dir: Directory containing raw data subdirectories (e.g., ~/data/raw_data/)
    :param train_dir: Destination directory for training data (e.g., ~/data/train/)
    :param val_dir: Destination directory for validation data (e.g., ~/data/val/)
    :param test_dir: Destination directory for test data (e.g., ~/data/test/). If None, no test split is performed.
    :param train_ratio: Fraction of subdirectories to use for training.
    :param val_ratio: Fraction of subdirectories to use for validation when test_dir is provided.
                      (When test_dir is None, validation gets all remaining folders.)
    :param copy_mode: If True, copy folders; if False, move them.
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

    # Helper function to process each folder
    def process_folder(folder, destination):
        src_path = os.path.join(raw_data_dir, folder)
        dst_path = os.path.join(destination, folder)
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)  # Remove existing folder to prevent conflicts
        if copy_mode:
            shutil.copytree(src_path, dst_path)
        else:
            shutil.move(src_path, dst_path)

    # Process training folders
    for folder in train_folders:
        process_folder(folder, train_dir)

    # Process validation folders
    for folder in val_folders:
        process_folder(folder, val_dir)

    # Process test folders if applicable
    if test_dir is not None:
        for folder in test_folders:
            process_folder(folder, test_dir)

    print("Split complete!")


if __name__ == "__main__":
    # Define paths for raw data, training, validation, and test directories
    RAW_DATA_DIR = os.path.expanduser("~/data/raw_data/")
    TRAIN_DIR = os.path.expanduser("~/data/train/")
    VAL_DIR = os.path.expanduser("~/data/val/")
    TEST_DIR = os.path.expanduser("~/data/test/")  # New test directory

    # Set the ratios for three splits: 75% training, 20% validation, and implicitly 5% test
    TRAIN_RATIO = 0.75
    VAL_RATIO = 0.20
    # Note: When test_dir is provided, test ratio is computed as 1 - TRAIN_RATIO - VAL_RATIO

    COPY_MODE = True

    # Set a random seed for reproducibility
    random.seed(42)

    # Execute the split function with test split enabled
    split_folders(
        raw_data_dir=RAW_DATA_DIR,
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        test_dir=TEST_DIR,  # Provide test directory to enable three-way split
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        copy_mode=COPY_MODE
    )
