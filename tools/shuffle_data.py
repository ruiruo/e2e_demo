import os
import shutil
import random


def split_folders(raw_data_dir: str, train_dir: str, val_dir: str, train_ratio: float = 0.8,
                  copy_mode: bool = True):
    """
    Randomly shuffle and split subdirectories from raw_data_dir into training and validation sets,
    then copy (or move/link) them into train_dir and val_dir.

    :param raw_data_dir: Directory containing raw data subdirectories (e.g., ~/data/raw_data/)
    :param train_dir: Destination directory for training data (e.g., ~/data/training_dir/)
    :param val_dir: Destination directory for validation data (e.g., ~/data/validation_dir/)
    :param train_ratio: Fraction of subdirectories to use for training (default 0.8 for 80%)
    :param copy_mode: If True, copy folders; if False, move them (or modify to use symlinks)
    """
    # List all subdirectories in raw_data_dir
    all_subfolders = []
    for entry in os.listdir(raw_data_dir):
        full_path = os.path.join(raw_data_dir, entry)
        if os.path.isdir(full_path):
            all_subfolders.append(entry)

    if not all_subfolders:
        print(f"No subdirectories found in {raw_data_dir}")
        return

    # Shuffle the list of subdirectories
    random.shuffle(all_subfolders)
    split_idx = int(len(all_subfolders) * train_ratio)
    train_folders = all_subfolders[:split_idx]
    val_folders = all_subfolders[split_idx:]

    # Print some info for verification
    print(f"Total subdirectories: {len(all_subfolders)}")
    print(f"Training folders ({len(train_folders)}): {train_folders[:5]} ...")
    print(f"Validation folders ({len(val_folders)}): {val_folders[:5]} ...")

    # Create destination directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Copy (or move) training folders to the training directory
    for folder in train_folders:
        src_path = os.path.join(raw_data_dir, folder)
        dst_path = os.path.join(train_dir, folder)
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)  # Remove existing folder to prevent conflicts
        if copy_mode:
            shutil.copytree(src_path, dst_path)
        else:
            shutil.move(src_path, dst_path)

    # Copy (or move) validation folders to the validation directory
    for folder in val_folders:
        src_path = os.path.join(raw_data_dir, folder)
        dst_path = os.path.join(val_dir, folder)
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        if copy_mode:
            shutil.copytree(src_path, dst_path)
        else:
            shutil.move(src_path, dst_path)

    print("Split complete!")


if __name__ == "__main__":
    # Define paths for raw data, training, and validation directories
    RAW_DATA_DIR = os.path.expanduser("~/data/raw_data/")
    TRAIN_DIR = os.path.expanduser("~/data/train/")
    VAL_DIR = os.path.expanduser("~/data/val/")

    # Set the training set ratio (e.g., 0.8 means 80% training, 20% validation)
    TRAIN_RATIO = 0.8

    COPY_MODE = True

    # Set a random seed for reproducibility
    random.seed(42)

    # Execute the split function
    split_folders(
        raw_data_dir=RAW_DATA_DIR,
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        train_ratio=TRAIN_RATIO,
        copy_mode=COPY_MODE
    )
