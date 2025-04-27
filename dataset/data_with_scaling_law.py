import os
import random
import shutil

data_root  = "/home/shaoqian.li/data/raw_data"

# Number of folders to sample for the test set
TEST_COUNT = 1000

# Desired sizes for scaling law datasets
SCALING_SIZES = {
    'scaling_law_6k': 6000,
    'scaling_law_12k': 12000,
    'scaling_law_24k': 24000,
}
parent_dir = os.path.join(os.path.dirname(data_root), 'scaling_law')

# Define second‐level directories under parent_dir
all_dir      = os.path.join(parent_dir, 'all')
test_dir     = os.path.join(parent_dir, 'test')
scaling_dirs = {
    name: os.path.join(parent_dir, name)
    for name in SCALING_SIZES
}

# 1. List all raw subdirectories under data_root
raw_subdirs = [
    d for d in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, d))
]

# 2. Prepare the test set under parent_dir/test
os.makedirs(test_dir, exist_ok=True)
existing_tests = {
    d for d in os.listdir(test_dir)
    if os.path.isdir(os.path.join(test_dir, d))
}
if len(existing_tests) < TEST_COUNT:
    needed = TEST_COUNT - len(existing_tests)
    candidates = [d for d in raw_subdirs if d not in existing_tests]
    samples = random.sample(candidates, needed)
    for sub in samples:
        shutil.copytree(
            os.path.join(data_root, sub),
            os.path.join(test_dir, sub)
        )
    existing_tests.update(samples)
    print(f"Copied {needed} folders to '{test_dir}'.")
else:
    print(f"'{test_dir}' already contains {len(existing_tests)} folders; skipping test extraction.")

# 3. Determine remaining subdirs (raw minus test)
remaining = [d for d in raw_subdirs if d not in existing_tests]

# 4. Build 'all' directory under parent_dir, containing only remaining
if os.path.isdir(all_dir):
    shutil.rmtree(all_dir)
os.makedirs(all_dir, exist_ok=True)
for sub in remaining:
    shutil.copytree(
        os.path.join(data_root, sub),
        os.path.join(all_dir, sub)
    )
print(f"Copied {len(remaining)} remaining folders into '{all_dir}' (excluding test).")

# 5. Create each scaling_law_* directory by sampling from 'all'
for name, size in SCALING_SIZES.items():
    target = scaling_dirs[name]
    os.makedirs(target, exist_ok=True)

    existing = os.listdir(target)
    if len(existing) >= size:
        print(f"'{name}' already has {len(existing)} folders; skipping.")
        continue

    if size > len(remaining):
        raise ValueError(f"Requested {size} for '{name}', but only {len(remaining)} available.")

    samples = random.sample(remaining, size)
    for sub in samples:
        shutil.copytree(
            os.path.join(all_dir, sub),
            os.path.join(target, sub)
        )
    print(f"Created '{name}' with {size} folders.")

# 6. Summary
print("\nProcessing complete:")
print(f" → all (no test) in: {all_dir} ({len(os.listdir(all_dir))} folders)")
print(f" → test in:         {test_dir} ({len(os.listdir(test_dir))} folders)")
for name in SCALING_SIZES:
    path = scaling_dirs[name]
    print(f" → {name} in: {path} ({len(os.listdir(path))} folders)")
