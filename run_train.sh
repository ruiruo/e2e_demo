# Initialize conda
source ~/.bashrc

# Activate your environment
conda activate reparke2e

# Navigate to your project directory
cd "/home/shaoqian.li/reparke2e/"

# Run the Python script in the background
nohup python train.py > train.output &
