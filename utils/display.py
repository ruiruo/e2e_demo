import matplotlib.pyplot as plt
import numpy as np
import os

def plot_and_save_trajectories(pred_traj, label_traj, agents_traj, save_dir="trajectory_figures"):
    """
    Draws and saves figures for each trajectory sample.

    Args:
        pred_traj (list): List of predicted trajectories [T, 2].
        label_traj (list): List of ground truth trajectories [T, 2].
        agents_traj (list): List of agents trajectories, where each element is an array of shape [n, T, 2].
        save_dir (str): Directory where figures will be saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, (pred, label, agents) in enumerate(zip(pred_traj, label_traj, agents_traj)):
        plt.figure(figsize=(8, 6))
        pred = np.array(pred)
        label = np.array(label)

        # Plot ground truth (in red) and predicted (in blue) trajectories
        plt.plot(label[:, 0], label[:, 1], 'ro-', label="Ground Truth")
        plt.plot(pred[:, 0], pred[:, 1], 'bo-', label="Predicted")

        # Add step numbers for ground truth trajectory
        for step_idx, (lx, ly) in enumerate(label):
            plt.text(lx, ly, str(step_idx), color='red', fontsize=9)

        # Add step numbers for predicted trajectory
        for step_idx, (px, py) in enumerate(pred):
            plt.text(px, py, str(step_idx), color='blue', fontsize=9)

        # Plot agents trajectories if available
        if agents.size > 0:
            for a_idx, agent in enumerate(agents):
                agent = np.array(agent)
                # Plot each agent with a dashed line
                if a_idx == 0:
                    plt.plot(agent[:, 0], agent[:, 1], 'k--', label="Agent")
                else:
                    plt.plot(agent[:, 0], agent[:, 1], 'k--')

        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.title(f"Trajectory Sample {idx}")
        plt.legend()

        save_path = os.path.join(save_dir, f"trajectory_{idx}.png")
        plt.savefig(save_path)
        plt.close()