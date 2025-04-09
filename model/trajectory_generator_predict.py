from model.trajectory_generator_model import TrajectoryGenerator
from utils.config import Configuration, InferenceConfiguration
from utils.trajectory_utils import detokenize_traj_waypoints
import pytorch_lightning as pl
import numpy as np
import torch
import json


class TrajectoryPredictModule(pl.LightningModule):
    def __init__(self, infer_cfg: InferenceConfiguration, train_cfg: Configuration, device: str):
        super().__init__()
        self.inf_cfg = infer_cfg
        self.train_cfg = train_cfg
        self.infer_device = torch.device('cuda') if device == "gpu" else torch.device('cpu')
        self.model = TrajectoryGenerator(self.train_cfg)
        self.load_model(self.inf_cfg.model_ckpt_path)
        with open(self.train_cfg.detokenizer, "r") as f:
            self.detokenizer = json.load(f)

    def load_model(self, ck_path):
        ckpt = torch.load(ck_path, weights_only=False)
        state_dict = {k.replace('gen_model.', ''): v for k, v in ckpt['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.to(self.infer_device)
        self.model.eval()

    def inference_batch(self, batch):
        """
        This method is automatically called for each batch during the prediction loop.
        """
        # data = {"input_ids", "labels", "agent_info", "ego_info", "goal"}
        # Use device conversion for each item.
        _ = {k: v.to(self.infer_device) for k, v in batch.items()}

        # Autoregressive prediction: let the model generate tokens one by one.
        pred_traj_points = self.model.predict(_, predict_token_num=self.train_cfg.max_frame)
        outputs = []
        for each in pred_traj_points:
            outputs.append(each.tolist())
        return outputs

    def predict(self, data):
        output = []
        for each in data:
            pred = self.inference_batch(each)
            after_detokenize = detokenize_traj_waypoints(
                pred,
                self.detokenizer,
                self.train_cfg.bos_token,
                self.train_cfg.eos_token,
                self.train_cfg.pad_token,
            )
            output.append(after_detokenize)
        return np.array(output)

    def test(self, data):
        """
        Tests the model in free-running (autoregressive) mode. It returns the predicted trajectories,
        the ground truth trajectories, and the agent trajectories.
        """
        pred_traj = []
        label_traj = []
        agents_traj = []
        for each in data:
            pred = self.inference_batch(each)
            for batch in pred:
                pred_after_detokenize = detokenize_traj_waypoints(
                    batch,
                    self.detokenizer,
                    self.train_cfg.bos_token,
                    self.train_cfg.eos_token,
                    self.train_cfg.pad_token,
                )
                pred_traj.append(pred_after_detokenize)
            for batch in each["labels"]:
                label_after_detokenize = detokenize_traj_waypoints(
                    batch,
                    self.detokenizer,
                    self.train_cfg.bos_token,
                    self.train_cfg.eos_token,
                    self.train_cfg.pad_token,
                )
                label_traj.append(label_after_detokenize)
            for batch in each["agent_info"]:
                _agents_traj = []
                for agent in batch.transpose(1, 0):
                    agent_traj = agent[:, 0]
                    agent_traj = detokenize_traj_waypoints(
                        agent_traj.tolist(),
                        self.detokenizer,
                        self.train_cfg.bos_token,
                        self.train_cfg.eos_token,
                        self.train_cfg.pad_token,
                    )
                    _agents_traj.append(agent_traj[1:])
                _agents_traj = np.array(_agents_traj)
                agents_traj.append(_agents_traj)
        # pred_traj -> list of predicted trajectories [T, 2]
        # label_traj -> list of true trajectories [T, 2]
        # agents_traj -> list of agents trajectories in the map [n, T, 2]
        return pred_traj, label_traj, agents_traj

    def test_teacher_forcing(self, data):
        """
        Tests the model using teacher forcing (i.e. feeding in the teacher tokens as inputs)
        to create trajectories. This method mimics the normal generation structure but uses
        the provided ground truth tokens rather than letting the model predict from its own outputs.

        Returns:
          - teacher_pred_traj: List of trajectories produced by teacher forcing (after detokenization)
          - label_traj: List of ground truth trajectories (after detokenization)
          - agents_traj: List of agents trajectories in the map
        """
        teacher_pred_traj = []
        label_traj = []
        agents_traj = []

        for each in data:
            # Move the batch to the designated inference device.
            batch = {k: v.to(self.infer_device) for k, v in each.items()}

            # Call the model with the entire data dictionary.
            # This forward call performs a teacher forcing pass (using input_ids as teacher signals).
            pred_logits, _, _ = self.model(batch)

            # Greedy decode: choose the token with maximum logit at each step.
            teacher_pred_tokens = pred_logits.argmax(dim=-1)  # shape: (batch_size, sequence_length)

            # Convert the teacher forced token predictions into trajectories.
            for tokens in teacher_pred_tokens:
                tokens_list = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
                teacher_pred = detokenize_traj_waypoints(
                    tokens_list,
                    self.detokenizer,
                    self.train_cfg.bos_token,
                    self.train_cfg.eos_token,
                    self.train_cfg.pad_token,
                )
                teacher_pred_traj.append(teacher_pred)

            # Also, convert the ground truth labels into trajectories.
            for label_tokens in batch["labels"]:
                label_tokens = label_tokens.tolist() if isinstance(label_tokens, torch.Tensor) else label_tokens
                teacher_label = detokenize_traj_waypoints(
                    label_tokens,
                    self.detokenizer,
                    self.train_cfg.bos_token,
                    self.train_cfg.eos_token,
                    self.train_cfg.pad_token,
                )
                label_traj.append(teacher_label)

            # Process agent trajectories similar to the normal test function.
            for agent_batch in batch["agent_info"]:
                _agents_traj = []
                # Transpose agent_batch: expected shape (seq_len, n_agents, features)
                for agent in agent_batch.transpose(1, 0):
                    # Assuming the first feature is the token; extract the token trajectory.
                    agent_traj = agent[:, 0]
                    agent_traj = detokenize_traj_waypoints(
                        agent_traj.tolist(),
                        self.detokenizer,
                        self.train_cfg.bos_token,
                        self.train_cfg.eos_token,
                        self.train_cfg.pad_token,
                    )
                    # Remove the initial token if needed.
                    _agents_traj.append(agent_traj[1:])
                _agents_traj = np.array(_agents_traj)
                agents_traj.append(_agents_traj)

        return teacher_pred_traj, label_traj, agents_traj
