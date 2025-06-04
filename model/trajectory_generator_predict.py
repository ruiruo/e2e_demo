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
        Autoregressive testing. Returns:
          - pred_traj:        list of detokenized XY trajectories
          - label_traj:       list of detokenized XY trajectories
          - agents_traj:      list of agent trajectories
          - pred_tokens_all:  list of raw predicted token arrays
          - label_tokens_all: list of raw ground-truth token arrays
        """
        pred_traj = []
        label_traj = []
        agents_traj = []

        pred_tokens_all = []
        label_tokens_all = []

        for batch_dict in data:
            # 1) Autoregressive token-level inference
            batch_pred_tokens = self.inference_batch(batch_dict)
            for single_pred_tokens in batch_pred_tokens:
                # Save tokens for outside metrics (e.g., token-level accuracy)
                pred_tokens_all.append(single_pred_tokens)

                # Detokenize for coordinate-based metrics or visualization
                detokenize_pred = detokenize_traj_waypoints(
                    single_pred_tokens,
                    self.detokenizer,
                    self.train_cfg.bos_token,
                    self.train_cfg.eos_token,
                    self.train_cfg.pad_token
                )
                pred_traj.append(detokenize_pred)

            # 2) Collect ground-truth token sequences and detokenize
            for single_label_tokens in batch_dict["labels"]:
                label_tokens_list = single_label_tokens.tolist()
                label_tokens_all.append(label_tokens_list)

                detok_label = detokenize_traj_waypoints(
                    label_tokens_list,
                    self.detokenizer,
                    self.train_cfg.bos_token,
                    self.train_cfg.eos_token,
                    self.train_cfg.pad_token
                )
                label_traj.append(detok_label)

            # 3) Detokenize agent trajectories
            for batch_agents in batch_dict["agent_info"]:
                _agents_traj = []
                # agent_batch shape: (seq_len, n_agents, features)
                # We transpose to get (n_agents, seq_len, features)
                for agent in batch_agents.transpose(1, 0):
                    agent_tokens = agent[:, 0]
                    agent_detok = detokenize_traj_waypoints(
                        agent_tokens.tolist(),
                        self.detokenizer,
                        self.train_cfg.bos_token,
                        self.train_cfg.eos_token,
                        self.train_cfg.pad_token
                    )
                    _agents_traj.append(agent_detok[1:])
                agents_traj.append(np.array(_agents_traj))

        return pred_traj, label_traj, agents_traj, pred_tokens_all, label_tokens_all

    def test_teacher_forcing(self, data):
        """
        Teacher-forcing test. Returns:
          - teacher_pred_traj
          - label_traj
          - agents_traj
          - pred_tokens_all
          - label_tokens_all
        """
        teacher_pred_traj = []
        label_traj = []
        agents_traj = []

        pred_tokens_all = []
        label_tokens_all = []

        for batch_dict in data:
            batch_dict = {k: v.to(self.infer_device) for k, v in batch_dict.items()}
            pred_logits, _, _ = self.model(batch_dict)

            # Greedy decode to get teacher-forced tokens
            teacher_pred_tokens = pred_logits.argmax(dim=-1)  # (B, seq_len)
            for tokens in teacher_pred_tokens:
                tokens_list = tokens.tolist()
                pred_tokens_all.append(tokens_list)

                detok_pred = detokenize_traj_waypoints(
                    tokens_list,
                    self.detokenizer,
                    self.train_cfg.bos_token,
                    self.train_cfg.eos_token,
                    self.train_cfg.pad_token
                )
                teacher_pred_traj.append(detok_pred)

            # Ground-truth
            for label_tokens in batch_dict["labels"]:
                label_tokens_list = label_tokens.tolist()
                label_tokens_all.append(label_tokens_list)

                detok_label = detokenize_traj_waypoints(
                    label_tokens_list,
                    self.detokenizer,
                    self.train_cfg.bos_token,
                    self.train_cfg.eos_token,
                    self.train_cfg.pad_token
                )
                label_traj.append(detok_label)

            # Agents
            for agent_batch in batch_dict["agent_info"]:
                _agents_traj = []
                for agent in agent_batch.transpose(1, 0):
                    agent_tokens = agent[:, 0]
                    agent_detok = detokenize_traj_waypoints(
                        agent_tokens.tolist(),
                        self.detokenizer,
                        self.train_cfg.bos_token,
                        self.train_cfg.eos_token,
                        self.train_cfg.pad_token
                    )
                    _agents_traj.append(agent_detok[1:])
                agents_traj.append(np.array(_agents_traj))

        return teacher_pred_traj, label_traj, agents_traj, pred_tokens_all, label_tokens_all
