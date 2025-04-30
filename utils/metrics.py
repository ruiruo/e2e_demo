from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import torch

from utils.config import Configuration
from utils.trajectory_utils import TrajectoryDistance, detokenize_traj_waypoints


class TrajectoryGeneratorMetric:
    """Compute distance metrics between predicted & GT token trajectories.

    Handles variable‑length detokenized polylines by **length‑aligning** the two
    sequences before feeding them into `TrajectoryDistance`.
    """

    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        with open(self.cfg.detokenizer, "r") as f:
            self.detokenizer = json.load(f)

    def calculate_distance(
        self,
        pred_tokens: torch.Tensor,  # (B,T,V) logits **or** (B,T) tokens
        true_tokens: torch.Tensor,  # (B,T)
    ) -> Dict[str, float]:
        """Return averaged L2 / Hausdorff / Fourier distances (if computable)."""

        # 0. ensure CPU numpy for heavy ops
        pred_waypoints = self._tokens_to_np(pred_tokens)  # List[np.ndarray]
        gt_waypoints = self._tokens_to_np(true_tokens)    # List[np.ndarray]

        assert len(pred_waypoints) == len(gt_waypoints)
        l2_vals, haus_vals, fourier_vals = [], [], []

        for p_xy, t_xy in zip(pred_waypoints, gt_waypoints):
            # 1. align lengths (TrajectoryDistance expects equal length)
            p_xy, t_xy = self._align_lengths(p_xy, t_xy)
            if p_xy.size == 0:
                continue

            dist_obj = TrajectoryDistance(p_xy, t_xy)
            l2_vals.append(dist_obj.get_l2_distance())
            if dist_obj.get_len() > 1:
                haus_vals.append(dist_obj.get_haus_distance())
                fourier_vals.append(dist_obj.get_fourier_difference())

        out: Dict[str, float] = {}
        if l2_vals:
            out["L2_distance"] = float(np.mean(l2_vals))
        if haus_vals:
            out["hausdorff_distance"] = float(np.mean(haus_vals))
        if fourier_vals:
            out["fourier_difference"] = float(np.mean(fourier_vals))
        return out


    def _tokens_to_np(self, x: torch.Tensor) -> List[np.ndarray]:
        """Convert (B,T,V) logits **or** (B,T) tokens -> list of N×2 np arrays."""
        if x.ndim == 3:
            x = torch.argmax(x, dim=-1)  # logits -> tokens
        x = x.long().cpu()
        out: List[np.ndarray] = []
        for seq in x:  # iterate over batch dim
            pts_np = detokenize_traj_waypoints(
                seq,
                self.detokenizer,
                self.cfg.bos_token,
                self.cfg.eos_token,
                self.cfg.pad_token,
            )
            out.append(pts_np.astype(np.float64))  # ensure float
        return out

    @staticmethod
    def _align_lengths(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Truncate the longer array so that `len(a)==len(b)`.

        If either becomes length < 1 after truncation, returns (empty, empty)
        so caller can skip.
        """
        n = min(len(a), len(b))
        if n < 1:
            return np.empty(0), np.empty(0)
        return a[:n], b[:n]
