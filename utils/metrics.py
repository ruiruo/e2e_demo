from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from shapely.geometry import LineString

from utils.config import Configuration
from utils.trajectory_utils import detokenize_traj_waypoints, TrajectoryDistance


class TrajectoryGeneratorMetric:
    """Distance metrics for trajectory generation.

    *   Local metrics: L2, Hausdorff, Fourier (via ``TrajectoryDistance``)
    *   Global metric: strip area (m²) and mean strip width (m)
    """

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        with open(cfg.detokenizer, "r") as f:
            self.detokenizer = json.load(f)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_distance(
        self,
        pred_tokens: torch.Tensor,  # (B,T,V) logits **or** (B,T) tokens
        true_tokens: torch.Tensor,  # (B,T)
        n_samples: int = 200,
    ) -> Dict[str, float]:
        """Return averaged metrics over a batch."""
        pred_pts = self._tokens_to_np(pred_tokens)
        gt_pts = self._tokens_to_np(true_tokens)

        l2_vals, haus_vals, fourier_vals = [], [], []
        area_vals, width_vals = [], []

        for p_xy, t_xy in zip(pred_pts, gt_pts):
            p_xy, t_xy = map(self._clean_xy, (p_xy, t_xy))
            if p_xy.size == 0 or t_xy.size == 0:
                continue
            p_xy, t_xy = self._align_lengths(p_xy, t_xy)
            if p_xy.size == 0:
                continue

            # ── Local metrics ──────────────────────────────────────────
            dist_obj = TrajectoryDistance(p_xy, t_xy)
            l2_vals.append(dist_obj.get_l2_distance())
            if dist_obj.get_len() > 1:
                haus_vals.append(dist_obj.get_haus_distance())
                fourier_vals.append(dist_obj.get_fourier_difference())

            # ── Global strip‑area metric ──────────────────────────────
            strip_area, mean_width = self._strip_area(t_xy, p_xy, n_samples)
            area_vals.append(strip_area)
            width_vals.append(mean_width)

        out: Dict[str, float] = {}
        if l2_vals:
            out["L2_distance"] = float(np.mean(l2_vals))
        if haus_vals:
            out["hausdorff_distance"] = float(np.mean(haus_vals))
        if fourier_vals:
            out["fourier_difference"] = float(np.mean(fourier_vals))
        if area_vals:
            out["strip_area"] = float(np.mean(area_vals))
            out["mean_strip_width"] = float(np.mean(width_vals))
        return out

    # ------------------------------------------------------------------
    # Converters & cleaners
    # ------------------------------------------------------------------

    def _tokens_to_np(self, x: torch.Tensor) -> List[np.ndarray]:
        """Convert logits/tokens → list[N×2] float arrays."""
        if x.ndim == 3:
            x = torch.argmax(x, dim=-1)
        x = x.long().cpu()
        out: List[np.ndarray] = []
        for seq in x:
            pts_np = detokenize_traj_waypoints(
                seq,
                self.detokenizer,
                self.cfg.bos_token,
                self.cfg.eos_token,
                self.cfg.pad_token,
            ).astype(np.float64)
            out.append(pts_np)
        return out

    @staticmethod
    def _clean_xy(xy: np.ndarray) -> np.ndarray:
        """Remove NaN rows & consecutive duplicates."""
        if xy.ndim != 2 or xy.shape[1] != 2:
            return np.empty((0, 2))
        mask = ~np.any(np.isnan(xy), axis=1)
        xy = xy[mask]
        if len(xy) < 2:
            return xy
        diff = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        keep = np.insert(diff > 1e-8, 0, True)
        return xy[keep]

    @staticmethod
    def _align_lengths(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = min(len(a), len(b))
        return (a[:n], b[:n]) if n > 0 else (np.empty((0, 2)), np.empty((0, 2)))

    # ------------------------------------------------------------------
    # Strip‑area computation
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_area(
        gt_xy: np.ndarray,
        pred_xy: np.ndarray,
        n_samples: int = 200,
    ) -> Tuple[float, float]:
        """Compute (area, mean_width).

        * area: ∫ |e(s)| ds  (m²)
        * mean_width: area / length(gt) (m)
        """
        gt_ls, pr_ls = LineString(gt_xy), LineString(pred_xy)
        gt_len = gt_ls.length
        if gt_len == 0:
            return 0.0, 0.0
        s_vals = np.linspace(0.0, gt_len, n_samples)
        gt_pts = np.array([gt_ls.interpolate(s).coords[0] for s in s_vals])
        pr_proj = np.array([
            pr_ls.interpolate(pr_ls.project(gt_ls.interpolate(s))).coords[0]
            for s in s_vals
        ])
        e = np.linalg.norm(gt_pts - pr_proj, axis=1)
        area = float(np.trapz(e, s_vals))
        return area, area / gt_len
