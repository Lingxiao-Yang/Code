
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from melt_dataset import MeltDataset

CANON_REGIONS = ("16s-23s", "23s-5s", "Thr-Tyr")


def _clean_xy(t: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove NaNs/inf, sort by temperature, and drop duplicates."""
    t = np.asarray(t, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    mask = np.isfinite(t) & np.isfinite(v)
    t = t[mask]
    v = v[mask]
    if t.size == 0:
        return t, v
    # sort and unique
    order = np.argsort(t)
    t = t[order]
    v = v[order]
    # remove duplicate temps (keep first)
    if t.size >= 2:
        uniq_mask = np.concatenate([[True], np.diff(t) != 0])
        t = t[uniq_mask]
        v = v[uniq_mask]
    return t, v


def _compute_derivative(temp: np.ndarray, y: np.ndarray, poly_order: int = 2) -> np.ndarray:
    """Savitzky–Golay first derivative (like HRMAnalyzer)."""
    temp, y = _clean_xy(temp, y)
    if len(temp) < 5:
        return np.zeros_like(temp, dtype=float)
    avg_step = float(np.mean(np.diff(temp)))
    window = max(5, int(round(1.0 / max(avg_step, 1e-6))))
    if window % 2 == 0:
        window += 1
    if window >= len(temp):
        window = max(3, len(temp) - 1 if (len(temp) - 1) % 2 else len(temp) - 2)
    if poly_order >= window:
        poly_order = max(1, window - 1)
    d = savgol_filter(y, window, poly_order, deriv=1, delta=avg_step)
    return -d  # negative derivative to match plots


def _align_and_normalize(t1: np.ndarray, v1: np.ndarray, t2: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align by interpolation to common grid over overlap; z-score normalize."""
    t1, v1 = _clean_xy(t1, v1)
    t2, v2 = _clean_xy(t2, v2)
    if t1.size < 2 or t2.size < 2:
        return np.array([]), np.array([]), np.array([])
    lo = max(t1.min(), t2.min())
    hi = min(t1.max(), t2.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([]), np.array([]), np.array([])
    grid = np.linspace(lo, hi, 400)
    a = np.interp(grid, t1, v1)
    b = np.interp(grid, t2, v2)
    def z(x):
        s = x.std()
        return (x - x.mean()) if s == 0 else (x - x.mean()) / s
    return grid, z(a), z(b)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float('nan')
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float('nan')
    return float(np.dot(a, b) / denom)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float('nan')
    return float(np.sqrt(np.mean((a - b) ** 2)))


@dataclass
class MeltSimilarity:
    pearson_r: float
    cosine: float
    rmse: float


class MeltCurveResolver:
    """
    Pre-computes simulated melt *derivative* curves for all bacteria across regions,
    and provides single-region and multi-region scoring.
    """
    def __init__(self, dataset_root: str | Path):
        self.ds = MeltDataset(dataset_root)
        self._sim_deriv: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        for name in self.ds.bacteria_names:
            self._sim_deriv[name] = {}
            m = self.ds[name]
            for region in CANON_REGIONS:
                try:
                    t = np.asarray(m.wave["temp"][region], dtype=float)
                    y = np.asarray(m.wave["RFU"][region], dtype=float)
                    d = _compute_derivative(t, y)
                    self._sim_deriv[name][region] = (t, d)
                except Exception:
                    continue

    def score_candidates(
        self,
        exp_temp: np.ndarray,
        exp_deriv: np.ndarray,
        region: str,
        candidate_names: List[str],
        out_dir: Path,
        plot_title: str,
    ) -> Tuple[Optional[str], Dict[str, MeltSimilarity]]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # clean experimental curve (drop NaNs etc.); if too small, skip
        exp_temp, exp_deriv = _clean_xy(exp_temp, exp_deriv)
        if exp_temp.size < 5:
            return None, {}

        metrics: Dict[str, MeltSimilarity] = {}
        overlay_t = None
        overlay_curves: Dict[str, np.ndarray] = {}

        for name in candidate_names:
            if name not in self._sim_deriv or region not in self._sim_deriv[name]:
                continue
            t_sim, d_sim = self._sim_deriv[name][region]
            grid, a, b = _align_and_normalize(exp_temp, exp_deriv, t_sim, d_sim)
            if grid.size == 0:
                continue
            overlay_t = grid
            overlay_curves[name] = b
            metrics[name] = MeltSimilarity(
                pearson_r=_pearson(a, b),
                cosine=_cosine(a, b),
                rmse=_rmse(a, b),
            )
            overlay_curves["_exp_"] = a

        best = None
        if metrics:
            best = max(metrics.items(), key=lambda kv: (kv[1].pearson_r, -kv[1].rmse))[0]

        if overlay_t is not None and overlay_curves:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.plot(overlay_t, overlay_curves.get("_exp_", np.array([])), label="Experimental (z)", linewidth=2.5)
            for name, sim in metrics.items():
                ax.plot(overlay_t, overlay_curves[name], label=f"{name} (r={sim.pearson_r:.3f}, RMSE={sim.rmse:.3f})", alpha=0.9)
            ax.set_title(plot_title)
            ax.set_xlabel("Temperature (°C)")
            ax.set_ylabel("Standardized -dF/dT")
            ax.grid(True, alpha=0.5)
            ax.legend(fontsize='small')
            out_path = out_dir / (plot_title.replace(' ', '_').replace('/', '-') + ".png")
            fig.tight_layout()
            fig.savefig(out_path, dpi=300)
            plt.close(fig)

        return best, metrics

    def score_multi_region(
        self,
        exp_region_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
        candidate_names: List[str],
        out_root: Path,
        title_prefix: str,
    ) -> Tuple[Optional[str], Dict[str, Dict[str, MeltSimilarity]], Dict[str, float]]:
        per_region_metrics: Dict[str, Dict[str, MeltSimilarity]] = {}
        aggregate: Dict[str, float] = {c: 0.0 for c in candidate_names}

        for region, (t_exp, y_exp) in exp_region_curves.items():
            try:
                out_dir = Path(out_root) / f"{region.replace(' ', '_')}"
                best, metrics = self.score_candidates(
                    exp_temp=t_exp, exp_deriv=y_exp, region=region,
                    candidate_names=candidate_names,
                    out_dir=out_dir,
                    plot_title=f"{title_prefix} [{region}]"
                )
                per_region_metrics[region] = metrics
                for cand, sim in metrics.items():
                    if np.isfinite(sim.pearson_r):
                        aggregate[cand] = aggregate.get(cand, 0.0) + float(sim.pearson_r)
            except Exception:
                # robust to bad region data
                continue

        best_overall = None
        if any(np.isfinite(v) for v in aggregate.values()):
            best_overall = max(aggregate.items(), key=lambda kv: kv[1])[0]

        return best_overall, per_region_metrics, aggregate
