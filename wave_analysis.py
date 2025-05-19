import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

ladder_bp = np.array([20,  30,  40,  60,  80, 100, 114, 120, 140, 160, 180, 200,
                    214, 220, 240, 250, 260, 280, 300, 314, 320, 340, 360, 380,
                    400, 414, 420, 440, 460, 480, 500, 514, 520, 540, 560, 580,
                    600, 614, 620, 640, 660, 680, 700, 714, 720, 740, 760, 780,
                    800, 820, 840, 850, 860, 880, 900, 920, 940, 960, 980, 1000,
                    1020,1040,1060,1080,1100,1120,1160,1200])

class CapillaryPeakCalibrator:
    """
    • Detect orange-channel ladder peaks
    • One-by-one pair them with the supplied bp ladder values
      (handling legitimate double peaks by merging the taller one)
    • Build a monotonic index→bp look-up via linear interpolation
    • Apply that mapping to green-channel peaks

    Parameters
    ----------
    orange, green : 1-D ndarray
        Fluorescence traces of ladder and sample.
    ladder_bp     : 1-D ndarray (ascending)
        The expected base-pair ladder (e.g. 68 values).
    smooth_win    : int
        Smoothing window width (default 5).
    baseline_win  : int
        Baseline-removal window (default 800).
    pk_height / pk_prom / pk_dist : float | int
        Thresholds passed to scipy.signal.find_peaks.
    merge_distance : int
        Two peaks ≤ merge_distance samples apart are considered “close”.
    allowed_bp_gap : int
        If those two peaks correspond to ladder bp values
        that differ by ≤ allowed_bp_gap, they may be merged.
    rmse_tol : float
        Still used to sanity-check interpolation error (can stay small).
    debug_dir : str | None
        If not None, PNGs of detected peaks are saved here.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        orange,
        green,
        smooth_win: int = 5,
        baseline_win: int = 800,
        pk_height: float = 200,
        pk_prom: float = 150,
        pk_dist: int = 30,
        merge_distance: int = 80,
        allowed_bp_gap: int = 10,
        rmse_tol: float = 3.0,
        debug_dir: str | None = None,
    ):
        self.orange = np.asarray(orange, dtype=float)
        self.green = np.asarray(green, dtype=float)
        self.ladder_bp = np.asarray(ladder_bp, dtype=float)

        self.smooth_win = smooth_win
        self.baseline_win = baseline_win
        self.pk_height = pk_height
        self.pk_prom = pk_prom
        self.pk_dist = pk_dist
        self.merge_distance = merge_distance
        self.allowed_bp_gap = allowed_bp_gap
        self.rmse_tol = rmse_tol
        self.debug_dir = debug_dir

        # will be filled by run()
        self.ladder_idx = None          # np.ndarray, length == len(ladder_bp)
        self.orange_peaks = None        # DataFrame
        self.green_peaks = None         # DataFrame

    # ==================================================================
    # -------------------     public one-liner     ---------------------
    # ==================================================================
    def run(self):
        """Full pipeline: ladder alignment → green bp assignment."""
        self._align_orange_ladder()
        self._assign_green_bp()
        return self.orange_peaks, self.green_peaks

    # ==================================================================
    # -----------------------  internal helpers  -----------------------
    # ==================================================================
    def _preprocess(self, trace):
        sm = uniform_filter1d(trace, self.smooth_win)
        baseline = uniform_filter1d(sm, self.baseline_win)
        sig = sm - baseline
        sig[sig < 0] = 0
        return sig

    def _detect_peaks(self, y):
        idx, props = find_peaks(
            y,
            height=self.pk_height,
            prominence=self.pk_prom,
            distance=self.pk_dist,
        )
        return idx.astype(int), props["peak_heights"]

    # ------------------------------------------------------------------
    # 1. detect & “clean” the orange ladder peaks
    # ------------------------------------------------------------------
    def _align_orange_ladder(self):
        y_or = self._preprocess(self.orange)
        idx_all, h_all = self._detect_peaks(y_or)

        # optional debug plot
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(18, 4))
            ax.plot(y_or, c="orange")
            ax.scatter(idx_all, h_all, c="red", zorder=3)
            ax.set_title("Raw detected orange peaks")
            fig.savefig(os.path.join(self.debug_dir, "orange_raw.png"))
            plt.close(fig)

        # sort left→right
        idx = idx_all[np.argsort(idx_all)].tolist()
        h = h_all[np.argsort(idx_all)].tolist()

        # sequentially merge legitimate doubles
        i = 0
        while len(idx) > len(self.ladder_bp):
            if i >= len(idx) - 1:
                raise RuntimeError("Too many peaks; unable to resolve doubles.")
            gap = idx[i + 1] - idx[i]
            if gap >= self.merge_distance:
                i += 1
                continue
            bp_gap = abs(self.ladder_bp[i + 1] - self.ladder_bp[i])
            if bp_gap > self.allowed_bp_gap:
                raise RuntimeError(
                    f"Close peaks {idx[i]} & {idx[i+1]} map to "
                    f"{self.ladder_bp[i]} bp and {self.ladder_bp[i+1]} bp "
                    f"(gap {bp_gap} > allowed {self.allowed_bp_gap})."
                )
            # merge → keep taller
            keep_left = h[i] >= h[i + 1]
            idx[i : i + 2] = [idx[i] if keep_left else idx[i + 1]]
            h[i : i + 2] = [h[i] if keep_left else h[i + 1]]
            if i:  # re-check previous pair
                i -= 1

        if len(idx) != len(self.ladder_bp):
            raise RuntimeError(
                f"Ladder mismatch: expected {len(self.ladder_bp)} peaks, "
                f"found {len(idx)} after merging."
            )

        idx = np.asarray(idx)
        h = np.asarray(h)

        # decide orientation: bp should monotonically increase with idx
        if np.corrcoef(idx, self.ladder_bp)[0, 1] < 0:
            self.ladder_bp = self.ladder_bp[::-1]
        self.ladder_idx = idx

        # tiny sanity check (piece-wise linear error)
        interp_bp = np.interp(idx, idx, self.ladder_bp)
        rmse = np.sqrt(((interp_bp - self.ladder_bp) ** 2).mean())
        if rmse > self.rmse_tol:
            raise RuntimeError(
                f"Interpolation sanity RMSE {rmse:.2f} bp > tol {self.rmse_tol}."
            )

        self.orange_peaks = pd.DataFrame(
            dict(idx=idx, height=h, bp=self.ladder_bp)
        )
        
        if self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(18, 4))
            ax.plot(y_or, c="orange")
            ax.scatter(idx, h, c="red", zorder=3)
            # Add text labels with slight vertical offsets to avoid overlap
            for i, p in enumerate(idx):
                # Alternate the vertical offset for adjacent labels
                offset = 50 if i % 2 == 0 else -50
                ax.text(p, h[i] + offset, f"{self.ladder_bp[i]:.0f}", ha="center", va="bottom", fontsize=8)
            ax.set_title("Orange peaks with bp ladder")
            fig.savefig(os.path.join(self.debug_dir, "orange_peaks.png"))
            plt.close(fig)

    # ------------------------------------------------------------------
    # 2. map green-channel peak indices to bp by interpolation
    # ------------------------------------------------------------------
    def _assign_green_bp(self):
        if self.ladder_idx is None:
            raise RuntimeError("Run _align_orange_ladder() first.")

        y_gr = self._preprocess(self.green)
        idx_g, h_g = self._detect_peaks(y_gr)

        # --- piece-wise linear interpolation inside ladder range ---
        bp_g = np.interp(idx_g, self.ladder_idx, self.ladder_bp)

        # --- linear extrapolation outside the ladder range ----------
        if len(self.ladder_idx) >= 2:          # need two points for slope
            # left side
            left_mask = idx_g < self.ladder_idx[0]
            if left_mask.any():
                slope_left = (
                    (self.ladder_bp[1] - self.ladder_bp[0])
                    / (self.ladder_idx[1] - self.ladder_idx[0])
                )
                bp_g[left_mask] = (
                    self.ladder_bp[0]
                    + slope_left * (idx_g[left_mask] - self.ladder_idx[0])
                )

            # right side
            right_mask = idx_g > self.ladder_idx[-1]
            if right_mask.any():
                slope_right = (
                    (self.ladder_bp[-1] - self.ladder_bp[-2])
                    / (self.ladder_idx[-1] - self.ladder_idx[-2])
                )
                bp_g[right_mask] = (
                    self.ladder_bp[-1]
                    + slope_right * (idx_g[right_mask] - self.ladder_idx[-1])
                )

        # store result as DataFrame
        self.green_peaks = pd.DataFrame(
            dict(idx=idx_g, height=h_g, bp=bp_g)
        )

        # optional debug
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(18, 4))
            ax.plot(y_gr, c="green")
            ax.scatter(idx_g, h_g, c="red", zorder=3)
            for i, p in enumerate(idx_g):
                ax.text(p, h_g[i], f"{bp_g[i]:.0f}", ha="center", va="bottom")
            ax.set_title("Green peaks with interpolated bp")
            fig.savefig(os.path.join(self.debug_dir, "green_peaks.png"))
            plt.close(fig)
