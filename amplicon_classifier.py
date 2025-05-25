_REGION_CANON = {
    "16s-23s":  "16s-23s",
    "16s23s":   "16s-23s",
    "23s-5s":   "23s-5s",
    "23s5s":    "23s-5s",
    "thr-tyr":  "Thr-Tyr",
    "thrtyr":   "Thr-Tyr",
    "thryptyr": "Thr-Tyr",
}

def _canon(region: str) -> str | None:
    """Return canonical region name or *None* if unrecognised."""
    return _REGION_CANON.get(region.lower().replace("_", "").replace(" ", ""))

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
from amplicon_dataset import AmpliconDataset
from math import log, pi, sqrt
from scipy.special import logsumexp


class AmpliconClassifier:
    """Weighted-region nearest-neighbour classifier for amplicon profiles.

    Parameters
    ----------
    dataset : AmpliconDataset
        Your reference database.
    tol : int, default 5
        ±bp tolerance for matching peaks.
    miss_penalty, extra_penalty : float
        Cost for missing / spurious peaks (per peak).
    region_weights : dict, optional
        How much each region should influence the overall score.
        Default = {'16s-23s': 1.0, '23s-5s': 0.5, 'Thr-Tyr': 0.2}.
        Use larger numbers to give a region more sway.
    """

    REGIONS = ("16s-23s", "23s-5s", "Thr-Tyr")

    def __init__(
        self,
        dataset: "AmpliconDataset",
        tol: int = 5,
        miss_penalty: float = 100.0,
        extra_penalty: float = 30.0,
        region_weights: Dict[str, float] | None = None,
    ):
        self.ds = dataset
        self.tol = tol
        self.miss_penalty = miss_penalty
        self.extra_penalty = extra_penalty

        default_w = {"16s-23s": 1.0, "23s-5s": 0.7, "Thr-Tyr": 0.3}
        self.w = default_w if region_weights is None else region_weights

    def _region_distance(
        self,
        test: List[int],
        ref:  List[int],
        sigma: float = 4.0,
        noise_floor: float = 1e-3,
    ) -> float:
        """
        Negative average log-likelihood **plus**
        • miss_penalty   for each expected band not seen,
        • extra_penalty  for each test band ≥ σ away from *all* reference bands.
        """
        test = np.asarray(test, dtype=np.float32)
        ref  = np.asarray(ref,  dtype=np.float32)

        # ---------------- edge cases ------------------------------------
        if test.size == 0 and ref.size == 0:
            return 0.0
        if ref.size == 0:
            return self.extra_penalty * test.size          # all peaks are “extra”
        if test.size == 0:
            return self.miss_penalty * ref.size            # all bands missing

        inv_2s2  = 1.0 / (2 * sigma * sigma)
        log_norm = -log(sigma * sqrt(2 * pi))

        # log pdf matrix  (m × n)
        logpdf = (
            log_norm
            - (test[:, None] - ref[None, :]) ** 2 * inv_2s2
        )

        # mixture log-prob for each test peak
        log_p = logsumexp(logpdf, axis=1) - log(ref.size)
        log_p = np.logaddexp(log_p, log(noise_floor))       # protect against −∞
        nll   = -log_p.mean()                               # average −log L

        # --------------- penalties --------------------------------------
        # (1) missing reference bands
        min_d_ref = np.min(np.abs(ref[:, None] - test[None, :]), axis=1)
        missed    = (min_d_ref > sigma).sum()
        nll += self.miss_penalty * missed / (ref.size + 1e-6)

        # (2) unexpected test peaks
        min_d_test = np.min(np.abs(test[:, None] - ref[None, :]), axis=1)
        extras     = (min_d_test > sigma).sum()
        nll += self.extra_penalty * extras / (test.size + 1e-6)

        return float(nll)

    # ───────────────────────── public API ────────────────────────────────
    def rank(
        self,
        test_profile: Dict[str, Any],     # accepts DataFrame / list / Series
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Accept **batch_results-style** input, i.e.::

            {'16s23s':  DataFrame(idx, height, bp),
             '23s5s':   DataFrame(...),
             'ThrTyr':  DataFrame(...)}

        Keys may use any of the aliases in _REGION_CANON.  Values may be
        • list[int] / np.ndarray
        • pd.Series of bp
        • pd.DataFrame with a 'bp' column.
        """
        # ─── 1. normalise keys & convert each region to a bp-list ───────
        query: Dict[str, List[int]] = {}
        for reg_tag, peaks in test_profile.items():
            canon = _canon(reg_tag)
            if canon is None:
                # silently ignore unknown tags; alternatively raise an error
                continue
            query[canon] = self._to_bp_list(peaks)

        if not query:
            raise ValueError("No recognised regions found in test_profile")

        # ─── 2. score every bacterium ───────────────────────────────────
        scores = []
        for name in self.ds.bacteria_names:
            weighted_sum = total_w = 0.0
            for region, test_peaks in query.items():
                w = self.w.get(region, 0.0)
                if w == 0.0:
                    continue
                ref_peaks = self._to_bp_list(self.ds.get_profile(name, region))
                weighted_sum += w * self._region_distance(test_peaks, ref_peaks)
                total_w += w
            if total_w:               # should always be true
                scores.append((name, weighted_sum / total_w))

        scores.sort(key=lambda x: x[1])
        return scores[:top_k]

    # ───────────────────────── helper (unchanged) ───────────────────────
    @staticmethod
    def _to_bp_list(peaks: Any) -> List[int]:
        if isinstance(peaks, (list, tuple, np.ndarray, pd.Series)):
            arr = peaks
        elif isinstance(peaks, pd.DataFrame):
            if "bp" not in peaks.columns:
                raise ValueError("DataFrame must contain a 'bp' column")
            arr = peaks["bp"].values
        else:
            raise TypeError(
                "Peaks must be list-like or a DataFrame with column 'bp'"
            )
        return sorted(int(x) for x in arr if not pd.isna(x))