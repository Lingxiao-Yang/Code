import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
from math import log, pi, sqrt
from scipy.special import logsumexp
from amplicon_dataset import AmpliconDataset

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
    """Return canonical region name or None if unrecognised."""
    return _REGION_CANON.get(region.lower().replace("_", "").replace(" ", ""))

class AmpliconClassifier:
    """Hierarchical nearest-neighbour classifier for amplicon profiles."""

    REGIONS = ("16s-23s", "23s-5s", "Thr-Tyr")

    def __init__(
        self,
        dataset: "AmpliconDataset",
        tol: int = 5,
        miss_penalty: float = 120.0,
        extra_penalty: float = 60.0,
    ):
        self.ds = dataset
        self.tol = tol
        self.miss_penalty = miss_penalty
        self.extra_penalty = extra_penalty

    # --- collapse, distance helpers unchanged ---
    def _collapse_close(self, peaks: np.ndarray, tol: float) -> np.ndarray:
        if peaks.size == 0:
            return peaks
        peaks = np.sort(peaks)
        groups = [[peaks[0]]]
        for p in peaks[1:]:
            if p - groups[-1][-1] <= tol:
                groups[-1].append(p)
            else:
                groups.append([p])
        return np.asarray([float(np.mean(g)) for g in groups], dtype=np.float32)

    def _region_distance(
        self,
        test: List[int],
        ref:  List[int],
        sigma: float = 10.0,
        noise_floor: float = 1e-3,
    ) -> float:
        test = np.asarray(test, dtype=np.float32)
        ref  = np.asarray(ref,  dtype=np.float32)
        test = self._collapse_close(test, sigma)
        test = np.unique(test)
        ref  = self._collapse_close(ref,  sigma)
        ref  = np.unique(ref)
        if test.size == 0 and ref.size == 0:
            return 0.0
        if ref.size == 0:
            return self.extra_penalty * test.size
        if test.size == 0:
            return self.miss_penalty * ref.size
        inv_2s2  = 1.0 / (2 * sigma * sigma)
        log_norm = -log(sigma * sqrt(2 * np.pi))
        logpdf = log_norm - (test[:, None] - ref[None, :])**2 * inv_2s2
        log_p  = logsumexp(logpdf, axis=1) - log(ref.size)
        log_p  = np.logaddexp(log_p, log(noise_floor))
        nll    = -log_p.mean()
        min_d_ref = np.min(np.abs(ref[:, None] - test[None, :]), axis=1)
        nll      += self.miss_penalty * (min_d_ref > sigma).sum() / (ref.size + 1e-6)
        min_d_test = np.min(np.abs(test[:, None] - ref[None, :]), axis=1)
        nll       += self.extra_penalty * (min_d_test > sigma).sum() / (test.size + 1e-6)
        return float(nll)

    @staticmethod
    def _to_bp_list(peaks: Any) -> List[int]:
        if isinstance(peaks, (list, tuple, np.ndarray, pd.Series)):
            arr = peaks
        elif isinstance(peaks, pd.DataFrame):
            if "bp" not in peaks.columns:
                raise ValueError("DataFrame must contain a 'bp' column")
            arr = peaks["bp"].values
        else:
            raise TypeError("Peaks must be list-like or a DataFrame with column 'bp'")
        return sorted(int(x) for x in arr if not pd.isna(x))

    def iterative_rank(
        self,
        test_profile: Dict[str, Any],
        close_pct: float = 0.2,
        top_k: int = 10,
    ) -> List[Tuple[str, List[Tuple[str, float]]]]:
        """
        Hierarchical classification:
        1) rank on first region;
        2) filter close candidates (<= top_score*(1+close_pct));
        3) if >1 candidate, repeat for next region.

        Returns pyramid: [(region, [(name, score), ...]), ...]
        """
        # prepare test peaks per region
        query: Dict[str, List[int]] = {}
        for tag, peaks in test_profile.items():
            canon = _canon(tag)
            if canon:
                query[canon] = self._to_bp_list(peaks)
        if not query:
            raise ValueError("No recognised regions in test_profile")

        pyramid: List[Tuple[str, List[Tuple[str, float]]]] = []
        candidates = list(self.ds.bacteria_names)

        for region in self.REGIONS:
            if region not in query:
                continue
            # compute distances for current region only
            dists: List[Tuple[str, float]] = []
            for name in candidates:
                ref = self._to_bp_list(self.ds.get_profile(name, region))
                score = self._region_distance(query[region], ref)
                dists.append((name, score))
            dists.sort(key=lambda x: x[1])
            # keep only top_k for debugging clarity
            level = dists[:top_k]
            pyramid.append((region, level))
            # filter close candidates
            top_score = level[0][1]
            thresh = top_score * (1 + close_pct)
            close_cands = [n for n, s in level if s <= thresh]
            if len(close_cands) <= 1:
                break
            candidates = close_cands

        return pyramid
