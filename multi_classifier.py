# AmpliconClassifier.py
# Complete rewrite with weighted iterative likelihood, equal-weight mixtures, k_max forwarding,
# optional adaptive k selection, and an "auto v2" decision rule with an improvement gate.

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import nnls  # used by K-sparse and single-scalar fits

from amplicon_dataset import AmpliconDataset


# -------------------- Region name normalisation --------------------

_REGION_CANON = {
    "16s-23s":  "16s-23s",
    "16s23s":   "16s-23s",
    "23s-5s":   "23s-5s",
    "23s5s":    "23s-5s",
    "thr-tyr":  "Thr-Tyr",
    "thrtyr":   "Thr-Tyr",
    "thryptyr": "Thr-Tyr",
}

def _canon(region: str) -> Optional[str]:
    """Return canonical region name or None if unrecognised."""
    return _REGION_CANON.get(region.lower().replace("_", "").replace(" ", ""))


# -------------------- Classifier --------------------

class AmpliconClassifier:
    """
    Modes:
      - iterative_rank(...)            : hierarchical single-organism ranking (distance model).
      - iterative_likelihood(...)      : converts weighted distances → softmax probability.
      - classify_k_sparse(...)         : Non-negative OMP with ≤k components.
      - classify_sparse(...)           : Nonnegative LASSO mixture (pruned).
      - classify_equal_weight_mixture(): choose support via solver, then force equal weights = 1/k (k ≤ k_max).
      - classify_auto_v2(...)          : Weighted iterative likelihood; pivot to CS(equal) only if confidence low AND CS improves residual.

    Notes
    -----
    - Region keys are canonicalised via _canon.
    - CS modes render peaks onto region-specific grids; region_weights applied consistently.
    """

    REGIONS = ("16s-23s", "23s-5s", "Thr-Tyr")

    def __init__(
        self,
        dataset: "AmpliconDataset",
        tol: float = 5.0,
        miss_penalty: float = 120.0,
        extra_penalty: float = 60.0,
        sigma: float = 10.0,            # Gaussian width (bp) when rendering densities
        bin_size: float = 2.0,          # bp grid resolution
        # Default region weights emphasize 16s-23s
        region_weights: Optional[Dict[str, float]] = None,
    ):
        self.ds = dataset
        self.tol = float(tol)
        self.miss_penalty = float(miss_penalty)
        self.extra_penalty = float(extra_penalty)

        self.sigma = float(sigma)
        self.bin_size = float(bin_size)
        self.region_weights = region_weights or {
            "16s-23s": 1.0,
            "23s-5s": 0.3,
            "Thr-Tyr": 0.3,
        }

        # Precompute per-region ranges for density grids
        self._region_ranges = self._compute_region_ranges()

    # ---------- Utilities ----------

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

    def _collapse_close(self, peaks: np.ndarray, tol: float) -> np.ndarray:
        if peaks.size == 0:
            return peaks
        peaks = np.sort(peaks.astype(np.float32))
        groups = [[peaks[0]]]
        for p in peaks[1:]:
            if p - groups[-1][-1] <= tol:
                groups[-1].append(p)
            else:
                groups.append([p])
        return np.asarray([float(np.mean(g)) for g in groups], dtype=np.float32)

    def _build_query(self, test_profile: Dict[str, Any]) -> Dict[str, List[int]]:
        """Canonicalise region keys and coerce peaks to sorted int lists."""
        query: Dict[str, List[int]] = {}
        for tag, peaks in test_profile.items():
            c = _canon(tag)
            if c:
                query[c] = self._to_bp_list(peaks)
        if not query:
            raise ValueError("No recognised regions in test_profile")
        return query

    # ---------- Region distance (single-organism scoring) ----------

    def _region_distance(
        self,
        test: List[int],
        ref:  List[int],
        sigma: Optional[float] = None,
        noise_floor: float = 1e-3,
    ) -> float:
        """
        Negative log-likelihood distance between 'test' and 'ref' peaks with Gaussian
        proximity plus miss/extra penalties. Lower is better.
        """
        sigma = float(self.tol) if sigma is None else float(sigma)

        test = np.asarray(test, dtype=np.float32)
        ref  = np.asarray(ref,  dtype=np.float32)
        test = self._collapse_close(test, sigma)
        test = np.unique(test)
        ref  = self._collapse_close(ref,  sigma)
        ref  = np.unique(ref)

        if test.size == 0 and ref.size == 0:
            return 0.0
        if ref.size == 0:
            return self.extra_penalty * float(test.size)
        if test.size == 0:
            return self.miss_penalty * float(ref.size)

        inv_2s2  = 1.0 / (2 * sigma * sigma)
        log_norm = -np.log(sigma * np.sqrt(2 * np.pi))

        # Log mixture-likelihood of each test peak under ref peaks
        logpdf = log_norm - (test[:, None] - ref[None, :])**2 * inv_2s2
        log_p  = logsumexp(logpdf, axis=1) - np.log(max(ref.size, 1))
        log_p  = np.logaddexp(log_p, np.log(noise_floor))  # noise floor
        nll    = -float(log_p.mean())

        # Miss / extra penalties (farther than ~sigma)
        if test.size:
            min_d_ref = np.min(np.abs(ref[:, None] - test[None, :]), axis=1) if ref.size else np.full(ref.size, np.inf)
            nll += self.miss_penalty * float((min_d_ref > sigma).sum()) / (ref.size + 1e-6)
        if ref.size:
            min_d_test = np.min(np.abs(test[:, None] - ref[None, :]), axis=1) if ref.size else np.full(test.size, np.inf)
            nll += self.extra_penalty * float((min_d_test > sigma).sum()) / (test.size + 1e-6)

        return float(nll)

    # ---------- 1) Hierarchical single-organism rank ----------

    def iterative_rank(
        self,
        test_profile: Dict[str, Any],
        close_pct: float = 0.2,
        top_k: int = 10,
    ) -> List[Tuple[str, List[Tuple[str, float]]]]:
        """
        Hierarchical ranking:
          1) Rank by first available region
          2) Keep candidates within (1+close_pct)*best
          3) Repeat on next region until one remains or regions exhausted

        Returns: [(region, [(name, score), ...]), ...]
        """
        query = self._build_query(test_profile)

        pyramid: List[Tuple[str, List[Tuple[str, float]]]] = []
        candidates = list(self.ds.bacteria_names)

        for region in self.REGIONS:
            if region not in query:
                continue

            dists: List[Tuple[str, float]] = []
            for name in candidates:
                ref = self._to_bp_list(self.ds.get_profile(name, region))
                score = self._region_distance(query[region], ref)
                dists.append((name, score))
            dists.sort(key=lambda x: x[1])

            level = dists[:top_k]
            pyramid.append((region, level))

            top_score = level[0][1]
            thresh = top_score * (1.0 + close_pct)
            close_cands = [n for n, s in level if s <= thresh]
            if len(close_cands) <= 1:
                break
            candidates = close_cands

        return pyramid


    def _pyramid_progress(
        self,
        test_profile: Dict[str, Any],
        close_pct: float = 0.2,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Run the hierarchical pyramid exactly like iterative_rank, but record:
        - ordered used regions (first..second..),
        - per-stage top_k distance tables,
        - the stage index where the pyramid resolves (<=1 close candidate).

        Returns:
        {
            "regions": [r1, r2, ...],
            "levels":  [
                {"region": r1, "scores": [(name, dist), ...], "close": [names...]},
                {"region": r2, "scores": [...],              "close": [names...]},
                ...
            ],
            "resolved_stage": 1-based index where resolved, or None
        }
        """
        query = self._build_query(test_profile)
        regions = [r for r in self.REGIONS if r in query]
        levels = []
        candidates = list(self.ds.bacteria_names)
        resolved_stage = None

        for idx, region in enumerate(regions, start=1):
            # score current candidate set on this region
            dists = []
            for name in candidates:
                ref = self._to_bp_list(self.ds.get_profile(name, region))
                d = self._region_distance(query[region], ref)
                dists.append((name, d))
            dists.sort(key=lambda t: t[1])
            scores = dists[: max(1, int(top_k))]

            best = scores[0][1]
            thresh = best * (1.0 + float(close_pct))
            close_names = [n for (n, v) in scores if v <= thresh]
            levels.append({"region": region, "scores": scores, "close": close_names})

            # update candidate list for next stage
            candidates = close_names if close_names else [scores[0][0]]

            # resolved if ≤1 candidate remains
            if resolved_stage is None and len(candidates) <= 1:
                resolved_stage = idx
                break

        return {"regions": [lv["region"] for lv in levels],
                "levels": levels,
                "resolved_stage": resolved_stage}


    # ---------- Iterative likelihood (WEIGHTED by region_weights) ----------

    def iterative_likelihood(
        self,
        test_profile: Dict[str, Any],
        beta: float = 0.5,
        top_k_per_level: int = 50,                 # used on multi-region path
        region_weights: Optional[Dict[str, float]] = None,
        # close-set restriction for the multi-region path
        close_set: str = "pct",
        close_param: float = 0.25,
        # per-sample attenuation (multi-region path)
        attenuate_noisy_regions: bool = True,
        tau: float = 12.0,
        # NEW: use only the first m stages if pyramid resolves at stage m (m=1 or 2)
        honor_pyramid_resolution: bool = True,
        stage_close_pct: float = 0.2,
        stage_top_k: int = 20,
    ) -> Dict[str, Any]:
        """
        If the pyramid resolves at stage m:
        - m=1  -> compute likelihood from stage-1 only.
        - m=2  -> compute likelihood from stages 1+2 only.
        Else: fall back to multi-region likelihood (all regions).
        """
        # --- Run pyramid to see where it resolves ---
        py = self._pyramid_progress(
            test_profile, close_pct=stage_close_pct, top_k=stage_top_k
        )
        resolved_m = py["resolved_stage"]  # None or 1/2/...

        # Helper to softmax a list[(name, dist)]
        def _softmax_from_scores(scores: List[Tuple[str, float]]) -> Tuple[str, float, List[Tuple[str,float]]]:
            names = [n for (n, _) in scores]
            vals  = np.array([v for (_, v) in scores], dtype=float)
            logits = -beta * (vals - vals.min())
            probs  = np.exp(logits - logsumexp(logits))
            best_idx = int(np.argmin(vals))
            return names[best_idx], float(probs[best_idx]), [(n, float(v)) for (n, v) in scores]

        # If resolved at stage 1 → use ONLY stage-1 scores
        if honor_pyramid_resolution and resolved_m == 1:
            s1 = py["levels"][0]["scores"]
            best_name, prob, scores = _softmax_from_scores(s1)
            return {
                "name": best_name,
                "prob": prob,
                "scores": scores,
                "used_regions": [py["levels"][0]["region"]],
                "mode": "pyramid_stage_1_only",
            }

        # If resolved at stage 2 → combine ONLY stage-1 and stage-2 (weighted)
        if honor_pyramid_resolution and resolved_m == 2 and len(py["levels"]) >= 2:
            rw = region_weights or self.region_weights
            # Build weighted totals from first two stages only
            totals: Dict[str, float] = {}
            used = [py["levels"][0]["region"], py["levels"][1]["region"]]
            for lv in py["levels"][:2]:
                region = lv["region"]
                w = float(rw.get(region, 1.0))
                for name, d in lv["scores"]:
                    totals[name] = totals.get(name, 0.0) + w * d
            scores = sorted(totals.items(), key=lambda t: t[1])
            best_name, prob, scores = _softmax_from_scores(scores)
            return {
                "name": best_name,
                "prob": prob,
                "scores": scores,
                "used_regions": used,
                "mode": "pyramid_stages_1_2_only",
            }

        # ---------- Multi-region path (unchanged logic you already use) ----------
        query = self._build_query(test_profile)
        regions = [r for r in self.REGIONS if r in query]
        if not regions:
            raise ValueError("No recognised regions available for likelihood.")
        rw_user = region_weights or self.region_weights

        # 1) per-region distances and best
        per_region_dlists: Dict[str, List[Tuple[str, float]]] = {}
        per_region_best: Dict[str, float] = {}
        for region in regions:
            dlist: List[Tuple[str, float]] = []
            for name in self.ds.bacteria_names:
                ref = self._to_bp_list(self.ds.get_profile(name, region))
                d = self._region_distance(query[region], ref)
                dlist.append((name, d))
            dlist.sort(key=lambda t: t[1])
            per_region_dlists[region] = dlist[: max(1, int(top_k_per_level))]
            per_region_best[region] = dlist[0][1] if dlist else float("inf")

        # 2) attenuation (optional)
        w_eff: Dict[str, float] = {}
        for region in regions:
            w_base = float(rw_user.get(region, 1.0))
            if attenuate_noisy_regions:
                d_best = per_region_best[region]
                atten = 1.0 / (1.0 + (max(d_best, 0.0) / max(tau, 1e-6)))
                w_eff[region] = w_base * atten
            else:
                w_eff[region] = w_base

        # 3) accumulate weighted totals
        totals: Dict[str, float] = {}
        for region in regions:
            w_r = w_eff[region]
            for name, d in per_region_dlists[region]:
                totals[name] = totals.get(name, 0.0) + w_r * d

        if not totals:
            return {"name": None, "prob": 0.0, "scores": [], "used_regions": regions, "mode": "multi_region"}

        # 4) global sort and close-set restriction
        scores_all = sorted(totals.items(), key=lambda t: t[1])
        best_val = scores_all[0][1]
        if close_set == "pct":
            keep = [(n, v) for (n, v) in scores_all if v <= best_val * (1.0 + float(close_param))]
        elif close_set == "abs":
            keep = [(n, v) for (n, v) in scores_all if v <= best_val + float(close_param)]
        elif close_set == "topk":
            keep = scores_all[: int(max(2, close_param))]
        else:
            keep = scores_all

        # 5) margin softmax over the kept set
        names = [n for (n, _) in keep]
        vals  = np.array([v for (_, v) in keep], dtype=float)
        logits = -beta * (vals - vals.min())
        probs  = np.exp(logits - logsumexp(logits))
        best_idx = int(np.argmin(vals))

        return {
            "name": names[best_idx],
            "prob": float(probs[best_idx]),
            "scores": scores_all,
            "used_regions": regions,
            "mode": "multi_region",
            "w_effective": {r: float(w_eff[r]) for r in regions},
        }




    # ---------- 2) CS: region ranges, grids, rendering ----------

    def _compute_region_ranges(self) -> Dict[str, Tuple[float, float]]:
        r: Dict[str, Tuple[float, float]] = {}
        for region in self.REGIONS:
            lows, highs = [], []
            for name in getattr(self.ds, "bacteria_names", []):
                peaks = self.ds.get_profile(name, region)
                if len(peaks):
                    lows.append(min(peaks))
                    highs.append(max(peaks))
            if lows:
                lo, hi = float(min(lows)), float(max(highs))
            else:
                lo, hi = 0.0, 1.0
            pad = 3.0 * self.sigma
            r[region] = (max(0.0, lo - pad), hi + pad)
        return r

    def _grid_for(self, region: str) -> np.ndarray:
        lo, hi = self._region_ranges[region]
        n = max(2, int(np.ceil((hi - lo) / self.bin_size)))
        return np.linspace(lo, hi, n, dtype=np.float32)

    def _render_density(self, peaks: List[int], grid: np.ndarray, sigma: float) -> np.ndarray:
        arr = np.asarray(peaks, dtype=np.float32)
        if arr.size == 0:
            return np.zeros_like(grid, dtype=np.float32)
        arr = self._collapse_close(arr, tol=self.tol)
        inv_2s2 = 1.0 / (2.0 * sigma * sigma)
        diffs = grid[None, :] - arr[:, None]
        dens = np.exp(-diffs * diffs * inv_2s2).sum(axis=0)
        s = np.sum(dens) * (grid[1] - grid[0])  # L1-normalize
        return (dens / s) if s > 0 else dens

    def _build_dictionary_and_target(
        self,
        test_profile: Dict[str, Any],
        candidates: Optional[List[str]] = None,
        sigma: Optional[float] = None,
        region_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Build region-stacked dictionary A (M x N) and target y (M,).
        Region weights are applied as sqrt(w) factors to both A and y blocks.
        """
        sigma = float(self.sigma if sigma is None else sigma)
        rw = region_weights or self.region_weights

        query = self._build_query(test_profile)
        regions = [r for r in self.REGIONS if r in query]
        names = candidates if candidates is not None else list(self.ds.bacteria_names)

        A_list, y_list = [], []
        for region in regions:
            w = float(rw.get(region, 1.0))
            grid = self._grid_for(region)
            y_r = self._render_density(query[region], grid, sigma=sigma)
            cols = []
            for name in names:
                ref = self.ds.get_profile(name, region)
                dens = self._render_density(ref, grid, sigma=sigma)
                cols.append(dens)
            A_r = np.stack(cols, axis=1) if cols else np.zeros((len(grid), 0), dtype=np.float32)
            A_list.append(np.sqrt(w) * A_r)
            y_list.append(np.sqrt(w) * y_r)

        A = np.vstack(A_list).astype(np.float64) if A_list else np.zeros((0,0), dtype=np.float64)
        y = np.concatenate(y_list).astype(np.float64) if y_list else np.zeros((0,), dtype=np.float64)

        # Drop all-zero columns
        if A.size:
            col_nonzero = (np.linalg.norm(A, axis=0) > 0)
            if not np.all(col_nonzero):
                A = A[:, col_nonzero]
                names = [n for n, keep in zip(names, col_nonzero) if keep]

        return A, y, names, regions

    # ---------- 2a) K-sparse NNOMP (≤k_max components) ----------

    def classify_k_sparse(
        self,
        test_profile: Dict[str, Any],
        k_max: int = 3,
        tol: float = 1e-3,              # stop if relative residual <= tol
        min_corr: float = 1e-6,         # stop if no positive correlation
        prefer_single: bool = True,
        single_threshold: float = 0.85,
        prune_tol: float = 1e-3,
        candidates: Optional[List[str]] = None,
        sigma: Optional[float] = None,
        region_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Greedy positive-correlation selection with NNLS refits on the chosen support.
        Returns normalized weights (sum≈1 after pruning) and relative residual.
        """
        A, y, names, used_regions = self._build_dictionary_and_target(
            test_profile, candidates=candidates, sigma=sigma, region_weights=region_weights
        )

        if A.size == 0 or np.allclose(A, 0):
            return {"used_regions": used_regions, "weights": [], "alpha": None,
                    "residual_rel": 1.0, "decision": {"mode": "mixture","label": None},
                    "num_active": 0, "n_dict": int(A.shape[1] if A.ndim==2 else 0)}

        y_norm = np.linalg.norm(y) + 1e-12
        r = y.copy()
        support: List[int] = []

        for _ in range(max(0, int(k_max))):
            corr = A.T @ r  # shape (N,)
            if support:
                corr[np.array(support, dtype=int)] = -np.inf
            j = int(np.argmax(corr))
            if not np.isfinite(corr[j]) or corr[j] <= min_corr:
                break
            support.append(j)

            As = A[:, support]
            xs, _ = nnls(As, y)
            r = y - As @ xs

            if (np.linalg.norm(r) / y_norm) <= tol:
                break

        # Final NNLS on support; build full coef vector
        x = np.zeros(A.shape[1], dtype=float)
        if support:
            As = A[:, support]
            xs, _ = nnls(As, y)
            x[support] = xs

        # Normalize for readability
        total = x.sum()
        if total > 0:
            x /= total

        # Prune tiny weights and sort
        weights = [(names[i], float(w)) for i, w in enumerate(x) if w >= prune_tol]
        weights.sort(key=lambda t: t[1], reverse=True)
        resid_rel = float(np.linalg.norm(A @ x - y) / (y_norm))

        decision = {"mode": "mixture", "label": None}
        if prefer_single and weights and weights[0][1] >= float(single_threshold):
            decision = {"mode": "single", "label": weights[0][0]}

        return {
            "used_regions": used_regions,
            "weights": weights,         # [(name, weight)], sorted; sum≈1 after pruning
            "alpha": None,              # for API parity with LASSO
            "residual_rel": resid_rel,
            "decision": decision,
            "num_active": len(weights),
            "n_dict": int(A.shape[1]),
        }

    # ---------- 2b) Compressive-sensing mixture (nonnegative LASSO) ----------

    def classify_sparse(
        self,
        test_profile: Dict[str, Any],
        alpha: Optional[float] = None,
        alphas: Optional[np.ndarray] = None,
        use_cv: bool = True,
        max_iter: int = 5000,
        prefer_single: bool = True,
        single_threshold: float = 0.85,
        prune_tol: float = 1e-3,
        candidates: Optional[List[str]] = None,
        sigma: Optional[float] = None,
        region_weights: Optional[Dict[str, float]] = None,
        random_state: int = 0,
    ) -> Dict[str, Any]:
        """
        LASSO / LASSO-CV with positive=True on column-normalised dictionary.
        """
        try:
            from sklearn.linear_model import Lasso, LassoCV
        except Exception as exc:
            raise ImportError("scikit-learn is required for classify_sparse (pip install scikit-learn)") from exc

        A, y, names, used_regions = self._build_dictionary_and_target(
            test_profile, candidates=candidates, sigma=sigma, region_weights=region_weights
        )

        if A.size == 0 or np.allclose(A, 0):
            return {"used_regions": used_regions, "weights": [], "alpha": None,
                    "residual_rel": 1.0, "decision": {"mode": "mixture","label": None},
                    "num_active": 0, "n_dict": int(A.shape[1] if A.ndim==2 else 0)}

        # Normalize columns (stabilizes alpha); normalize y to unit L2
        col_norms = np.linalg.norm(A, axis=0) + 1e-12
        A_n = A / col_norms
        y_n = y / (np.linalg.norm(y) + 1e-12)

        if use_cv:
            if alphas is None:
                alphas = np.logspace(-4, 0, 25)
            model = LassoCV(
                alphas=alphas, fit_intercept=False, positive=True,
                max_iter=max_iter, random_state=random_state
            )
        else:
            if alpha is None:
                raise ValueError("alpha must be provided when use_cv=False")
            model = Lasso(alpha=float(alpha), fit_intercept=False, positive=True, max_iter=max_iter)

        model.fit(A_n, y_n)
        coef_n = model.coef_.astype(np.float64)  # on normalized columns
        x = coef_n / col_norms  # rescale back

        # Normalize for readability
        total = x.sum()
        if total > 0:
            x = x / total

        # Prune tiny weights and sort
        weights = [(n, float(w)) for n, w in zip(names, x) if w >= prune_tol]
        weights.sort(key=lambda t: t[1], reverse=True)
        num_active = len(weights)

        # Relative residual on original scale
        resid_rel = float(np.linalg.norm(A @ x - y) / (np.linalg.norm(y) + 1e-12))

        decision = {"mode": "mixture", "label": None}
        if prefer_single and weights and weights[0][1] >= float(single_threshold):
            decision = {"mode": "single", "label": weights[0][0]}

        return {
            "used_regions": used_regions,
            "weights": weights,          # [(name, weight)], sorted; sum≈1 after pruning
            "alpha": (getattr(model, "alpha_", None) if use_cv else alpha),
            "residual_rel": resid_rel,   # lower is better
            "decision": decision,
            "num_active": int(num_active),
            "n_dict": int(A.shape[1]),
        }

    # ---------- Equal-weight composition (limit to ≤5, all weights=1/k) ----------

    @staticmethod
    def _equalize_support(weights: List[Tuple[str, float]], k_max: int = 5) -> List[Tuple[str, float]]:
        """
        Keep top-k_max by weight, then force equal weights 1/k (k = #kept).
        """
        if not weights:
            return []
        top = sorted(weights, key=lambda t: t[1], reverse=True)[: max(1, int(k_max))]
        k = len(top)
        equal_w = 1.0 / k
        return [(n, equal_w) for n, _ in top]

    def _pick_k_via_sweep(
        self,
        test_profile: Dict[str, Any],
        k_max: int,
        prune_tol: float,
        candidates: Optional[List[str]] = None,
        sigma: Optional[float] = None,
        region_weights: Optional[Dict[str, float]] = None,
        tol: float = 1e-3,
        min_corr: float = 1e-6,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Try k = 1..k_max on KSparse; choose the best by residual + small complexity penalty.
        """
        best: Optional[Tuple[float, int, Dict[str, Any]]] = None
        for k in range(1, max(1, int(k_max)) + 1):
            trial = self.classify_k_sparse(
                test_profile,
                k_max=k,
                tol=tol,
                min_corr=min_corr,
                prefer_single=False,
                prune_tol=prune_tol,
                candidates=candidates,
                sigma=sigma,
                region_weights=region_weights,
            )
            penalty = 0.01 * k  # tiny complexity penalty to discourage always-max k
            score = float(trial["residual_rel"]) + penalty
            if (best is None) or (score < best[0]):
                best = (score, k, trial)
        assert best is not None
        return best[1], best[2]  # (k*, trial)

    def classify_equal_weight_mixture(
        self,
        test_profile: Dict[str, Any],
        solver: str = "ksparse",     # "ksparse" or "lasso"
        k_max: int = 5,
        prune_tol: float = 1e-3,
        choose_k: bool = False,      # let KSparse adapt k in 1..k_max
        # Forwarded knobs
        candidates: Optional[List[str]] = None,
        sigma: Optional[float] = None,
        region_weights: Optional[Dict[str, float]] = None,
        tol: float = 1e-3,
        min_corr: float = 1e-6,
        use_cv: bool = True,
        alphas: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
        random_state: int = 0,
    ) -> Dict[str, Any]:
        """
        Choose support via solver, then OVERRIDE amplitudes to equal weights (1/k).
        Caps k ≤ k_max. Also returns a binary presence vector.
        """
        solver = solver.lower()
        if solver == "ksparse":
            if choose_k:
                _, mix = self._pick_k_via_sweep(
                    test_profile,
                    k_max=k_max,
                    prune_tol=prune_tol,
                    candidates=candidates,
                    sigma=sigma,
                    region_weights=region_weights or self.region_weights,
                    tol=tol,
                    min_corr=min_corr,
                )
            else:
                mix = self.classify_k_sparse(
                    test_profile,
                    k_max=k_max,                     # ← forward k_max (bug fix)
                    tol=tol,
                    min_corr=min_corr,
                    prefer_single=False,
                    prune_tol=prune_tol,
                    candidates=candidates,
                    sigma=sigma,
                    region_weights=region_weights or self.region_weights,
                )
        elif solver == "lasso":
            mix = self.classify_sparse(
                test_profile,
                alpha=alpha,
                alphas=alphas,
                use_cv=use_cv,
                max_iter=5000,
                prefer_single=False,
                single_threshold=0.85,
                prune_tol=prune_tol,
                candidates=candidates,
                sigma=sigma,
                region_weights=region_weights or self.region_weights,
                random_state=random_state,
            )
        else:
            raise ValueError("solver must be one of {'ksparse','lasso'}")

        # Equalize weights (1/k)
        eq_weights = self._equalize_support(mix["weights"], k_max=k_max)
        # Binary presence (all 1s for selected)
        binary = [(n, 1.0) for n, _ in eq_weights]

        # Simple residual-based confidence proxy (higher = better): 1 - residual_rel
        conf = float(np.clip(1.0 - float(mix.get("residual_rel", 1.0)), 0.0, 1.0))

        return {
            "used_regions": mix.get("used_regions", []),
            "weights": eq_weights,            # equal weights: each = 1/k
            "weights_binary": binary,         # all ones (presence)
            "residual_rel": mix.get("residual_rel", 1.0),
            "confidence": conf,
            "n_dict": mix.get("n_dict", 0),
            "num_active": len(eq_weights),
            "base_solver": solver,
            "raw_solver_weights": mix.get("weights", []),  # for debugging
        }

    # ---------- Single best-by-distance helpers ----------

    def _best_single_by_distance(
        self,
        test_profile: Dict[str, Any],
        sigma: Optional[float] = None,
        region_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, float]:
        """Return (name, total_weighted_distance) using weighted sum of region distances."""
        query = self._build_query(test_profile)
        rw = region_weights or self.region_weights
        best_name, best_total = None, float("inf")
        for name in self.ds.bacteria_names:
            total = 0.0
            for region, test_peaks in query.items():
                w = float(rw.get(region, 1.0))
                ref = self._to_bp_list(self.ds.get_profile(name, region))
                total += w * self._region_distance(test_peaks, ref, sigma=sigma)
            if total < best_total:
                best_total = total
                best_name = name
        return best_name, float(best_total)

    def _single_residual(
        self,
        test_profile: Dict[str, Any],
        name: str,
        region_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Relative residual for best nonnegative scalar fit using only 'name'
        on the same stacked target space used in CS: ||a*x* - y|| / ||y||.
        """
        A, y, names, _ = self._build_dictionary_and_target(
            test_profile, candidates=[name], region_weights=region_weights
        )
        if A.shape[1] == 0 or np.allclose(A, 0):
            return 1.0
        a = A[:, 0]
        denom = float(np.dot(a, a)) + 1e-12
        x_star = max(float(np.dot(a, y)) / denom, 0.0)
        return float(np.linalg.norm(a * x_star - y) / (np.linalg.norm(y) + 1e-12))

    # ---------- AUTO v2: weighted likelihood + improvement gate ----------

    def classify_auto_v2(
        self,
        test_profile: Dict[str, Any],
        # Iterative likelihood
        beta: float = 0.2,
        likelihood_thresh: float = 0.90,
        # CS equal-weight settings
        solver: str = "ksparse",
        k_max: int = 5,
        prune_tol: float = 1e-3,
        choose_k: bool = False,
        # Improvement gate
        improvement_min: float = 0.15,    # require ≥15% relative improvement to call mixture
        # Region weights
        region_weights: Optional[Dict[str, float]] = None,
        # Diagnostics
        return_both_paths: bool = True,
        # Forwarded knobs
        candidates: Optional[List[str]] = None,
        sigma: Optional[float] = None,
        tol: float = 1e-3,
        min_corr: float = 1e-6,
        use_cv: bool = True,
        alphas: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
        random_state: int = 0,
    ) -> Dict[str, Any]:
        """
        Flow:
          1) Weighted iterative likelihood with region_weights.
          2) If prob ≥ likelihood_thresh -> SINGLE (return single with residual).
          3) Else compute single residual and run CS(equal weights). Call MIXTURE only if
             residual improves by ≥ improvement_min; otherwise stick with SINGLE.
        """
        like = self.iterative_likelihood(
            test_profile, beta=beta, region_weights=region_weights
        )
        used_regions = like["used_regions"]
        prob = float(like["prob"])
        best_name = like["name"]

        # If confident single
        if best_name is not None and prob >= float(likelihood_thresh):
            resid_single = self._single_residual(test_profile, best_name, region_weights=region_weights)
            return {
                "decision": {"mode": "single", "label": best_name},
                "iterative": {
                    "name": best_name,
                    "prob": prob,
                    "residual_rel": float(resid_single),
                    "scores": like["scores"],
                },
                "mixture": None,
                "used_regions": used_regions,
                "criteria": {
                    "likelihood_thresh": likelihood_thresh,
                    "beta": beta,
                    "k_max": k_max,
                    "solver": solver,
                    "prune_tol": prune_tol,
                    "choose_k": choose_k,
                    "improvement_min": improvement_min,
                    "region_weights": region_weights or self.region_weights,
                },
            }

        # Compute single residual even when prob is low
        resid_single = self._single_residual(
            test_profile, best_name, region_weights=region_weights
        ) if best_name else 1.0

        # Run CS(equal weights) with same region_weights
        mix = self.classify_equal_weight_mixture(
            test_profile,
            solver=solver,
            k_max=k_max,
            prune_tol=prune_tol,
            choose_k=choose_k,
            candidates=candidates,
            sigma=sigma,
            region_weights=region_weights or self.region_weights,
            tol=tol,
            min_corr=min_corr,
            use_cv=use_cv,
            alphas=alphas,
            alpha=alpha,
            random_state=random_state,
        )
        resid_mix = float(mix["residual_rel"])
        improvement = (resid_single - resid_mix) / max(resid_single, 1e-12)

        if improvement >= float(improvement_min):
            decision = {"mode": "mixture", "label": None}
            out = {
                "decision": decision,
                "iterative": {
                    "name": best_name,
                    "prob": prob,
                    "scores": like["scores"],
                    "residual_rel": resid_single,
                },
                "mixture": mix,
                "used_regions": list(set(used_regions) | set(mix.get("used_regions", []))),
                "criteria": {
                    "likelihood_thresh": likelihood_thresh,
                    "beta": beta,
                    "k_max": k_max,
                    "solver": solver,
                    "prune_tol": prune_tol,
                    "choose_k": choose_k,
                    "improvement_min": improvement_min,
                    "region_weights": region_weights or self.region_weights,
                },
            }
            if return_both_paths:
                return out
            return {"decision": decision, "mixture": mix, "used_regions": out["used_regions"], "criteria": out["criteria"]}

        # Mixture didn't improve enough → stay single (low-confidence)
        decision = {"mode": "single", "label": best_name}
        return {
            "decision": decision,
            "iterative": {
                "name": best_name,
                "prob": prob,
                "scores": like["scores"],
                "residual_rel": resid_single,
            },
            "mixture": {
                "used_regions": mix.get("used_regions", []),
                "weights": mix.get("weights", []),
                "residual_rel": resid_mix,
                "confidence": mix.get("confidence", None),
                "num_active": mix.get("num_active", 0),
            },
            "used_regions": used_regions,
            "criteria": {
                "likelihood_thresh": likelihood_thresh,
                "beta": beta,
                "k_max": k_max,
                "solver": solver,
                "prune_tol": prune_tol,
                "choose_k": choose_k,
                "improvement_min": improvement_min,
                "region_weights": region_weights or self.region_weights,
            },
        }

    # ---------- (Original) AUTO kept for compatibility ----------

    def classify_auto(
        self,
        test_profile: Dict[str, Any],
        improvement_thresh: float = 0.12,
        min_components: int = 2,
        min_component_weight: float = 0.07,
        single_threshold: float = 0.85,
        prune_tol: float = 1e-3,
        solver: str = "ksparse",
        k_max: int = 3,
        **sparse_kwargs,
    ) -> Dict[str, Any]:
        """
        Legacy auto mode (kept for backward compatibility).
        """
        best_single, single_dist = self._best_single_by_distance(test_profile)
        resid_single = self._single_residual(test_profile, best_single)

        if solver.lower() == "ksparse":
            mix = self.classify_k_sparse(
                test_profile,
                k_max=k_max,
                prefer_single=False,
                prune_tol=prune_tol,
                **{k: v for k, v in sparse_kwargs.items()
                   if k in ("candidates","sigma","region_weights","tol")}
            )
        elif solver.lower() == "lasso":
            mix = self.classify_sparse(
                test_profile,
                prefer_single=False,
                prune_tol=prune_tol,
                **sparse_kwargs,
            )
        else:
            raise ValueError("solver must be one of {'ksparse','lasso'}")

        num_active = mix["num_active"]
        topw = mix["weights"][0][1] if mix["weights"] else 0.0
        if (num_active <= 1) or (topw >= single_threshold):
            decision = {"mode": "single", "label": best_single}
            final_mode = "single"
        else:
            resid_mix = float(mix["residual_rel"])
            improvement = (resid_single - resid_mix) / max(resid_single, 1e-12)
            k_substantial = sum(w >= min_component_weight for _, w in mix["weights"])
            if (improvement >= improvement_thresh) and (k_substantial >= min_components):
                decision = {"mode": "mixture", "label": None}
                final_mode = "mixture"
            else:
                decision = {"mode": "single", "label": best_single}
                final_mode = "single"

        return {
            "decision": decision,
            "auto_mode": final_mode,
            "single": {
                "name": best_single,
                "distance_total": single_dist,
                "residual_rel": resid_single,
            },
            "sparse": mix,
            "criteria": {
                "improvement_thresh": improvement_thresh,
                "min_components": min_components,
                "min_component_weight": min_component_weight,
                "single_threshold": single_threshold,
                "prune_tol": prune_tol,
                "solver": solver,
                "k_max": k_max,
            },
        }
