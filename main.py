import re
import os

import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Dict

from data_parser import read_fsa
from wave_analysis import CapillaryPeakCalibrator
from amplicon_dataset import AmpliconDataset
from amplicon_classifier import AmpliconClassifier

DATA_DIR = "./Amplicon Length/Test Data/3.17.25 Promega (Redo) UCx #1 03052025/"
AMPLICON_DIR = "./Amplicon Length/Final Amplicon Profile.xlsx"
DEBUG_DIR = "./debug/"
STEM_RE = re.compile(r"^(.*?-\d+)([ABC])_(.*)$")

REGION_MAP = {"A": "16s23s", "B": "23s5s", "C": "ThrTyr"}

def stem_parts(stem: str):
    """Return (batch_key, replicate_letter, tail) or (None, None, None)."""
    m = STEM_RE.match(stem)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, None

# ───────────────────────────── main ────────────────────────────────
def _bp_list(peaks: Any) -> List[int]:
    """
    Accepts
        • list / tuple / numpy array
        • pd.Series
        • pd.DataFrame (must contain a “bp” column)
    Returns
        Sorted list[int] with NaNs removed.
    """
    if peaks is None:
        return []
    if isinstance(peaks, pd.DataFrame):
        arr = peaks["bp"].values
    elif isinstance(peaks, pd.Series):
        arr = peaks.values
    else:                          # list, tuple, ndarray …
        arr = peaks
    return sorted(int(x) for x in arr if pd.notna(x))


# ----------------------------------------------------------------------
#  Main pipeline
# ----------------------------------------------------------------------
def main() -> None:
    amplicon_dataset = AmpliconDataset(AMPLICON_DIR)

    # -------- group all .fsa files into “batches” ----------------------
    batches: Dict[str, List[str]] = defaultdict(list)
    for fname in sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".fsa")):
        stem = Path(fname).stem
        batch_key, _, _ = stem_parts(stem)
        key = batch_key if batch_key else stem
        batches[key].append(fname)

    # ------------------------------------------------------------------
    for batch_key, batch_files in batches.items():
        batch_files.sort()
        if len(batch_files) != 3:
            print(f"Warning: batch '{batch_key}' has {len(batch_files)} file(s); expected 3.")

        print(f"\nProcessing batch {batch_key}: {', '.join(batch_files)}")

        # ---- collect green-channel peak tables from all three runs ----
        batch_results: Dict[str, pd.DataFrame] = {}
        for fsa_file in batch_files:
            stem = Path(fsa_file).stem
            dash_idx = stem.find('-')
            replicate_subfolder = stem[dash_idx + 1:] if dash_idx != -1 else stem

            batch_key2, repl_letter, _ = stem_parts(stem)
            if not batch_key2:
                batch_key2 = stem.split('-', 1)[0]
            region_tag = REGION_MAP.get(repl_letter, "unknown")

            debug_dir = os.path.join(DEBUG_DIR, batch_key2, replicate_subfolder)
            fsa_path  = os.path.join(DATA_DIR, fsa_file)

            green_channel, orange_channel = read_fsa(fsa_path)

            calibrator = CapillaryPeakCalibrator(
                orange    = orange_channel,
                green     = green_channel,
                debug_dir = debug_dir,
                region    = region_tag,
            )

            try:
                _, green_peaks = calibrator.run()          # → DataFrame(idx, height, bp)
                batch_results[region_tag] = green_peaks
            except RuntimeError as e:
                print(f"Calibration failed for {fsa_file}: {e}")

        # ---- show test peaks (for debugging) --------------------------
        print("  Detected green peaks (bp):")
        for region, peaks_df in batch_results.items():
            print(f"    {region:9s}: {_bp_list(peaks_df)}")

        # ---- rank candidate bacteria ---------------------------------
        clf = AmpliconClassifier(amplicon_dataset)
        candidates = clf.rank(batch_results, top_k=5)

        print("\n  Top matches:")
        for bacterium, score in candidates:
            print(f"    {bacterium:30s}  −log L = {score:.2f}")
            # print reference profile for each region
            for reg in ("16s-23s", "23s-5s", "Thr-Tyr"):
                ref_bp = amplicon_dataset.get_profile(bacterium, region=reg)
                print(f"        {reg:9s}: {ref_bp}")
        print("-" * 72)

        # ---- Find best parameters ----------------------
        '''best_tol = 0
        best_miss_penalty = 0
        best_extra_penalty = 0
        best_region_weights = {}
        best_score=100000000
        best_candidates = []
        tols=[5,8,10]
        miss_penalties=[100.0]#np.linspace(30.0, 100.0, 10)
        extra_penalties=[30.0]#np.linspace(30.0, 100.0, 10)
        region_weightss = [{"16s-23s": 1.0, "23s-5s": 1.0, "Thr-Tyr": 1.0},{"16s-23s": 1.0, "23s-5s": 0.7, "Thr-Tyr": 0.3},{ "16s-23s": 1.0, "23s-5s": 0.5, "Thr-Tyr": 0.3}]
        for tol in tols:
            for miss_penalty in miss_penalties:
                for extra_penalty in extra_penalties:
                    for region_weights in region_weightss:
                        clf = AmpliconClassifier(amplicon_dataset, miss_penalty=miss_penalty, extra_penalty=extra_penalty,region_weights=region_weights)
                        candidates = clf.rank(batch_results, top_k=1)
                        nll_score=0
                        for bacterium, score in candidates:
                            nll_score += score
                        nll_score /= len(candidates)
                        if nll_score < best_score:
                            best_score = nll_score
                            best_tol = tol
                            best_miss_penalty = miss_penalty
                            best_extra_penalty = extra_penalty
                            best_region_weights = region_weights
                            best_candidates = candidates[0][0]
        print(f'Best parameters for batch {batch_key}: tol={best_tol}, miss_penalty={best_miss_penalty}, extra_penalty={best_extra_penalty}, region_weights={best_region_weights}, mean_nll={best_score}')
        print(f'Best candidate(s) for batch {batch_key}: {best_candidates}')
        for reg in ("16s-23s", "23s-5s", "Thr-Tyr"):
            ref_bp = amplicon_dataset.get_profile(best_candidates, region=reg)
            print(f"        {reg:9s}: {ref_bp}")
        print('-' * 72)'''


    print("\nCalibration completed for all batches.\n")

if __name__ == "__main__":
    main()