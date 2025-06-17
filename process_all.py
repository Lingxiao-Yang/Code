import re
import os
import sys
import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_parser import read_fsa
from wave_analysis import CapillaryPeakCalibrator
from amplicon_dataset import AmpliconDataset
from amplicon_classifier import AmpliconClassifier

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
DATA_ROOT    = Path("./Amplicon Length/Test Data")
AMPLICON_XLS = "./Amplicon Length/Final Amplicon Profile.xlsx"
DEBUG_ROOT   = Path("./debug/")

REGION_MAP   = {"A": "16s23s", "B": "23s5s", "C": "ThrTyr"}
REGION_KEY   = {"16s23s": "16s-23s", "23s5s": "23s-5s", "ThrTyr": "Thr-Tyr"}

BACTERIUM_KEY={
    "E. coli": "Escherichia coli",
    "Staph aureus": "Staphylococcus aureus",
}

_SAMPLE_PATTERN = re.compile(r"(\d+)([ABCabc]|_(16s-23s|23s-5s|thr-tyr))", re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────
def list_subfolders(root: Path) -> list[Path]:
    return [p for p in root.iterdir() if p.is_dir()]

def find_fsa_in(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.fsa"))

def group_by_sample(fsa_paths: list[Path]) -> dict[str, list[Path]]:
    batches = defaultdict(list)
    for p in fsa_paths:
        m = _SAMPLE_PATTERN.search(p.stem)
        key = m.group(1) if m else p.stem
        batches[key].append(p)
    return batches

def read_identities(folder: Path) -> dict[int, str]:
    for fn in ("Sample identities.txt", "identities.txt"):
        f = folder / fn
        if f.is_file():
            out = {}
            for line in f.read_text().splitlines():
                if "=" not in line:
                    continue
                left, right = [s.strip() for s in line.split("=", 1)]
                m = re.search(r"(\d+)$", left)
                if m:
                    out[int(m.group(1))] = right
            return out
    return {}

def bp_list(peaks) -> list[int]:
    if peaks is None:
        return []
    if isinstance(peaks, pd.DataFrame):
        arr = peaks["bp"].values
    elif isinstance(peaks, pd.Series):
        arr = peaks.values
    else:
        arr = peaks
    return sorted(int(x) for x in arr if pd.notna(x))


# ──────────────────────────────────────────────────────────────────────────────
# Peak extraction with robust error handling
# ──────────────────────────────────────────────────────────────────────────────
def extract_peaks(fsa_path: Path) -> tuple[str, pd.DataFrame] | None:
    """
    Calibrate one .fsa file → (region_tag, peaks_df).
    Catches instantiation/run errors, logs them, and returns None on failure.
    """
    m = _SAMPLE_PATTERN.search(fsa_path.stem)
    letter = m.group(2).upper() if m else None
    region_tag = REGION_MAP.get(letter, "unknown")

    rel = fsa_path.parent.relative_to(DATA_ROOT)
    dbg = DEBUG_ROOT / rel / fsa_path.stem
    dbg.mkdir(parents=True, exist_ok=True)

    try:
        green, orange = read_fsa(str(fsa_path))
    except Exception as e:
        print(f"  ! Failed to read FSA {fsa_path.name}: {e}")
        return None

    try:
        calibrator = CapillaryPeakCalibrator(
            orange=orange,
            green=green,
            debug_dir=str(dbg),
            region=region_tag,
        )
        _, peaks = calibrator.run()
        return region_tag, peaks

    except Exception as e:
        print(f"  ! Calibration error for {fsa_path.name}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Batch processing
# ──────────────────────────────────────────────────────────────────────────────
def process_subfolder(subfolder: Path, ds: AmpliconDataset) -> None:
    all_fsa = find_fsa_in(subfolder)
    if not all_fsa:
        return

    valid, unrelated = [], []
    for p in all_fsa:
        if _SAMPLE_PATTERN.search(p.stem):
            valid.append(p)
        else:
            print(f"Ignoring file {p.name}: no sample pattern found")

    if not valid:
        return

    batches = group_by_sample(valid)
    id_map = read_identities(subfolder)

    for sample_key, paths in sorted(batches.items(), key=lambda x: x[0]):
        try:
            print(f"\n--- Folder '{subfolder.name}', Sample '{sample_key}' ({len(paths)} files) ---")
            results = {}

            for p in paths:
                res = extract_peaks(p)
                if res is None:
                    continue
                region, peaks = res
                results[region] = peaks

            print("  Detected green peaks:")
            for region, df in results.items():
                print(f"    {region:9s}: {bp_list(df)}")

            try:
                idx = int(sample_key)
            except ValueError:
                idx = None
            if idx and idx in id_map:
                org = id_map[idx]
                if org in BACTERIUM_KEY:
                    org = BACTERIUM_KEY[org]
                print(f"  Ground truth sample {idx}: {org}")
                for short, nice in REGION_KEY.items():
                    try:
                        profile = ds.get_profile(org, region=nice)
                        print(f"    {nice:9s}: {profile}")
                    except Exception as e:
                        print(f"    ! Profile error for region {nice}: {e}")

            try:
                clf = AmpliconClassifier(ds)
                candidates = clf.rank(results, top_k=10)
                print("  Top matches:")
                for bacterium, score in candidates:
                    print(f"    {bacterium:30s}  -log L = {score:.2f}")
                    # print reference profile for each region
                    for reg in ("16s-23s", "23s-5s", "Thr-Tyr"):
                        ref_bp = ds.get_profile(bacterium, region=reg)
                        print(f"        {reg:9s}: {ref_bp}")
                print("-" * 72)
            except Exception as e:
                print(f"  ! Classification error: {e}")

            print("-" * 50)

        except Exception as e:
            print(f"!! Unhandled error in sample '{sample_key}': {e}")
            continue


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
    ds = AmpliconDataset(AMPLICON_XLS)
    subfolders = list_subfolders(DATA_ROOT)

    for sub in tqdm(subfolders, desc="Processing folders", unit="folder"):
        # prepare a log file for this subfolder
        rel = sub.relative_to(DATA_ROOT)
        folder_dbg = DEBUG_ROOT / rel
        folder_dbg.mkdir(parents=True, exist_ok=True)

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = folder_dbg / f"output{now}.txt"
        log_fh = open(log_path, "w")

        # redirect prints for this subfolder
        orig_stdout = sys.stdout
        sys.stdout = log_fh

        try:
            process_subfolder(sub, ds)
        except Exception as e:
            print(f"!! Unhandled error in folder '{sub.name}': {e}")
        finally:
            sys.stdout = orig_stdout
            log_fh.close()

    print("All batches processed. See debug subfolders for logs.")

if __name__ == "__main__":
    main()
