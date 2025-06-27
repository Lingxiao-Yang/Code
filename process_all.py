import re
import os
import sys
import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_parser import read_fsa
from wave_analysis import CapillaryPeakCalibrator
from amplicon_dataset import AmpliconDataset
from amplicon_classifier import AmpliconClassifier

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
DATA_ROOT    = Path("./Amplicon Length/Test Data/")
AMPLICON_XLS = "./Amplicon Length/Final Amplicon Profile.xlsx"
DEBUG_ROOT   = Path("./debug/")

REGION_MAP   = {"A": "16s23s", "B": "23s5s", "C": "ThrTyr"}
REGION_KEY   = {"16s23s": "16s-23s", "23s5s": "23s-5s", "ThrTyr": "Thr-Tyr"}

_SAMPLE_PATTERN = re.compile(r"(\d+)([ABC])", re.IGNORECASE)


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


def process_subfolder(
    subfolder: Path,
    ds,  # AmpliconDataset, not used directly here
    close_pct: float = 0.1,
    top_k: int = 10
) -> None:
    """
    For each .xlsx in subfolder:
      - Reads 'qPCR' and 'Derivative' sheets
      - Groups columns by suffix: (16s-23s), (23s-5s), (Thr-Tyr)
      - Plots each group in its own subplot (3 rows)
      - Bolds lines for NTC and gBlock entries
    """
    group_suffixes = ["(16s-23s)", "(23s-5s)", "(Thr-Tyr)"]

    xlsx_files = list(subfolder.glob("*.xlsx"))
    if not xlsx_files:
        return

    out_dir = DEBUG_ROOT / subfolder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for xlsx in xlsx_files:
        # --- qPCR sheet ---
        try:
            df = pd.read_excel(xlsx, sheet_name="qPCR")
        except ValueError:
            print(f"Sheet 'qPCR' not found in {xlsx.name}")
            continue

        if "Cycle" not in df.columns:
            print(f"'Cycle' column missing in {xlsx.name}['qPCR']")
        else:
            cycles = df["Cycle"]
            fig, axes = plt.subplots(
                nrows=3, ncols=1, figsize=(12, 18), dpi=150, sharex=True
            )
            for ax, suffix in zip(axes, group_suffixes):
                cols = [c for c in df.columns if c.endswith(suffix)]
                if not cols:
                    ax.set_visible(False)
                    continue
                for col in cols:
                    is_ctrl = col.upper().startswith(("NTC", "GBLOCK"))
                    ax.plot(
                        cycles,
                        df[col],
                        marker='o',
                        linestyle='-',
                        linewidth=3.0 if is_ctrl else 1.5,
                        alpha=1.0 if is_ctrl else 0.7,
                        label=col
                    )
                ax.set_title(f"qPCR {suffix}")
                ax.set_ylabel("Fluorescence")
                ax.legend(loc="best", fontsize="small")
                ax.grid(True)
            axes[-1].set_xlabel("Cycle")

            fname = out_dir / f"{xlsx.stem}_qPCR_grouped.png"
            plt.tight_layout()
            plt.savefig(fname, bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"Saved grouped qPCR plot: {fname}")

        # --- Derivative sheet ---
        try:
            df_d = pd.read_excel(xlsx, sheet_name="Derivative")
        except ValueError:
            print(f"Sheet 'Derivative' not found in {xlsx.name}")
            continue

        if "Temperature" not in df_d.columns:
            print(f"'Temperature' column missing in {xlsx.name}['Derivative']")
        else:
            temps = df_d["Temperature"]
            fig, axes = plt.subplots(
                nrows=3, ncols=1, figsize=(12, 18), dpi=150, sharex=True
            )
            for ax, suffix in zip(axes, group_suffixes):
                cols = [c for c in df_d.columns if c.endswith(suffix)]
                if not cols:
                    ax.set_visible(False)
                    continue
                for col in cols:
                    is_ctrl = col.upper().startswith(("NTC", "GBLOCK"))
                    ax.plot(
                        temps,
                        df_d[col],
                        marker='o',
                        linestyle='-',
                        linewidth=3.0 if is_ctrl else 1.5,
                        alpha=1.0 if is_ctrl else 0.7,
                        label=col
                    )
                ax.set_title(f"Derivative {suffix}")
                ax.set_ylabel("d(Fluorescence)/dT")
                ax.legend(loc="best", fontsize="small")
                ax.grid(True)
            axes[-1].set_xlabel("Temperature")

            fname = out_dir / f"{xlsx.stem}_Derivative_grouped.png"
            plt.tight_layout()
            plt.savefig(fname, bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"Saved grouped Derivative plot: {fname}")



    all_fsa = find_fsa_in(subfolder)
    if not all_fsa:
        return

    valid = [p for p in all_fsa if _SAMPLE_PATTERN.search(p.stem)]
    for p in all_fsa:
        if p not in valid:
            print(f"Ignoring file {p.name}: no sample pattern found")
    if not valid:
        return

    batches = group_by_sample(valid)
    id_map = read_identities(subfolder)

    for sample_key, paths in sorted(batches.items(), key=lambda x: x[0]):
        print(f"\n--- {subfolder.name} / Sample {sample_key} ({len(paths)} files) ---")
        # extract peaks per region
        results: Dict[str, List[int]] = {}
        for p in paths:
            res = extract_peaks(p)
            if res:
                region, df = res
                results[region] = df

        if not results:
            print("  No valid peaks detected.")
            continue

        # show detected peaks
        print("  Detected peaks:")
        for region, peaks in results.items():
            print(f"    {region:9s}: {peaks}")

        # ground truth if available
        try:
            idx = int(sample_key)
        except ValueError:
            idx = None
        if idx in id_map:
            org = id_map[idx]
            print(f"  Ground truth for sample {idx}: {org}")
            for region in AmpliconClassifier.REGIONS:
                try:
                    profile = ds.get_profile(org, region)
                    print(f"    {region:9s}: {profile}")
                except Exception as e:
                    print(f"    ! Error loading {region}: {e}")

        # iterative classification pyramid
        clf = AmpliconClassifier(ds)
        pyramid = clf.iterative_rank(results, close_pct=close_pct, top_k=top_k)

        print("  Classification pyramid:")
        for level, (region, candidates) in enumerate(pyramid):
            indent = '    ' + '  ' * level
            print(f"{indent}{region}:")
            for name, score in candidates:
                ref_peaks = clf._to_bp_list(ds.get_profile(name, region))
                print(f"{indent}  {name:25s} -log L = {score:7.2f} | test: {results.get(region)} | ref: {ref_peaks}")
        print("-" * 60)


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
