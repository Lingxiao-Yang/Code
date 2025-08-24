
import re
import os
import sys
import traceback
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

# NEW: melt tiebreaker (all three regions)
from melt_classifier import MeltCurveResolver, CANON_REGIONS

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
DATA_ROOT    = Path("./Amplicon Length/Test Data/")
AMPLICON_XLS = "./Amplicon Length/Final Amplicon Profile.xlsx"
DEBUG_ROOT   = Path("./debug/")

# path to simulated melt dataset root (containing xlsx + region folders)
MELT_DS_ROOT = Path("Amplicon Length/Melt Dataset/")

REGION_MAP   = {"A": "16s23s", "B": "23s5s", "C": "ThrTyr"}
REGION_KEY   = {"16s23s": "(16s-23s)", "23s5s": "(23s-5s)", "ThrTyr": "(Thr-Tyr)"}

_SAMPLE_PATTERN = re.compile(r"(\d+)([ABC])", re.IGNORECASE)


def list_subfolders(root: Path) -> List[Path]:
    return [p for p in root.iterdir() if p.is_dir()]


def safe(s):
    return s.replace('\xa0', ' ')  # convert NBSP to normal space


def find_fsa_in(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.fsa"))


def group_by_sample(fsa_paths: List[Path]) -> Dict[str, List[Path]]:
    batches = defaultdict(list)
    for p in fsa_paths:
        m = _SAMPLE_PATTERN.search(p.stem)
        key = m.group(1) if m else p.stem
        batches[key].append(p)
    return batches


def read_identities(folder: Path) -> Dict[int, str]:
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


def extract_peaks(fsa_path: Path) -> Union[Tuple[str, pd.DataFrame], None]:
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


def _load_melt_derivative_sheet(xlsx: Path) -> Union[pd.DataFrame, None]:
    try:
        return pd.read_excel(xlsx, sheet_name="Derivative")
    except Exception:
        return None


def _lookup_exp_derivative(df_d: pd.DataFrame, sample_key: str, canon_region: str) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    """
    Given the 'Derivative' sheet dataframe, return (temps, values) for a sample id
    and canonical region name like '16s-23s'.
    """
    if df_d is None or "Temperature" not in df_d.columns:
        return None, None
    suffix = f"({canon_region})"
    cols = [c for c in df_d.columns if str(c).strip().startswith(str(sample_key)) and str(c).strip().endswith(suffix)]
    if not cols:
        return None, None
    col = cols[0]
    temps = np.asarray(df_d["Temperature"].values, dtype=float)
    vals = np.asarray(df_d[col].values, dtype=float)
    return temps, vals


def process_subfolder(
    subfolder: Path,
    ds,  # AmpliconDataset
    close_pct: float = 0.1,
    top_k: int = 10
) -> None:
    try:
        # new: prepare melt resolver & derivative sheet if present
        xlsx_files = list(subfolder.glob("*.xlsx"))
        df_derivative = _load_melt_derivative_sheet(xlsx_files[0]) if xlsx_files else None
        melt_resolver = MeltCurveResolver(MELT_DS_ROOT)

        # 1) Pre-scan qPCR with 80% of NTC max threshold
        dumped: Dict[Tuple[str, str], bool] = {}
        if xlsx_files:
            wb = xlsx_files[0]
            try:
                df_q = pd.read_excel(wb, sheet_name="qPCR")
                cycles = df_q["Cycle"]
            except Exception:
                df_q = None

            if df_q is not None:
                for region_tag, suffix in REGION_KEY.items():
                    col_ntc = f"NTC {suffix}"
                    if col_ntc not in df_q.columns:
                        continue

                    ntc_curve = df_q[col_ntc].values
                    thr = 0.8 * np.max(ntc_curve)
                    idxs_ntc = np.where(ntc_curve > thr)[0]
                    Ct_NTC = cycles.iloc[idxs_ntc[0]] if idxs_ntc.size else np.inf
                    print(f"  [NTC] {suffix}: max={np.max(ntc_curve):.1f}, thr={thr:.1f}, Ct_NTC={Ct_NTC}")

                    for col in df_q.columns:
                        if not col.endswith(suffix) or col.upper().startswith("NTC"):
                            continue
                        samp_curve = df_q[col].values
                        idxs_s = np.where(samp_curve > thr)[0]
                        Ct_s = cycles.iloc[idxs_s[0]] if idxs_s.size else np.inf
                        key = (col.split()[0], suffix)
                        if Ct_s >= Ct_NTC:
                            dumped[key] = True
                            print(f"  [DUMPED qPCR] Sample {key[0]} {suffix}: Ct_sample={Ct_s} >= Ct_NTC={Ct_NTC}")

        # 2) Plot qPCR & Derivative with bold controls (same look as process_all.py)
        group_suffixes = list(REGION_KEY.values())
        for xlsx in xlsx_files:
            out_dir = DEBUG_ROOT / subfolder.name
            out_dir.mkdir(parents=True, exist_ok=True)

            # qPCR
            try:
                df = pd.read_excel(xlsx, sheet_name="qPCR")
            except ValueError:
                print(f"Sheet 'qPCR' not found in {xlsx.name}")
                continue

            if "Cycle" in df.columns:
                cycles = df["Cycle"]
                fig, axes = plt.subplots(3, 1, figsize=(12, 18), dpi=150, sharex=True)
                for ax, suffix in zip(axes, group_suffixes):
                    cols = [c for c in df.columns if c.endswith(suffix)]
                    if not cols:
                        ax.set_visible(False)
                        continue
                    for col in cols:
                        is_ctrl = col.upper().startswith(("NTC", "GBLOCK"))
                        ax.plot(
                            cycles, df[col],
                            marker='o', linestyle='-',
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
                plt.savefig(fname, dpi=300)
                plt.close(fig)
                print(f"Saved qPCR plot: {fname}")

            # Derivative
            try:
                df_d = pd.read_excel(xlsx, sheet_name="Derivative")
            except ValueError:
                print(f"Sheet 'Derivative' not found in {xlsx.name}")
                continue

            if "Temperature" in df_d.columns:
                temps = df_d["Temperature"]
                fig, axes = plt.subplots(3, 1, figsize=(12, 18), dpi=150, sharex=True)
                for ax, suffix in zip(axes, group_suffixes):
                    cols = [c for c in df_d.columns if c.endswith(suffix)]
                    if not cols:
                        ax.set_visible(False)
                        continue
                    for col in cols:
                        is_ctrl = col.upper().startswith(("NTC", "GBLOCK"))
                        ax.plot(
                            temps, df_d[col],
                            marker='o', linestyle='-',
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
                plt.savefig(fname, dpi=300)
                plt.close(fig)
                print(f"Saved Derivative plot: {fname}")

        # 3) FSA-based peak extraction & classification
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

        for sample_key, paths in sorted(batches.items(), key=lambda x: int(x[0])):
            print(f"\n--- {subfolder.name} / Sample {sample_key} ({len(paths)} files) ---")
            results: Dict[str, pd.DataFrame] = {}

            for p in paths:
                m = _SAMPLE_PATTERN.search(p.stem)
                letter = m.group(2).upper()
                region_tag = REGION_MAP[letter]
                suffix = REGION_KEY[region_tag]

                if dumped.get((sample_key, suffix), False):
                    print(f"  [SKIPPED peaks] Sample {sample_key} {suffix} was dumped by qPCR control")
                    continue

                res = extract_peaks(p)
                if res:
                    region, df_peaks = res
                    results[region] = df_peaks

            if not results:
                print("  No valid peaks detected.")
                continue

            print("  Detected peaks:")
            for region, peaks in results.items():
                print(f"    {region:9s}: {peaks}")

            # Ground-truth lookup
            idx = int(sample_key)
            if idx in id_map:
                org = id_map[idx]
                print(f"  Ground truth for sample {idx}: {org}")
                for region in AmpliconClassifier.REGIONS:
                    try:
                        profile = ds.get_profile(org, region)
                        print(f"    {region:9s}: {profile}")
                    except Exception as e:
                        print(f"    ! Error loading {region}: {e}")

            # Classification pyramid
            try:
                clf = AmpliconClassifier(ds)
                pyramid = clf.iterative_rank(results, close_pct=close_pct, top_k=top_k)
                print("  Classification pyramid:")
                for level, (region, candidates) in enumerate(pyramid):
                    indent = '    ' + '  ' * level
                    print(f"{indent}{region}:")
                    for name, score in candidates:
                        ref_peaks = clf._to_bp_list(ds.get_profile(name, region))
                        name_print = safe(name)
                        print(f"{indent}  {name_print:25s} -log L = {score:7.2f} | ref: {ref_peaks}")
                print("-" * 60)

                # ── NEW: multi‑region melt-curve tiebreaker if ambiguous
                final_region, final_level = pyramid[-1]
                top_score = final_level[0][1]
                thresh = top_score * (1.0 + close_pct)
                close_cands = [n for n, s in final_level if s <= thresh]

                if len(close_cands) > 1 and df_derivative is not None:
                    exp_curves = {}
                    for canon in CANON_REGIONS:
                        t, y = _lookup_exp_derivative(df_derivative, str(sample_key), canon)
                        if t is not None and y is not None:
                            exp_curves[canon] = (t, y)

                    if exp_curves:
                        out_dir = DEBUG_ROOT / subfolder.name / f"melt_{sample_key}_ALL"
                        title = f"Melt comparison S{sample_key} (ALL REGIONS)"
                        best, per_region_metrics, aggregate = melt_resolver.score_multi_region(
                            exp_region_curves=exp_curves,
                            candidate_names=close_cands,
                            out_root=out_dir,
                            title_prefix=title
                        )
                        print(f"  [MELT] Multi-region tiebreaker among {len(close_cands)} candidates:")
                        for region, metrics in per_region_metrics.items():
                            print(f"    Region {region}:")
                            for n in close_cands:
                                m = metrics.get(n)
                                if m is None:
                                    print(f"      {n:25s}: (no simulated curve)")
                                else:
                                    print(f"      {n:25s}: r={m.pearson_r: .4f}, cosine={m.cosine: .4f}, RMSE={m.rmse: .4f}")
                        print("    Aggregate Pearson sum across regions:")
                        for n, v in aggregate.items():
                            print(f"      {n:25s}: sum_r={v: .4f}")
                        if best:
                            print(f"  [MELT] Winner by aggregated correlation → {best}")
                        else:
                            print("  [MELT] Could not resolve tie (no usable metrics).")
                    else:
                        print("  [MELT] No experimental derivative curves found in any region.")

            except Exception as e:
                print(f"!! Error during classification for sample {sample_key}: {e}")
                traceback.print_exc()

            print("-" * 60)

    except Exception as e:
        print(f"!! Unhandled error in process_subfolder: {e}")
        traceback.print_exc()


def main():
    DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
    ds = AmpliconDataset(AMPLICON_XLS)
    subfolders = list_subfolders(DATA_ROOT)

    for sub in tqdm(subfolders, desc="Processing folders", unit="folder"):
        rel = sub.relative_to(DATA_ROOT)
        folder_dbg = DEBUG_ROOT / rel
        folder_dbg.mkdir(parents=True, exist_ok=True)

        log_path = folder_dbg / f"output.txt"
        with open(log_path, "w") as log_fh:
            orig_stdout = sys.stdout
            sys.stdout = log_fh
            try:
                process_subfolder(sub, ds)
            except Exception as e:
                print(f"!! Unhandled error in folder '{sub.name}': {e}")
                traceback.print_exc()
            finally:
                sys.stdout = orig_stdout

    print("All batches processed. See debug subfolders for logs.")


if __name__ == "__main__":
    main()
