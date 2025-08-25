import re
import os
import sys
import datetime
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
from multi_classifier import AmpliconClassifier


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
DATA_ROOT    = Path("./Amplicon Length/Polymicrobial")
AMPLICON_XLS = "./Amplicon Length/Final Amplicon Profile.xlsx"
DEBUG_ROOT   = Path("./debug/")

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


def _split_org_list(text: str) -> List[str]:
    """
    Split a right-hand-side like:
      'S stimulans (CoNS)'
      'Morganella morganii, Klebsiella pneumoniae, S caprae'
      'A and B, C'
    into a list of organism strings (preserving any parentheses/qualifiers).
    """
    # First split on commas, then split any leftover ' and ' within parts
    parts = []
    for chunk in re.split(r",", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.extend([c.strip() for c in re.split(r"\band\b", chunk) if c.strip()])
    # Normalize internal whitespace
    parts = [re.sub(r"\s+", " ", p) for p in parts]
    return parts


def read_identities(folder: Path) -> Dict[int, List[str]]:
    """
    Reads ground truth file allowing multiple organisms per sample:
      e.g. 'Sample 2 = Morganella morganii, Klebsiella pneumoniae, S caprae'
    Returns: {2: ['Morganella morganii', 'Klebsiella pneumoniae', 'S caprae'], ...}
    """
    for fn in ("Sample identities.txt", "identities.txt"):
        f = folder / fn
        if f.is_file():
            out: Dict[int, List[str]] = {}
            for raw in f.read_text().splitlines():
                if "=" not in raw:
                    continue
                left, right = [s.strip() for s in raw.split("=", 1)]
                m = re.search(r"(\d+)$", left)
                if not m:
                    continue
                idx = int(m.group(1))
                org_list = _split_org_list(right)
                out[idx] = org_list
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


def _set_metrics(pred: List[str], truth: List[str]) -> Tuple[float, float, float, List[str], List[str], List[str]]:
    """Simple multi-label set metrics."""
    pset, tset = set(pred), set(truth)
    tp = len(pset & tset)
    prec = tp / max(len(pset), 1)
    rec  = tp / max(len(tset), 1)
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1, sorted(pset & tset), sorted(pset - tset), sorted(tset - pset)


def process_subfolder(
    subfolder: Path,
    ds,  # AmpliconDataset
    close_pct: float = 0.1,
    top_k: int = 10
) -> None:
    try:
        # 1) Pre-scan qPCR with 80% of NTC max threshold
        dumped: Dict[Tuple[str, str], bool] = {}
        xlsx_files = list(subfolder.glob("*.xlsx"))
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

        # 2) Plot qPCR & Derivative with bold controls
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

        # per-folder summary
        summary_rows = []

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

            # Ground-truth lookup (multi-label)
            idx = int(sample_key)
            truth_list: List[str] = id_map.get(idx, [])
            if truth_list:
                print(f"  Ground truth for sample {idx}: {truth_list}")
                for region in AmpliconClassifier.REGIONS:
                    for org in truth_list:
                        try:
                            profile = ds.get_profile(org, region)
                            print(f"    {region:9s} [{org}]: {profile}")
                        except Exception as e:
                            print(f"    ! Error loading {region} for '{org}': {e}")

            # Classification pyramid
            try:
                def _safe(x):
                    try:
                        return safe(x)
                    except NameError:
                        return str(x)

                clf = AmpliconClassifier(ds, tol=5, sigma=10, bin_size=2, region_weights={"16s-23s": 1.0, "23s-5s": 0.3, "Thr-Tyr": 0.1})

                # --- 1) Single-organism pyramid (print for visibility) ---
                pyr_close_pct, pyr_top_k = 0.2, 10
                pyramid = clf.iterative_rank(results, close_pct=pyr_close_pct, top_k=pyr_top_k)
                print("  Classification pyramid (single-organism):")
                for level, (region, candidates) in enumerate(pyramid):
                    indent = '    ' + '  ' * level
                    print(f"{indent}{region}:")
                    for name, score in candidates:
                        ref_peaks = clf._to_bp_list(ds.get_profile(name, region))
                        print(f"{indent}  {_safe(name):25s} -log L = {score:7.2f} | ref: {ref_peaks}")
                print("-" * 60)

                # --- 2) Email-driven AUTO: iterative likelihood → CS(equal weights ≤5) ---
                #     NOTE: weights are forced to 1/k in mixture mode; also returns binary presence.
                auto = clf.classify_auto_v2(
                    results,
                    likelihood_thresh=0.60,
                    beta=0.2,
                    solver="ksparse",
                    k_max=3,
                    prune_tol=1e-3,
                    improvement_min=0.15,
                    region_weights={"16s-23s": 1.0, "23s-5s": 0.3, "Thr-Tyr": 0.3},
                )
                decision   = auto["decision"]
                used_regs  = auto["used_regions"]

                print(f"AUTO v2 decision: {decision}")
                if decision["mode"] == "single":
                    single_name = auto["decision"]["label"]
                    single_prob = auto["iterative"]["prob"]
                    single_res  = auto["iterative"]["residual_rel"]
                    print(f"  Single: {single_name} | likelihood={single_prob:.2%} | residual={single_res:.4f}")
                    pred_list = [single_name] if single_name else []
                    resid_mix = None
                else:
                    mix = auto["mixture"]
                    resid_mix = mix["residual_rel"]
                    conf_mix  = mix["confidence"]
                    eq_weights = mix["weights"]  # equal weights 1/k
                    print(f"  Mixture (equal weights) residual={resid_mix:.4f} | confidence≈{conf_mix:.2f}")
                    for name, w in eq_weights:
                        print(f"    {_safe(name):30s} {w*100:6.2f}%")
                    pred_list = [n for n, _ in eq_weights]

                    # Also show union of component peaks per region
                    print("  Union of component peaks per region:")
                    for region in used_regs:
                        union = []
                        for n, _ in eq_weights:
                            union.extend(clf._to_bp_list(ds.get_profile(n, region)))
                        union_arr = np.asarray(sorted(union), dtype=np.float32)
                        union_collapsed = list(clf._collapse_close(union_arr, clf.tol)) if len(union_arr) else []
                        print(f"    {region:7s}: {union_collapsed}")

                # --- 3) Multi-label metrics vs ground truth (if available) ---
                prec = rec = f1 = np.nan
                tp_names = fp_names = fn_names = []
                if truth_list:
                    prec, rec, f1, tp_names, fp_names, fn_names = _set_metrics(pred_list, truth_list)
                    print(f"  Metrics vs GT  -> P={prec:.2f} R={rec:.2f} F1={f1:.2f}")
                    if tp_names: print(f"    TP: {tp_names}")
                    if fp_names: print(f"    FP: {fp_names}")
                    if fn_names: print(f"    FN: {fn_names}")

                # collect summary row
                summary_rows.append({
                    "folder": subfolder.name,
                    "sample": int(sample_key),
                    "mode": decision["mode"],
                    "single_label": decision.get("label"),
                    "mixture_residual": resid_mix,
                    "iterative_prob": auto["iterative"]["prob"] if auto["iterative"] else None,
                    "pred_components": "; ".join(pred_list),
                    "gt_components": "; ".join(truth_list),
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "tp": "; ".join(tp_names),
                    "fp": "; ".join(fp_names),
                    "fn": "; ".join(fn_names),
                })

            except Exception as e:
                print(f"!! Error during classification for sample {sample_key}: {e}")
                traceback.print_exc()

            print("-" * 60)

        # write summary CSV for this subfolder
        if summary_rows:
            df_sum = pd.DataFrame(summary_rows).sort_values(["folder", "sample"])
            out_csv = DEBUG_ROOT / subfolder.name / "summary.csv"
            df_sum.to_csv(out_csv, index=False)
            print(f"Saved folder summary: {out_csv}")

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
