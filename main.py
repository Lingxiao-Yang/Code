import re
import os
import pandas as pd
from collections import defaultdict
from pathlib import Path

from data_parser import read_fsa
from wave_analysis import CapillaryPeakCalibrator
from amplicon_dataset import AmpliconDataset

DATA_DIR = "./Amplicon Length/Test Data/3.17.25 Promega (Redo) UCx #1 03052025/"
AMPLICON_DIR = "./Amplicon Length/Final Amplicon Profile.xlsx"
DEBUG_DIR = "./debug/"
STEM_RE = re.compile(r"^(.*?-\d+)([ABC])_(.*)$")

def stem_parts(stem: str):
    """Return (batch_key, replicate_letter, tail) or (None, None, None)."""
    m = STEM_RE.match(stem)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, None

# ───────────────────────────── main ────────────────────────────────
def main() -> None:
    # load reference profiles
    amplicon_dataset = AmpliconDataset(AMPLICON_DIR)

    batches = defaultdict(list)
    for fname in sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".fsa")):
        stem = Path(fname).stem
        batch_key, _, _ = stem_parts(stem)
        key = batch_key if batch_key else stem
        batches[key].append(fname)

    for batch_key, batch_files in batches.items():
        batch_files.sort()
        if len(batch_files) != 3:
            print(f"Warning: batch '{batch_key}' has {len(batch_files)} file(s); expected 3.")

        print(f"\nProcessing batch {batch_key}: {', '.join(batch_files)}")
        for fsa_file in batch_files:
            stem = Path(fsa_file).stem
            dash_idx = stem.find('-')
            if dash_idx != -1:
                replicate_subfolder = stem[dash_idx + 1:]
            else:
                replicate_subfolder = stem

            batch_key, _, _ = stem_parts(stem)
            if not batch_key:
                batch_key = stem.split('-', 1)[0]

            debug_dir = os.path.join(DEBUG_DIR, batch_key, replicate_subfolder)

            fsa_path = os.path.join(DATA_DIR, fsa_file)
            green_channel, orange_channel = read_fsa(fsa_path)

            calibrator = CapillaryPeakCalibrator(
                orange    = orange_channel,
                green     = green_channel,
                debug_dir = debug_dir,
            )

            # orange_peaks and green_peaks are in form pd.DataFrame(dict(idx, height, bp)) after run()
            orange_peaks = green_peaks = None
            try:
                orange_peaks, green_peaks = calibrator.run()
            except RuntimeError as e:
                print(f"Calibration failed for {fsa_file}: {e}")

            # Save orange and green peaks to debug directory
            os.makedirs(debug_dir, exist_ok=True)
            output_path = os.path.join(debug_dir, f"{stem}_peaks.xlsx")
            with pd.ExcelWriter(output_path) as writer:
                orange_peaks.to_excel(writer, sheet_name="Orange Peaks", index=False)
                green_peaks.to_excel(writer, sheet_name="Green Peaks", index=False)

    print("\nCalibration completed for all batches.\n")

    # ----------------------------- quick dataset check --------------------------
    print("Dataset Test:")
    print(amplicon_dataset["Alteromonas tetraodonis"])
    print(amplicon_dataset.get_profile("Alteromonas tetraodonis", region="16s-23s"))

if __name__ == "__main__":
    main()