import re
import os
from collections import defaultdict
from pathlib import Path

from data_parser import read_fsa
from wave_analysis import CapillaryPeakCalibrator
from amplicon_dataset import AmpliconDataset

DATA_DIR = "./Amplicon Length/Test Data/3.17.25 Promega (Redo) UCx #1 03052025/"
AMPLICON_DIR = "./Amplicon Length/Final Amplicon Profile.xlsx"
DEBUG_DIR = "./debug/"
BATCH_RE = re.compile(r"^(.*?-\d+)[ABC]_")

def main() -> None:
    # ---------------------------------- load profiles ----------------------------------
    amplicon_dataset = AmpliconDataset(AMPLICON_DIR)

    # ----------------------------- collect & group .fsa files ---------------------------
    batches = defaultdict(list)
    for fname in sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".fsa")):
        m = BATCH_RE.match(fname)
        key = m.group(1) if m else Path(fname).stem
        batches[key].append(fname)

    # ----------------------------------- calibration -----------------------------------
    for batch_key, batch_files in batches.items():
        batch_files.sort()
        if len(batch_files) != 3:
            print(f"Warning: batch '{batch_key}' has {len(batch_files)} file(s); expected 3 (A, B, C).")

        print(f"\nProcessing batch {batch_key}: {', '.join(batch_files)}")
        for fsa_file in batch_files:
            fsa_path = os.path.join(DATA_DIR, fsa_file)
            green_channel, orange_channel = read_fsa(fsa_path)

            calibrator = CapillaryPeakCalibrator(
                orange    = orange_channel,
                green     = green_channel,
                debug_dir = os.path.join(DEBUG_DIR, Path(fsa_file).stem),
            )
            try:
                calibrator.run()
            except RuntimeError as e:
                print(f"Calibration failed for {fsa_file}: {e}")

    print("\nCalibration completed for all batches.\n")

    # ----------------------------- quick dataset sanity check --------------------------
    print("Dataset Test:")
    print(amplicon_dataset["Alteromonas tetraodonis"])
    print(amplicon_dataset.get_profile("Alteromonas tetraodonis", region="16s-23s"))

if __name__ == "__main__":
    main()