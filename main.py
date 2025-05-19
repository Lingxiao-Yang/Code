import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from data_parser import read_fsa
from wave_analysis import CapillaryPeakCalibrator
from amplicon_dataset import AmpliconDataset

DATA_DIR = "./Amplicon Length/Test Data/3.17.25 Promega (Redo) UCx #1 03052025/"
AMPLICON_DIR = "./Amplicon Length/Final Amplicon Profile.xlsx"
DEBUG_DIR = "./debug/"

def main():
    # Load the amplicon dataset
    amplicon_dataset = AmpliconDataset(AMPLICON_DIR)

    files = os.listdir(DATA_DIR)
    fsa_files = [f for f in files if f.endswith('.fsa')]
    fsa_files.sort()
    
    for fsa_file in fsa_files:
        fsa_path = os.path.join(DATA_DIR, fsa_file)
        green_channel, orange_channel = read_fsa(fsa_path)
        
        # Initialize the calibrator
        calibrator = CapillaryPeakCalibrator(
            orange=orange_channel,
            green=green_channel,
            debug_dir=os.path.join(DEBUG_DIR, Path(fsa_file).stem),
        )
        
        # Calibrate the orange channel
        try:
            calibrator.run()
        except RuntimeError as e:
            print(f"Calibration failed for {fsa_file}: {e}")
            continue
    
    print("Calibration completed for all files.")
    
    print("Dataset Test:")
    print(amplicon_dataset["Mycobacterium simiae"])
    print(amplicon_dataset.get_profile("Mycobacterium simiae", region="16s-23s"))

if __name__ == "__main__":
    main()