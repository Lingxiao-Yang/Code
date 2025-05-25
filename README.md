# Bacterial Identification through Signal Processing

This project provides tools for analyzing capillary electrophoresis data to identify bacterial species based on their amplicon profiles.

## Overview

The system analyzes FSA (fluorescence sequencer archive) files containing electropherogram data with:

- Orange channel: Size ladder/standard markers
- Green channel: Sample amplicon fragments

By calibrating the sample peaks against the known ladder, we can determine the exact size (in base pairs) of the DNA fragments in the sample, which can then be compared to a reference database for bacterial identification.

## Files

- `main.py` - Main entry point for processing FSA files in batches
- `data_parser.py` - Utilities to read FSA files and visualize signals
- `wave_analysis.py` - Core algorithm for peak detection and size calibration
- `amplicon_dataset.py` - Reference database of bacterial amplicon profiles

## Dependencies

- NumPy
- Pandas
- Matplotlib
- SciPy
- BioPython

## Usage

```python
python main.py
```

This will:

1. Scan the test data directory for FSA files
2. Group them into batches
3. Process each file to calibrate green channel peaks using the orange channel ladder
4. Output calibrated peak data to Excel files in the debug directory

## Parameters in `CapillaryPeakCalibrator`

The `CapillaryPeakCalibrator` class in `wave_analysis.py` contains numerous configurable parameters:

### Signal Processing Parameters

- `smooth_win` (default=5): Window size for signal smoothing; larger values produce smoother signals but may obscure small peaks
- `baseline_win` (default=800): Window size for baseline removal; affects the background correction

### Orange Channel (Ladder) Peak Detection

- `pk_height_orange` (default=200): Minimum height threshold for orange ladder peaks
- `pk_prom_orange` (default=150): Minimum prominence threshold for orange peaks
- `pk_dist_orange` (default=30): Minimum separation distance between orange peaks

### Green Channel (Sample) Peak Detection

- `pk_height_green` (default=1000): Minimum height threshold for green sample peaks
- `pk_prom_green` (default=800): Minimum prominence threshold for green peaks
- `pk_dist_green` (default=80): Minimum separation distance between green peaks

### Peak Calibration Parameters

- `merge_distance` (default=80): Two peaks ≤ this distance apart are considered "close"
- `allowed_bp_gap` (default=10): Maximum allowed gap in base pairs between ladder values for merging close peaks
- `rmse_tol` (default=3.0): Tolerance for interpolation error in the ladder calibration
- `green_min_bp` (default=80): Green peaks with bp < this value are discarded
- `debug_dir`: If provided, diagnostic plots of peak detection will be saved here

## Amplicon Dataset

The AmpliconDataset class loads bacterial reference data from an Excel workbook containing amplicon profiles for:

- 16S-23S rDNA regions
- 23S-5S rDNA regions
- Thr-Tyr intergenic regions

## Data Structure

The code expects FSA files to follow a naming pattern like `XX-NNNA_description.fsa` where:

- `XX-NNN` is the batch identifier
- `A`, `B`, or `C` is the replicate letter
- `description` is additional metadata

## Example

```python
from data_parser import read_fsa
from wave_analysis import CapillaryPeakCalibrator

# Read the FSA file
green_channel, orange_channel = read_fsa("sample.fsa")

# Create calibrator with custom parameters 
calibrator = CapillaryPeakCalibrator(
    orange=orange_channel, 
    green=green_channel,
    pk_height_green=800,  # Adjust to sample sensitivity
    pk_prom_green=600,    # Adjust based on signal-to-noise ratio
    debug_dir="./debug"   # Save diagnostic plots
)

# Run calibration
orange_peaks, green_peaks = calibrator.run()

# Access results
print(green_peaks)  # DataFrame with columns: idx, height, bp
```

## Output

The calibration process generates:

- Excel files with peak data (position, height, base pair size)
- Diagnostic plots showing detected peaks on both channels
