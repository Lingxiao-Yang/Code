import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from pathlib import Path

from melt_dataset import MeltDataset, extract_xml
from melt_parser import Melt, plot_melt_wave, find_xlsx_in, read_melt_from_xlsx, find_csv_in,read_melt_from_csv

MELT_ROOT = Path("Amplicon Length/UCx Melt and Ct Data")
RFU_ROOT = Path("Amplicon Length/Melt Raw Fluorescence")


EXCLUSION_NAMES = ['NTC']
REGION_NAMES = ['16s-23s', '23s-5s', 'Thr-Tyr']

SHEET_DICT = {'Amplification': 'qPCR', 'Melt': 'Derivative'}
REGION_DICT = {'16s-23s': 'A', '23s-5s': 'B', 'Thr-Tyr': 'C'}

DATABASE_ROOT = Path("Amplicon Length/Melt Dataset/")
REFERENCE_16s23s_Path = Path("Amplicon Length/Melt Dataset/reference_strand_16s-23s.xml")
REFERENCE_23s5s_Path = Path("Amplicon Length/Melt Dataset/reference_strand_23s-5s.xml")
REFERENCE_ThrTyr_Path = Path("Amplicon Length/Melt Dataset/reference_strand_Thr-Tyr.xml")

class HRMAnalyzer:
    def __init__(self, data:list[Melt],region:str="16s-23s",reference_sample:str="GBlock"):
        self.samples,self.sample_names = self._load_data(data, region)
        self.reference_sample = reference_sample

        self.region= region
        self.derivatives = None
        self.melt_regions = {}
        self.difference_curves = None

    def _load_data(self,data:list[Melt],region:str):
        samples= {}
        sample_names=[]
        temp=None
        for sample in data:
            sample_names.append(sample.name)
            samples[sample.name]= sample.wave["RFU"][region]
            if temp is None:
                try:
                    temp = sample.wave["temp"][region]
                except KeyError:
                    continue

        if temp is None:
            raise ValueError("No temperature data found in the provided samples.")
        samples["Temperature"]= temp
        df=pd.DataFrame(samples)
        df=df.set_index("Temperature").astype(float)

        return df,sample_names


    def process(self, poly_order=2, start_angle=50, end_angle=30, ebs_offset=1.5):
        """
        If processing dataset, skip step 2-4 as data is pre-processed.
        """

        print("Starting HRM analysis...")
        self._calculate_derivative(poly_order=poly_order)
        print("1. Calculated derivatives using Savitzky-Golay filter.")

        self._find_melt_regions(start_angle=start_angle, end_angle=end_angle)
        print("2. Identified melt regions.")

        self._perform_ebs(offset=ebs_offset)
        print("3. Performed Exponential Background Subtraction (EBS).")

        self._normalize_curves()
        print("4. Normalized EBS curves.")

        self._generate_difference_plot()
        print("5. Generated difference plot.")
        print("Analysis complete.")

    def _calculate_derivative(self, poly_order=2):
        temp= self.samples.index
        avg_temp_step = np.mean(np.diff(temp))
        window_length = int(1.0 / avg_temp_step)

        if window_length % 2 == 0:
            window_length += 1

        if window_length >= len(temp):
            window_length = len(temp) - 1 if (len(temp) - 1) % 2 != 0 else len(temp) - 2

        if poly_order >= window_length:
            poly_order = window_length - 1

        derivs = {}
        for name in self.sample_names:
            fluo = self.samples[name].values
            dF_dT = savgol_filter(fluo, window_length, poly_order, deriv=1, delta=avg_temp_step)
            derivs[name] = -dF_dT

        self.derivatives = pd.DataFrame(derivs, index=temp)

    def _find_melt_regions(self, start_angle=50, end_angle=30, window_size=5):
        temp = self.derivatives.index.values

        for sample in self.sample_names:
            deriv_curve = self.derivatives[sample].values

            t_start = np.nan
            for i in range(len(temp) - window_size):
                window_indices = range(i, i + window_size)
                temp_window = temp[window_indices]
                deriv_window = deriv_curve[window_indices]

                slope, _ = np.polyfit(temp_window, deriv_window, 1)
                angle = np.degrees(np.arctan(slope))

                if angle >= start_angle:
                    t_start = temp[i]
                    break

            t_end = np.nan
            for i in range(len(temp) - window_size - 1, -1, -1):
                window_indices = range(i, i + window_size)
                try:
                    temp_window = temp[window_indices]
                    deriv_window = deriv_curve[window_indices]
                except IndexError:
                    continue

                slope, _ = np.polyfit(temp_window, deriv_window, 1)
                angle = -np.degrees(np.arctan(slope))

                if angle >= end_angle:
                    t_end = temp[i + window_size - 1]
                    break

            self.melt_regions[sample] = {'start': t_start, 'end': t_end}

    def _perform_ebs(self, offset=1.5):
        temp = self.samples.index
        ebs_data = {}

        for sample in self.sample_names:
            fluo = self.samples[sample]

            t_start_melt = self.melt_regions[sample]['start']
            t_end_melt = self.melt_regions[sample]['end']

            if np.isnan(t_start_melt) or np.isnan(t_end_melt):
                print(f"Warning: Could not find melt region for {sample}. Skipping EBS.")
                ebs_data[sample] = fluo.values
                continue

            t_l = t_start_melt - offset
            t_r = t_end_melt + offset

            pre_melt_region = fluo[(temp >= t_l) & (temp < t_l + 1.0)]
            post_melt_region = fluo[(temp >= t_r) & (temp < t_r + 1.0)]

            if pre_melt_region.empty or post_melt_region.empty:
                print(f"Warning: Not enough data points for EBS on {sample}. Skipping.")
                ebs_data[sample] = fluo.values
                continue

            def get_slope_at_temp(region, temp_point):
                x = region.index.values
                y = region.values
                beta, ln_k = np.polyfit(x, np.log(y), 1)
                k = np.exp(ln_k)
                slope = k * beta * np.exp(beta * temp_point)
                return slope

            df_dt_l = get_slope_at_temp(pre_melt_region, t_l)
            df_dt_r = get_slope_at_temp(post_melt_region, t_r)

            if df_dt_l == 0 or df_dt_r / df_dt_l <= 0:
                a = 0
            else:
                a = np.log(df_dt_r / df_dt_l) / (t_r - t_l)

            if a == 0:
                C = 0
            else:
                C = df_dt_l / a

            background = C * np.exp(a * (temp - t_l))

            ebs_data[sample] = fluo - background

        self.samples = pd.DataFrame(ebs_data, index=temp)

    def _normalize_curves(self,wave:str="RFU"):
        normalized_data = {}
        for sample in self.sample_names:
            match wave:
                case "RFU":
                    curve = self.samples[sample]
                case "Derivative":
                    curve = self.derivatives[sample]
            min_val = curve.min()
            max_val = curve.max()
            if (max_val - min_val) == 0:
                normalized_data[sample] = curve
            else:
                normalized_curve = 100 * (curve - min_val) / (max_val - min_val)
                normalized_data[sample] = normalized_curve

        match wave:
            case "RFU":
                self.samples = pd.DataFrame(normalized_data, index=self.samples.index)
            case "Derivative":
                self.derivatives = pd.DataFrame(normalized_data, index=self.derivatives.index)

    def _generate_difference_plot(self):
        reference_name=next((name for name in self.sample_names if self.reference_sample in name), None)
        if reference_name is None:
            raise ValueError("Reference sample name not found in the dataset.")
        reference = self.samples[reference_name]

        self.difference_curves = self.samples.subtract(reference, axis=0)

    def plot_derivatives(self, title="Negative First Derivative Curves"):
        if self.derivatives is None:
            print("Derivatives have not been calculated yet. Run analysis first.")
            return
        plt.figure(figsize=(10, 6))
        for sample in self.sample_names:
            plt.plot(self.derivatives.index, self.derivatives[sample], label=sample)
        plt.title(title)
        plt.xlabel("Temperature (°C)")
        plt.ylabel("-dF/dT")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()

    def plot_difference_curves(self, title="Fluorescence Difference Plot"):
        if self.difference_curves is None:
            print("Difference curves have not been generated yet. Run analysis first.")
            return
        plt.figure(figsize=(10, 6))
        for sample in self.sample_names:
            plt.plot(self.difference_curves.index, self.difference_curves[sample], label=sample)
        plt.axhline(0, color='black', linestyle='--')
        plt.title(title)
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Δ Fluorescence")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()

def main():
    # Example use for simulated dataset
    dataset = MeltDataset(DATABASE_ROOT)
    reference=Melt("Simulated Reference", species="GBlock")

    x, y, _, _ = extract_xml(REFERENCE_16s23s_Path, need_indices=False)
    
    reference.add_wave("RFU", "16s-23s", y)
    reference.add_wave("temp", "16s-23s", x)
    x,y, _, _ = extract_xml(REFERENCE_23s5s_Path, need_indices=False)

    reference.add_wave("RFU", "23s-5s", y)
    reference.add_wave("temp", "23s-5s", x)
    x,y, _, _ = extract_xml(REFERENCE_ThrTyr_Path, need_indices=False)

    reference.add_wave("RFU", "Thr-Tyr", y)
    reference.add_wave("temp", "Thr-Tyr", x)

    simulation= [dataset["Escherichia coli"],
                 dataset["Enterococcus faecalis"],
                 dataset["Enterococcus gallinarum"],
                 dataset["Clostridium septicum"],
                 dataset["Mycobacterium marinum"],
                 dataset["Mycolicibacterium phlei"],
                 dataset["Mycobacterium avium"],
                 reference]
    plot_melt_wave(simulation,"RFU", region_set=["16s-23s"], save=False, folder=MELT_ROOT)
    hrm= HRMAnalyzer(simulation, region="16s-23s", reference_sample="Simulated Reference")
    hrm.process(poly_order=2, start_angle=50, end_angle=30, ebs_offset=1.5)
    hrm.plot_derivatives()
    hrm.plot_difference_curves()

    # Example use for excel sheets data
    melt_file_list = find_xlsx_in(MELT_ROOT)
    for file in melt_file_list:
        print(f"Processing {file.name}...")

        melt_set = read_melt_from_xlsx(file, num_samples=6)
        analyzer=HRMAnalyzer(melt_set, region="16s-23s",reference_sample="GBlock")
        analyzer.process(poly_order=2, start_angle=50, end_angle=30, ebs_offset=1.5)
        analyzer.plot_derivatives()
        analyzer.plot_difference_curves()

    # Example use for csv RFU data
    # rfu_file_list=find_csv_in(RFU_ROOT)
    # for file in rfu_file_list:
    #     if "Frozen" in file.name:
    #         continue
    #     print(f"Processing {file.name}...")

    #     melt_set = read_melt_from_csv(file)
    #     analyzer = HRMAnalyzer(melt_set, region="23s-5s", reference_sample="Gblock")
    #     analyzer.process(poly_order=2, start_angle=50, end_angle=30, ebs_offset=1.5)
    #     analyzer.plot_derivatives()
    #     analyzer.plot_difference_curves()


if __name__== "__main__":
    main()