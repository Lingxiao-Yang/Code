import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt

ROOT=Path('Amplicon Length/UCx Melt and Ct Data')

EXCLUSION_NAMES = [ 'NTC']
REGION_NAMES = ['16s-23s', '23s-5s', 'Thr-Tyr']

SHEET_DICT={'Amplification': 'qPCR', 'Melt': 'Derivative'}
REGION_DICT={'16s-23s': 'A', '23s-5s': 'B', 'Thr-Tyr': 'C'}

class Melt:
    def __init__(self, name: str, species: str="Unknown"):
        self.name = name
        self.species = species
        self.wave={"RFU":{region: [] for region in REGION_NAMES},
                   "temp": {region: [] for region in REGION_NAMES}
                   }

    def add_wave(self,wave_type:str, region: str, data: list[float]):
        if region in self.wave[wave_type]:
            self.wave[wave_type][region] = data
        else:
            raise ValueError(f"Region {region} not recognized. Valid regions are: {', '.join(self.wave[wave_type].keys())}")

def find_xlsx_in(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.xlsx"))

def find_csv_in(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.csv"))

def find_column_in_df(df: pd.DataFrame, col_name: str) -> str:
    for col in df.columns:
        if col_name in str(col):
            return col
    return ''

def plot_melt_wave(melt_set: list[Melt], wave_type:str,region_set:list[str]=None, save:bool=False, folder: str | Path = ''):
    """
    melt_set: List of Melt objects containing the melt data
    region_set: List of regions of interest to plot, defaults to all regions
    save: If True, saves the plot to the specified folder, otherwise shows the plot
    """
    if region_set is None:
        region_set = REGION_NAMES
    plt.figure(figsize=(16, 10))
    for melt in melt_set:
        for region in region_set:
            plt.plot(melt.wave["temp"][region], melt.wave[wave_type][region], label=f"{melt.name} ({region})")

    plt.title(f"Melt Curves")
    plt.xlabel("Temperature (°C)")
    match wave_type:
        case "RFU":
            plt.ylabel("Raw Fluorescence")
        case "Diff":
            plt.ylabel("Normalized Raw Fluorescence Difference")
        case "derivative":
            plt.ylabel("Derivative (-dF/dT)")
    plt.legend()
    plt.locator_params(axis='x', tight=True)
    plt.locator_params(axis='y', tight=True)
    plt.grid()
    if save:
        save_path = Path(folder) / "melt_curve.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def read_melt_from_xlsx(melt_path: str | Path, num_samples: int):
    """
    Read a melt xlsx file and return a list of Melt objects.
    """
    sheets=["Melt"]

    df_bulk={sheet: pd.read_excel(melt_path, sheet_name=sheet,engine='openpyxl') for sheet in sheets}

    melt_path=melt_path.with_suffix('.xlsx')

    for sheet in sheets:
        columns=[str(col) for col in df_bulk[sheet].columns]
        exclude_list=[col for col in columns if any(exclusion in col for exclusion in EXCLUSION_NAMES)]
        try:
            df_bulk[sheet]= df_bulk[sheet].drop(columns=exclude_list)
        except KeyError as e:
            print(f"Error filtering columns in sheet {sheet} for {melt_path}: {e}")
            continue

    melt_list=[]
    try:
        temp_loc = df_bulk["Melt"].columns.tolist()[0]
    except KeyError as e:
        raise KeyError(f"Error finding temperature column: {e}")

    for i in range(1,num_samples+1):
        melt=Melt(f"{melt_path.stem} Sample {i}", species="Unknown")


        for region in REGION_NAMES:
            col=find_column_in_df(df_bulk["Melt"], f"{i} ({region})")
            try:
                melt.add_wave("temp",region,df_bulk["Melt"][temp_loc])
                melt.add_wave("RFU",region, df_bulk["Melt"][col])
            except KeyError as e:
                print(f"Error adding wave for {region} in sample {i} of {melt_path}: {e}")

        melt_list.append(melt)

    # GBlock
    melt= Melt(f"{melt_path.stem} GBlock", species="reference")
    for region in REGION_NAMES:
        col=find_column_in_df(df_bulk["Melt"], f"Gblock ({region})")
        try:
            melt.add_wave("temp",region,df_bulk["Melt"][temp_loc])
            melt.add_wave("RFU",region, df_bulk["Melt"][col])
        except KeyError as e:
            print(f"Error adding GBlock wave for {region} in {melt_path}: {e}")

    melt_list.append(melt)

    return melt_list

DICT_Well_to_Sample ={
    "B2": "1 (16s-23s)",
    "B3": "2 (16s-23s)",
    "B4": "3 (16s-23s)",
    "B5": "4 (16s-23s)",
    "B6": "5 (16s-23s)",
    "B7": "6 (16s-23s)",
    "B9": "Gblock (16s-23s)",
    "D2": "1 (23s-5s)",
    "D3": "2 (23s-5s)",
    "D4": "3 (23s-5s)",
    "D5": "4 (23s-5s)",
    "D6": "5 (23s-5s)",
    "D7": "6 (23s-5s)",
    "D9": "Gblock (23s-5s)",
    "F2": "1 (Thr-Tyr)",
    "F3": "2 (Thr-Tyr)",
    "F4": "3 (Thr-Tyr)",
    "F5": "4 (Thr-Tyr)",
    "F6": "5 (Thr-Tyr)",
    "F7": "6 (Thr-Tyr)",
    "F9": "Gblock (Thr-Tyr)"
}

WELL_LIST_A=["B2", "B3", "B4", "B5", "B6", "B7", "B9"]
WELL_LIST_B=["D2", "D3", "D4", "D5", "D6", "D7", "D9"]
WELL_LIST_C=["F2", "F3", "F4", "F5", "F6", "F7", "F9"]

def read_melt_from_csv(melt_path: str):
    df=pd.read_csv(melt_path)
    melt_set=[]

    for i in range(len(WELL_LIST_A)):
        melt=Melt(melt_path.stem+DICT_Well_to_Sample[WELL_LIST_A[i]], species="Unknown")
        try:
            melt.add_wave("temp","16s-23s",df['Temperature'])
            melt.add_wave("temp", "23s-5s", df['Temperature'])
            melt.add_wave("temp", "Thr-Tyr", df['Temperature'])
            melt.add_wave("RFU","16s-23s", df[WELL_LIST_A[i]])
            melt.add_wave("RFU","23s-5s", df[WELL_LIST_B[i]])
            melt.add_wave("RFU","Thr-Tyr", df[WELL_LIST_C[i]])
        except KeyError as e:
            print(f"Error adding wave for in {melt_path}: {e}")
            continue
        melt_set.append(melt)
    return melt_set


def main():

    file=Path("Amplicon Length/Melt Raw Fluorescence/3.10.25 UCx Processing w. ATTO565 (#2) -  Melt Curve RFU Results_SYBR.csv")
    melt_set=read_melt_from_csv(file)
    plot_melt_wave(melt_set, "RFU",region_set=["Thr-Tyr"], save=False, folder=ROOT)
    return 0

if __name__ == "__main__":
    main()