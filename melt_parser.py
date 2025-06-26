import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

ROOT=Path('Amplicon Length/UCx Melt and Ct Data')

EXCLUSION_NAMES = ['Gblock', 'NTC']
REGION_NAMES = ['16s-23s', '23s-5s', 'Thr-Tyr']

SHEET_DICT={'Amplification': 'qPCR', 'Melt': 'Derivative'}
REGION_DICT={'16s-23s': 'A', '23s-5s': 'B', 'Thr-Tyr': 'C'}

class Melt:
    def __init__(self, melt_path: str | Path, sample_id:int):
        self.path = melt_path
        self.id = sample_id
        self.Cq={}
        self.wave= {'qPCR': {'X':[], 'Y':{}},'Melt': {'X':[], 'Y':{}}, 'Derivative': {'X':[], 'Y':{}}}

    def add_x_axis(self, wave_type:str, x: list[int]):
        self.wave[wave_type]['X'] = x

    def add_y_axis(self, wave_type:str,amplicon: str, data: list[float]):
        self.wave[wave_type]['Y'][amplicon]=data

def find_xlsx_in(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.xlsx"))

def find_column_in_df(df: pd.DataFrame, col_name: str) -> str:
    for col in df.columns:
        if col_name in str(col):
            return col
    return ''

def plot_melt_wave(melt_set: list[Melt], choice='Melt', region_set:list[str]=None, save:bool=False, folder: str | Path = ''):
    """
    melt_set: List of Melt objects containing the melt data
    choice: 'qPCR', 'Melt', or 'Derivative' to specify which curve to plot
    region_set: List of regions of interest to plot, defaults to all regions
    save: If True, saves the plot to the specified folder, otherwise shows the plot
    """
    if region_set is None:
        region_set = REGION_NAMES
    plt.figure(figsize=(16, 10))
    sample_list= [melt.id for melt in melt_set]
    for melt in melt_set:
        for region, data in melt.wave[choice]['Y'].items():
            if region in region_set:
                plt.plot(melt.wave[choice]['X'], data, label=f"{melt.id} ({region})")
    plt.title(f"{melt_set[0].path.name}: {choice} Curve for sample {','.join(map(str,sample_list))}")

    if choice=='qPCR':
        plt.xlabel('Cycle')
    else:
        plt.xlabel("Temperature (°C)")
    plt.ylabel("Fluorescence")
    plt.legend()
    plt.locator_params(axis='x', tight=True)
    plt.locator_params(axis='y', tight=True)
    plt.grid()
    if save:
        save_path = folder / f"{choice}_curve.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def read_melt(melt_path: str | Path, num_samples: int)-> list[Melt]:
    """
    Read a melt xlsx file and return a list of Melt objects.
    """
    sheets=pd.ExcelFile(melt_path,engine='openpyxl').sheet_names[1:]  # Skip the first sheet which is usually metadata

    file_type="CES" in melt_path.name

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
    for i in range(1,num_samples+1):
        melt= Melt(melt_path, i)
        for sheet in sheets:
            x_loc= df_bulk[sheet].columns.tolist()[0]
            if file_type:
                melt.add_x_axis(SHEET_DICT[sheet], df_bulk[sheet][x_loc])
            else:
                melt.add_x_axis(sheet,df_bulk[sheet][x_loc])

            for region in REGION_NAMES:
                if file_type:
                    col=  find_column_in_df(df_bulk[sheet], f"_{i}{REGION_DICT[region]}")
                    melt.add_y_axis(SHEET_DICT[sheet],region,df_bulk[sheet][col])
                else:
                    col = find_column_in_df(df_bulk[sheet], f"{i} ({region})")
                    melt.add_y_axis(sheet, region, df_bulk[sheet][col])
        melt_list.append(melt)
    return melt_list

def main():
    debug_folder= Path("debug_melt")
    debug_folder.mkdir(exist_ok=True)
    melt_file_list=find_xlsx_in(ROOT)

    for file in melt_file_list:
        print(f"Processing {file.name}...")
        file_name=file.name
        sub_path=debug_folder/file_name
        sub_path.mkdir(exist_ok=True)

        melt_set = read_melt(file, num_samples=6)

        plot_melt_wave(melt_set, 'qPCR', save=True, folder=sub_path)
        plot_melt_wave(melt_set, 'Melt', save=True, folder=sub_path)
        plot_melt_wave(melt_set, 'Derivative', save=True, folder=sub_path)

    return 0

if __name__ == "__main__":
    main()